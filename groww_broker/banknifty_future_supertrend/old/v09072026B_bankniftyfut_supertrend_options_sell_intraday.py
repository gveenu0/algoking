"""
BankNifty Options Intraday — Sell ATM PUT on Bullish ST / Sell ATM CALL on Bearish ST
======================================================================================
Instrument : BankNifty Monthly Expiry ATM CE / PE (NSE FNO)
Quantity   : 1 lot per direction (CE and PE are independent)
Supertrend : length=20, factor=1.8  on 1-minute candles of BANKNIFTY underlying

PUT SELL logic  (Bullish signal)
  Entry  : SELL ATM PUT  when Supertrend flips BULLISH (bearish → bullish)
           (If already bullish at startup, enters immediately)
  Exit   : BUY  ATM PUT  when Supertrend turns BEARISH  AND  |option_LTP − sell_price| >= 111 pts

CALL SELL logic  (Bearish signal)
  Entry  : SELL ATM CALL when Supertrend flips BEARISH (bullish → bearish)
           (If already bearish at startup, enters immediately)
  Exit   : BUY  ATM CALL when Supertrend turns BULLISH  AND  |option_LTP − sell_price| >= 111 pts

Entry window  : 09:23  15:05 IST  (no new entries outside this window)
Force close   : 15:08 IST  (both legs closed regardless of P&L)

Notes:
  • ATM strike is determined from the BANKNIFTY underlying LTP, rounded to nearest 100.
  • Monthly expiry = the furthest-dated contract expiring in the current calendar month.
    If none remain this month, uses the next month's monthly expiry.
  • Both legs (CE and PE) run independently; having a CE trade does NOT block PE trade.
  • Each transaction is logged with the execution price.
"""

import os
import csv
import time
import socket
import logging
import tempfile
import warnings
from collections import deque
import pyotp
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from growwapi import GrowwAPI

try:
    from growwapi.groww.exceptions import GrowwAPIRateLimitException
except ImportError:  # pragma: no cover - defensive against SDK path changes
    class GrowwAPIRateLimitException(Exception):
        """Fallback stand-in if the SDK's exception module path changes.
        Real rate-limit errors are still caught via message-text matching
        in _is_rate_limit_error() below."""
        pass

try:
    from growwapi.groww.exceptions import GrowwAPITimeoutException
except ImportError:  # pragma: no cover - defensive against SDK path changes
    class GrowwAPITimeoutException(Exception):
        """Fallback stand-in if the SDK's exception module path changes.
        Real timeout errors are still caught via type/message matching
        in _is_transient_network_error() below."""
        pass


# ─── NETWORK TIMEOUT SAFETY NET ───────────────────────────────────────────────
# The growwapi SDK builds its own `requests` calls internally and does NOT
# expose a way to pass a connect/read timeout in — confirmed by production
# tracebacks showing "connect timeout=None". Without a bound, a dropped/slow
# connection to api.groww.in can hang for the OS-level TCP timeout (~110s on
# Linux, errno 110) before failing, which is far too long for a strategy that
# needs to check exits every minute.
#
# socket.setdefaulttimeout() is a process-wide safety net: any socket that
# doesn't already have an explicit timeout set (which is the case for the
# SDK's internal requests session) will now raise `socket.timeout` after this
# many seconds instead of hanging indefinitely. This must be set once, before
# any network calls are made.
NETWORK_TIMEOUT_SECONDS = 15
socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)


# ─── USER CONFIGURATION ──────────────────────────────────────────────────────
TOTP_API_KEY = "TOTP_API_KEY"   # Replace with your Groww TOTP token
TOTP_SECRET  = "TOTP_SECRET"    # Replace with your Groww TOTP secret

# Supertrend settings
ST_LENGTH = 20
ST_FACTOR = 1.8

# Trade settings
QUANTITY              = 1    # lots per direction
EXIT_POINTS_THRESHOLD = 111  # points; exit when |option_LTP − sell_price| >= this

UNDERLYING = "BANKNIFTY"
STRIKE_STEP = 100  # BankNifty option strike interval

# Candle data — we compute ST on BANKNIFTY index (NSE CASH), not on the option itself
# The underlying trading symbol used for the index is:
BANKNIFTY_INDEX_SYMBOL = "NIFTY BANK"  # NSE CASH segment

# Time gates (IST, HH:MM strings)
ENTRY_START_TIME = "09:23"
ENTRY_END_TIME   = "15:05"
SQUARE_OFF_TIME  = "15:08"
MARKET_OPEN      = "09:15"
MARKET_CLOSE     = "15:30"

LOOKBACK_BARS = 120  # number of 1-min candles to keep in rolling buffer

# How many CALENDAR days back to look when seeding the candle buffer at
# startup, so Supertrend is already warmed up with prior-session candles
# instead of waiting ~25 min into a fresh session for enough bars to build up.
SEED_LOOKBACK_DAYS = 5

IST = timezone(timedelta(hours=5, minutes=30))

# Minimum acceptable option premium for entry.
# If ATM price < this, walk ITM (one strike at a time) until premium >= threshold.
MIN_OPTION_PRICE = 400


# ─── LOGGING ─────────────────────────────────────────────────────────────────
# Everything still prints to stdout (visible in Groww Cloud's live log view),
# but is ALSO written to a dated file under LOG_DIR so you have a persistent,
# downloadable record even after the cloud console scrolls/refreshes/restarts.
def _resolve_log_dir() -> str:
    """
    Pick a writable directory for the log file / trade journal.

    Some cloud deployments run the code from a read-only directory (e.g.
    Groww Cloud runs from /tmp/code), so a plain relative "logs" folder can
    fail with PermissionError even though the box has plenty of writable
    space elsewhere. Try, in order:
      1. LOG_DIR env var, if set (lets you pin an exact path per deployment)
      2. "./logs" (relative to cwd — works fine for local runs)
      3. <system temp dir>/logs (always writable, used as the safe fallback)
    """
    candidates = []
    env_dir = os.environ.get("LOG_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.append("logs")
    candidates.append(os.path.join(tempfile.gettempdir(), "logs"))

    last_exc = None
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            return candidate
        except OSError as e:
            last_exc = e
            continue
    # Unreachable in practice (tempfile.gettempdir() is always writable),
    # but re-raise rather than silently logging nowhere if it ever happens.
    raise last_exc


LOG_DIR = _resolve_log_dir()
_LOG_FILE_PATH = os.path.join(
    LOG_DIR, f"strategy_{datetime.now(IST).strftime('%Y-%m-%d')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
log.info(f"Logging to console and to file: {_LOG_FILE_PATH}")

# ─── TRADE JOURNAL (CSV) ──────────────────────────────────────────────────────
# A separate, structured record of every ENTRY/EXIT — one row per fill — so
# you can monitor trades (or open the file in Excel) without scrolling
# through the verbose text log. Lives alongside the log file under LOG_DIR.
TRADE_JOURNAL_PATH = os.path.join(LOG_DIR, "trade_journal.csv")
TRADE_JOURNAL_FIELDS = [
    "timestamp",        # IST execution time this row was logged
    "leg",              # "PUT SELL" or "CALL SELL"
    "action",           # "ENTRY" or "EXIT"
    "symbol",           # option trading symbol
    "quantity",         # total quantity (lots x lot_size)
    "price",            # execution/fill price (or LTP fallback if unconfirmed)
    "fill_confirmed",   # True/False — False means price is an LTP estimate
    "order_id",         # Groww order id
    "pnl",              # per-trade P&L for EXIT rows (blank for ENTRY)
    "reason",           # e.g. "ST flip BULLISH", "ST flip BEARISH + threshold"
]


def log_trade(
    leg: str,
    action: str,
    symbol: str,
    quantity: int,
    price: float | None,
    fill_confirmed: bool,
    order_id: str,
    reason: str,
    pnl: float | None = None,
    when: datetime | None = None,
) -> None:
    """Append one row to the trade journal CSV (writes the header on first use)."""
    ts = (when or datetime.now(IST)).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp":      ts,
        "leg":            leg,
        "action":         action,
        "symbol":         symbol,
        "quantity":       quantity,
        "price":          f"{price:.2f}" if price is not None else "",
        "fill_confirmed": fill_confirmed,
        "order_id":       order_id,
        "pnl":            f"{pnl:.2f}" if pnl is not None else "",
        "reason":         reason,
    }
    try:
        file_exists = os.path.isfile(TRADE_JOURNAL_PATH)
        with open(TRADE_JOURNAL_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_JOURNAL_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        log.error(f"Failed to write trade journal row ({leg} {action}): {e}")


# ─── RATE LIMITING ────────────────────────────────────────────────────────────
# Per Groww's published limits (groww.in/trade-api/docs/python-sdk#rate-limits):
#   Orders      (create/modify/cancel)                         : 10/sec, 250/min
#   Live Data   (quote, LTP, OHLC — historical candles as well): 10/sec, 300/min
#   Non Trading (order status/list, trades, positions, margin) : 20/sec, 500/min
# Limits are shared across every API within a type, so a burst on one call
# (e.g. LTP) can also throttle a different call in the same bucket (e.g.
# historical candles). We pace ourselves comfortably under these ceilings
# so we essentially never trigger GrowwAPIRateLimitException in the first
# place, and we still handle it gracefully (with backoff) if it ever fires
# anyway — e.g. because another process/device shares this same account.
RATE_LIMITS = {
    "orders":      {"per_sec": 10, "per_min": 250},
    "live_data":   {"per_sec": 10, "per_min": 300},  # incl. historical candles
    "non_trading": {"per_sec": 20, "per_min": 500},
}
RATE_LIMIT_SAFETY_MARGIN = 0.7  # only ever use ~70% of the documented ceiling

_rate_limit_history: dict[str, deque] = {k: deque() for k in RATE_LIMITS}


def _throttle(category: str) -> None:
    """Block just long enough to stay under the safety-margined per-second
    and per-minute limits for `category`, based on this process's own
    rolling call history. Purely client-side pacing — doesn't talk to Groww."""
    limits      = RATE_LIMITS[category]
    history     = _rate_limit_history[category]
    max_per_sec = max(1, int(limits["per_sec"] * RATE_LIMIT_SAFETY_MARGIN))
    max_per_min = max(1, int(limits["per_min"] * RATE_LIMIT_SAFETY_MARGIN))

    while True:
        now = time.monotonic()
        while history and now - history[0] > 60:
            history.popleft()

        recent_1s = [t for t in history if now - t <= 1.0]
        if len(recent_1s) < max_per_sec and len(history) < max_per_min:
            history.append(now)
            return

        if len(recent_1s) >= max_per_sec:
            wait = 1.0 - (now - recent_1s[0]) + 0.01
        else:
            wait = 60.0 - (now - history[0]) + 0.01
        time.sleep(max(0.01, wait))


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, GrowwAPIRateLimitException):
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "too many requests" in msg or "429" in msg


# Fixed backoff schedule (seconds) for transient network/timeout errors.
# Deliberately short and fixed (not exponential) — these are brief connection
# blips to api.groww.in, not sustained outages, so we want to recover fast
# rather than back off aggressively. If all retries are exhausted, the error
# still propagates up to run_strategy()'s top-level handler as a final
# safety net (60s sleep + resume next cycle).
NETWORK_RETRY_DELAYS = [2, 5, 10]


def _is_transient_network_error(exc: Exception) -> bool:
    """
    True for connection/timeout failures to api.groww.in that are almost
    always transient (a brief network blip) rather than a fault in the
    request itself — e.g. the errors seen in production:
      - growwapi.groww.exceptions.GrowwAPITimeoutException
      - requests.exceptions.ConnectTimeout / ReadTimeout / ConnectionError
      - urllib3.exceptions.MaxRetryError / ConnectTimeoutError (wrapped above)
      - socket.timeout (raised by the NETWORK_TIMEOUT_SECONDS safety net)
    """
    if isinstance(exc, GrowwAPITimeoutException):
        return True
    if isinstance(exc, (
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        socket.timeout,
        TimeoutError,
    )):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in (
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "max retries exceeded",
        "failed to establish a new connection",
        "connection refused",
    ))


def _call(category: str, fn, *args, **kwargs):
    """
    Throttled, fault-tolerant wrapper around every Groww SDK call.

    1. Paces the call via _throttle() so we stay under Groww's documented
       ceiling for `category` before we even attempt the request.
    2. Retries two distinct classes of *transient* failure, each on its own
       backoff schedule, before propagating to the caller:
         - Rate limit errors      : exponential backoff (1s, 2s, 4s), 3 retries
         - Network/timeout errors : fixed backoff (2s, 5s, 10s), 3 retries
       (e.g. another process throttling the account, or a dropped/slow
       connection to api.groww.in). Any other exception is raised immediately
       — it's not a class of error we know is safe to blindly retry.
    3. If retries are exhausted, the original exception is raised to the
       caller. run_strategy()'s top-level handler is still there as a final
       safety net (logs, sleeps 60s, resumes on the next cycle), so a
       persistent outage still degrades gracefully instead of crashing.
    """
    _throttle(category)
    rate_limit_attempt = 0
    network_attempt = 0

    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit_error(e):
                if rate_limit_attempt >= 3:
                    raise
                wait = 2 ** rate_limit_attempt
                log.warning(
                    f"Groww rate limit hit ({category}, attempt "
                    f"{rate_limit_attempt + 1}/4) — backing off {wait}s: {e}"
                )
                time.sleep(wait)
                rate_limit_attempt += 1
                _throttle(category)
                continue

            if _is_transient_network_error(e):
                if network_attempt >= len(NETWORK_RETRY_DELAYS):
                    log.error(
                        f"Groww API network/timeout error persisted after "
                        f"{len(NETWORK_RETRY_DELAYS)} retries ({category}): {e}"
                    )
                    raise
                wait = NETWORK_RETRY_DELAYS[network_attempt]
                log.warning(
                    f"Network/timeout error calling Groww API ({category}, "
                    f"attempt {network_attempt + 1}/{len(NETWORK_RETRY_DELAYS) + 1}) "
                    f"— retrying in {wait}s: {e}"
                )
                time.sleep(wait)
                network_attempt += 1
                _throttle(category)
                continue

            raise


# ─── AUTHENTICATION ──────────────────────────────────────────────────────────
def authenticate() -> GrowwAPI:
    """Authenticate via TOTP with retry logic."""
    max_retries = 5
    retry_delay = 30

    secret = TOTP_SECRET.replace(" ", "").replace("-", "").upper()

    for attempt in range(1, max_retries + 1):
        try:
            totp = pyotp.TOTP(secret).now()
            access_token = GrowwAPI.get_access_token(api_key=TOTP_API_KEY, totp=totp)
            log.info(f"Authenticated with Groww API (TOTP, attempt {attempt}).")
            return GrowwAPI(access_token)
        except Exception as e:
            log.error(f"Authentication attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                log.info(f"Retrying in {retry_delay}s…")
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Authentication failed after {max_retries} attempts — "
        "check TOTP credentials and network connectivity to api.groww.in."
    )


# ─── INSTRUMENT HELPERS ───────────────────────────────────────────────────────
def get_all_instruments_df(groww: GrowwAPI) -> pd.DataFrame:
    """Return the full instruments dataframe with normalised column types."""
    df = _call("non_trading", groww.get_all_instruments)
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    return df


_instruments_cache: dict = {"df": None, "ts": 0.0}
INSTRUMENTS_CACHE_TTL_SECONDS = 1800  # 30 min — instrument list doesn't change intraday


def get_cached_instruments_df(groww: GrowwAPI) -> pd.DataFrame:
    """
    Reuse the instruments dataframe for INSTRUMENTS_CACHE_TTL_SECONDS instead
    of re-fetching Groww's full instruments file on every PUT/CALL entry
    signal. The instrument list doesn't change intraday, so refetching it on
    every entry was an avoidable extra API call (and extra latency) each
    time — caching it also helps keep total call volume well under the
    Non-Trading rate limit.
    """
    now = time.monotonic()
    stale = (
        _instruments_cache["df"] is None
        or (now - _instruments_cache["ts"]) > INSTRUMENTS_CACHE_TTL_SECONDS
    )
    if stale:
        _instruments_cache["df"] = get_all_instruments_df(groww)
        _instruments_cache["ts"] = now
    return _instruments_cache["df"]


def get_monthly_expiry_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    Find the BankNifty monthly expiry for the current/next month.
    As per the latest NSE circular, BankNifty monthly expiry falls on the
    last Tuesday of the expiry month (changed from last Thursday).
    We pick the latest expiry within the current calendar month whose
    expiry date has not yet passed today.  If none remain this month,
    we use the next month's monthly expiry.
    The actual date is always derived from the live instruments CSV so
    the correct expiry is resolved automatically regardless of day changes.
    """
    today = pd.Timestamp(datetime.now(IST).date())

    mask = (
        (df["exchange"] == "NSE")
        & (df["underlying_symbol"] == UNDERLYING)
        & (df["instrument_type"].str.upper().isin(["CE", "PE"]))
        & (df["segment"].str.upper() == "FNO")
        & (df["expiry_date"] >= today)
    )
    expiry_dates = df[mask]["expiry_date"].dropna().unique()
    expiry_dates = sorted(expiry_dates)

    if not expiry_dates:
        raise RuntimeError("No future BankNifty FNO expiry dates found in instrument CSV.")

    current_month = today.month
    current_year  = today.year

    # Pick expiry dates that fall in the current month
    this_month = [e for e in expiry_dates if e.month == current_month and e.year == current_year]

    if this_month:
        # Latest expiry in current month = monthly expiry (last Tuesday per latest NSE circular)
        chosen = max(this_month)
    else:
        # None left this month — pick the latest in the next calendar month
        next_month = current_month % 12 + 1
        next_year  = current_year + (1 if current_month == 12 else 0)
        next_month_dates = [
            e for e in expiry_dates if e.month == next_month and e.year == next_year
        ]
        if next_month_dates:
            chosen = max(next_month_dates)
        else:
            # Fallback: nearest available expiry
            chosen = expiry_dates[0]
            log.warning(
                f"Could not determine monthly expiry; falling back to nearest: {chosen.date()}"
            )

    log.info(f"Selected BankNifty monthly expiry: {chosen.date()}")
    return chosen


def get_active_banknifty_fut_symbol(df: pd.DataFrame) -> tuple[str, str]:
    """
    Return (trading_symbol, groww_symbol) of the nearest-expiry BANKNIFTY
    futures contract on NSE FNO.  Used for historical candle fetching
    (SEGMENT_CASH index is not supported by the Groww historical candle API).
    The groww_symbol is required by get_historical_candles().
    """
    today = pd.Timestamp(datetime.now(IST).date())
    mask = (
        (df["exchange"] == "NSE")
        & (df["underlying_symbol"] == UNDERLYING)
        & (df["instrument_type"].str.upper() == "FUT")
        & (df["segment"].str.upper() == "FNO")
        & (df["expiry_date"] >= today)
    )
    active = df[mask].copy().sort_values("expiry_date")
    if active.empty:
        raise RuntimeError(
            "No active BANKNIFTY FUT contract found on NSE. "
            "Check the instruments CSV or market calendar."
        )
    row          = active.iloc[0]
    sym          = str(row["trading_symbol"])
    groww_symbol = str(row["groww_symbol"])
    log.info(
        f"Using BANKNIFTY futures symbol for candle data: {sym} "
        f"(groww_symbol={groww_symbol})"
    )
    return sym, groww_symbol


def get_atm_strike(banknifty_ltp: float) -> int:
    """Round LTP to the nearest STRIKE_STEP to get ATM strike."""
    return int(round(banknifty_ltp / STRIKE_STEP) * STRIKE_STEP)


def find_option_symbol(
    df: pd.DataFrame,
    expiry: pd.Timestamp,
    strike: int,
    option_type: str,   # "CE" or "PE"
) -> tuple[str, int]:
    """
    Return (trading_symbol, lot_size) for the BankNifty option matching
    the given expiry, strike and option type.
    """
    mask = (
        (df["exchange"] == "NSE")
        & (df["underlying_symbol"] == UNDERLYING)
        & (df["instrument_type"].str.upper() == option_type.upper())
        & (df["segment"].str.upper() == "FNO")
        & (df["expiry_date"].dt.normalize() == pd.Timestamp(expiry.date()))
        & (df["strike_price"].astype(float).round().fillna(-1).astype(int) == strike)
    )
    rows = df[mask]
    if rows.empty:
        raise RuntimeError(
            f"No instrument found for BANKNIFTY {option_type} strike={strike} "
            f"expiry={expiry.date()}. ATM strike may not be listed — "
            "check the instruments CSV."
        )
    row      = rows.iloc[0]
    raw_lot  = row.get("lot_size")
    if pd.notna(raw_lot) and raw_lot:
        lot_size = int(raw_lot)
    else:
        lot_size = 30
        log.warning(
            f"lot_size missing/invalid for {option_type} strike={strike} "
            f"expiry={expiry.date()} — falling back to default lot_size=30. "
            "Verify this matches the current exchange lot size before trading."
        )
    sym      = str(row["trading_symbol"])
    log.info(
        f"Resolved {option_type} instrument: {sym}  "
        f"strike={strike}  expiry={expiry.date()}  lot_size={lot_size}"
    )
    return sym, lot_size


def find_option_with_min_price(
    groww: GrowwAPI,
    df: pd.DataFrame,
    expiry: pd.Timestamp,
    atm_strike: int,
    option_type: str,   # "CE" or "PE"
    min_price: float = MIN_OPTION_PRICE,
    max_itm_steps: int = 10,
) -> tuple[str, int, float]:
    """
    Return (trading_symbol, lot_size, ltp) for the cheapest strike whose LTP
    meets the minimum premium threshold.

    Logic:
      1. Try ATM first.
      2. If ATM LTP < min_price, walk ITM one strike at a time:
           PE  → strike increases (higher strike = deeper ITM for puts)
           CE  → strike decreases (lower  strike = deeper ITM for calls)
      3. Returns the first strike whose LTP >= min_price, or the deepest
         ITM option tried if none meets the threshold (with a warning).

    Raises RuntimeError only if no instrument can be found at all.
    """
    direction = +STRIKE_STEP if option_type.upper() == "PE" else -STRIKE_STEP
    strike    = atm_strike
    best_sym  = None
    best_lot  = None
    best_ltp  = None

    for step in range(max_itm_steps + 1):
        label = "ATM" if step == 0 else f"ITM+{step}"
        try:
            sym, lot = find_option_symbol(df, expiry, strike, option_type)
        except RuntimeError as e:
            log.warning(f"  {label} strike={strike} not found in instruments: {e}. Stopping ITM walk.")
            break

        ltp = get_option_ltp(groww, sym)
        log.info(
            f"  {label} {option_type} strike={strike}  LTP=₹{ltp:.2f}  "
            f"(threshold=₹{min_price:.2f})"
        )

        if best_sym is None:
            # Always keep at least the ATM as fallback
            best_sym, best_lot, best_ltp = sym, lot, ltp

        if ltp >= min_price:
            if step > 0:
                log.info(
                    f"  ATM LTP was below ₹{min_price:.2f} — "
                    f"selected {label} {option_type} strike={strike} @ ₹{ltp:.2f}"
                )
            return sym, lot, ltp

        # Not enough premium yet — go deeper ITM
        best_sym, best_lot, best_ltp = sym, lot, ltp
        strike += direction

    # No strike met the threshold
    log.warning(
        f"No {option_type} strike found with LTP >= ₹{min_price:.2f} after "
        f"{max_itm_steps} ITM steps. Using deepest tried: "
        f"strike={strike - direction}  LTP=₹{best_ltp:.2f}. Proceeding anyway."
    )
    return best_sym, best_lot, best_ltp


# ─── BANKNIFTY UNDERLYING LTP ─────────────────────────────────────────────────
def get_banknifty_ltp(groww: GrowwAPI, fut_symbol: str) -> float:
    """Fetch the last traded price of the nearest BANKNIFTY futures contract (NSE FNO).
    The cash index symbol is not reliably supported by the Groww LTP API."""
    key = f"NSE_{fut_symbol}"
    resp = _call(
        "live_data", groww.get_ltp,
        segment=groww.SEGMENT_FNO,
        exchange_trading_symbols=key,
    )
    ltp = resp.get(key)
    if ltp is None:
        raise RuntimeError(f"BankNifty LTP not found for {fut_symbol}. Response: {resp}")
    return float(ltp)


def get_option_ltp(groww: GrowwAPI, trading_symbol: str) -> float:
    """Fetch the last traded price of an FNO option contract."""
    key = f"NSE_{trading_symbol}"
    resp = _call(
        "live_data", groww.get_ltp,
        segment=groww.SEGMENT_FNO,
        exchange_trading_symbols=key,
    )
    ltp = resp.get(key)
    if ltp is None:
        raise RuntimeError(f"Option LTP not found for {trading_symbol}. Response: {resp}")
    return float(ltp)


# ─── CANDLE BUFFER FOR BANKNIFTY INDEX ────────────────────────────────────────
_candle_buffer: pd.DataFrame | None = None


def reset_candle_buffer() -> None:
    global _candle_buffer
    _candle_buffer = None


def fetch_1min_candles(groww: GrowwAPI, groww_symbol: str) -> pd.DataFrame:
    """
    Fetch 1-minute candles for the given BANKNIFTY futures contract (NSE FNO) —
    used for Supertrend.  The cash index is not supported by the Groww
    historical candle API; futures prices track the index closely.

    Uses get_historical_candles() (the non-deprecated replacement for
    get_historical_candle_data()). Candle timestamps are returned as
    "yyyy-MM-ddTHH:mm:ss" strings, NOT epoch milliseconds.

    Incremental fetch after initial seed to stay within API rate limits.

    The initial seed deliberately reaches back several CALENDAR days (not just
    LOOKBACK_BARS minutes) so that if the strategy is started at/soon after
    market open, the buffer is already backfilled with the previous trading
    session's candles. Without this, a 09:15 start would have 0 candles to
    work with and the strategy would need to wait ~25 minutes (ST_LENGTH + 5)
    into the new session before Supertrend could even be computed — which is
    why entries were only appearing after ~09:40. Seeding across days matches
    how a continuous (non-session-reset) 1-min Supertrend behaves on charting
    platforms, so ST is already "hot" and can signal right at open.
    """
    global _candle_buffer

    end_dt = datetime.now(IST).replace(tzinfo=None)
    full_seed_needed = _candle_buffer is None or len(_candle_buffer) < ST_LENGTH + 5

    if full_seed_needed:
        # Look back enough CALENDAR days to guarantee we cross at least one
        # prior trading session, even across a long weekend + holiday.
        # (SEED_LOOKBACK_DAYS safely covers Fri->Mon and most single-day
        # holidays; the API simply returns nothing for closed days, so this
        # is harmless on days when the market was open more recently.)
        start_dt = end_dt - timedelta(days=SEED_LOOKBACK_DAYS)
        log.info(
            f"Seeding BANKNIFTY candle buffer with prior-session history "
            f"(from {start_dt.strftime('%Y-%m-%d %H:%M')})…"
        )
    else:
        start_dt = end_dt - timedelta(minutes=15)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        response = _call(
            "live_data", groww.get_historical_candles,
            exchange=groww.EXCHANGE_NSE,
            segment=groww.SEGMENT_FNO,
            groww_symbol=groww_symbol,
            start_time=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            candle_interval=groww.CANDLE_INTERVAL_MIN_1,
        )

    candles = response.get("candles", [])
    if not candles:
        if _candle_buffer is not None and len(_candle_buffer) >= ST_LENGTH + 5:
            log.warning("No new candles returned — using cached buffer.")
            return _candle_buffer
        raise RuntimeError(
            f"No candle data returned for {groww_symbol} (BANKNIFTY futures). "
            "Market may be closed or the symbol may differ."
        )

    ncols     = len(candles[0])
    base_cols = ["ts", "open", "high", "low", "close", "volume"]
    cols      = base_cols + (["oi"] if ncols >= 7 else [])
    df_new    = pd.DataFrame(candles, columns=cols)
    # Timestamps come back as "yyyy-MM-ddTHH:mm:ss" strings (local IST time),
    # not epoch milliseconds.
    df_new["ts"] = pd.to_datetime(df_new["ts"], errors="coerce")

    if _candle_buffer is None:
        # Cap to LOOKBACK_BARS even on the multi-day seed fetch — we only need
        # enough trailing bars to warm up Supertrend, not every candle since
        # SEED_LOOKBACK_DAYS ago.
        _candle_buffer = (
            df_new.sort_values("ts").tail(LOOKBACK_BARS).reset_index(drop=True)
        )
    else:
        combined = pd.concat([_candle_buffer, df_new])
        _candle_buffer = (
            combined
            .drop_duplicates(subset=["ts"])
            .sort_values("ts")
            .tail(LOOKBACK_BARS)
            .reset_index(drop=True)
        )

    return _candle_buffer


# ─── SUPERTREND INDICATOR ─────────────────────────────────────────────────────
def compute_supertrend_direction(df: pd.DataFrame, length: int, factor: float) -> int:
    """
    Compute Supertrend with Wilder's ATR.
    Returns +1 (bullish) or -1 (bearish) for the LAST ROW of df.

    Callers must pass a df that already ends on a fully-closed candle (see
    the completed-candle trim in run_strategy) — this function no longer
    assumes the last row is a forming candle and looks at -2 internally.
    """
    n     = len(df)
    high  = df["high"].to_numpy(dtype=np.float64)
    low   = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    # True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )

    # Wilder's ATR
    atr = np.empty(n)
    atr[: length - 1] = np.nan
    atr[length - 1]   = np.mean(tr[:length])
    for i in range(length, n):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length

    # Basic bands
    hl2      = (high + low) / 2.0
    basic_ub = hl2 + factor * atr
    basic_lb = hl2 - factor * atr

    # Final bands & direction
    final_ub  = np.full(n, np.nan)
    final_lb  = np.full(n, np.nan)
    direction = np.ones(n, dtype=np.int8)

    for i in range(length - 1, n):
        if i == length - 1:
            final_ub[i]  = basic_ub[i]
            final_lb[i]  = basic_lb[i]
            direction[i] = 1 if close[i] >= hl2[i] else -1
            continue

        final_ub[i] = (
            basic_ub[i]
            if basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]
            else final_ub[i - 1]
        )
        final_lb[i] = (
            basic_lb[i]
            if basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]
            else final_lb[i - 1]
        )

        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < final_lb[i] else 1
        else:
            direction[i] = 1 if close[i] > final_ub[i] else -1

    return int(direction[-1])


# ─── ORDER HELPERS ────────────────────────────────────────────────────────────
def _place_order(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    price: float,
    label: str,
) -> dict:
    log.info(
        f">>> Placing LIMIT {label}  qty={quantity} x {trading_symbol}  @ ₹{price:.2f}"
    )
    resp = _call(
        "orders", groww.place_order,
        trading_symbol=trading_symbol,
        quantity=quantity,
        validity=groww.VALIDITY_DAY,
        exchange=groww.EXCHANGE_NSE,
        segment=groww.SEGMENT_FNO,
        product=groww.PRODUCT_MIS,
        order_type=groww.ORDER_TYPE_LIMIT,
        transaction_type=transaction_type,
        price=price,
    )
    log.info(f"    {label} order response : {resp}")
    return resp


def _place_limit_order(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    price: float,
    label: str,
) -> dict:
    """Thin wrapper around place_order for a LIMIT order (no logging side effects beyond _place_order)."""
    return _place_order(groww, trading_symbol, transaction_type, quantity, price, label)


def _place_market_order(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    label: str,
) -> dict:
    """
    Place a MARKET order (no price). Used only as a last-resort EXIT fallback
    after all LIMIT escalation offsets fail to fill — see
    EXIT_MARKET_FALLBACK_ENABLED / _execute_with_escalation.

    NOTE: an earlier comment in this file says MARKET orders weren't
    accepted for this product on this account/segment, which is why
    entries/exits normally use escalating LIMIT orders instead. If Groww
    rejects this call, the caller (_execute_with_escalation) catches it and
    falls back to the old behaviour of leaving the last LIMIT order
    resting / unconfirmed rather than crashing.
    """
    log.info(f">>> Placing MARKET {label}  qty={quantity} x {trading_symbol}")
    resp = _call(
        "orders", groww.place_order,
        trading_symbol=trading_symbol,
        quantity=quantity,
        validity=groww.VALIDITY_DAY,
        exchange=groww.EXCHANGE_NSE,
        segment=groww.SEGMENT_FNO,
        product=groww.PRODUCT_MIS,
        order_type=groww.ORDER_TYPE_MARKET,
        transaction_type=transaction_type,
    )
    log.info(f"    {label} order response : {resp}")
    return resp


# Escalating price offsets (₹) tried in order until the order fills.
# Groww's API does not support MARKET orders for this product, so we widen
# the LIMIT price step by step to chase a fill without crossing the full
# spread blindly on the first attempt.
ESCALATION_OFFSETS = (2, 5, 10)
FILL_WAIT_RETRIES  = 6     # polls per offset attempt
FILL_WAIT_INTERVAL = 1.0   # seconds between polls

# 4th-retry fallback (EXIT ONLY): if all 3 LIMIT offsets above fail to fill,
# cancel the resting order and fire one MARKET order instead of leaving the
# exit unconfirmed. Entries (sell_option) intentionally do NOT use this —
# only exits (buy_to_cover_option) do, since getting OUT of a short matters
# more than getting a good price.
EXIT_MARKET_FALLBACK_ENABLED = True
MARKET_FALLBACK_POLL_RETRIES  = 10    # polls after the market order is placed
MARKET_FALLBACK_POLL_INTERVAL = 1.0   # seconds between polls


def _get_filled_quantity(detail: dict) -> float:
    """
    Best-effort extraction of how much quantity has actually executed on an
    order, checking the field names Groww's order-detail response is known
    or likely to use. Returns 0.0 if none of these fields are present —
    we'd rather under-detect a partial fill than crash on an unexpected
    response schema.
    """
    for key in ("filled_quantity", "quantity_filled", "executed_quantity", "filled_qty"):
        val = detail.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return 0.0


def _market_fallback(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    label: str,
    resting_order_id: str,
) -> tuple[str, float | None]:
    """
    The 4th retry, used only when _execute_with_escalation is called with
    use_market_fallback=True (i.e. exits) and all 3 LIMIT offsets failed to
    fill. Cancels the still-resting LIMIT order from the last offset, then
    places ONE MARKET order and polls up to MARKET_FALLBACK_POLL_RETRIES
    times for a fill.

    Always returns a (order_id, fill_price_or_None) tuple — same contract as
    _execute_with_escalation — so callers don't need special-casing. A
    fill_price of None means "treat as PENDING/UNCONFIRMED, verify on Groww",
    exactly as with the existing LIMIT-only behavior.
    """
    FILLED_STATUSES = ("EXECUTED", "COMPLETED", "DELIVERY_AWAITED")

    # Re-check the resting LIMIT order first — it may have filled in the
    # moments since the last poll, in which case there's nothing to cancel.
    try:
        detail = _call(
            "non_trading", groww.get_order_detail,
            groww_order_id=resting_order_id, segment=groww.SEGMENT_FNO,
        )
        if detail.get("order_status") in FILLED_STATUSES:
            fill_price = float(detail.get("average_fill_price") or 0)
            if fill_price:
                log.info(
                    f"{label}: resting order {resting_order_id} filled right "
                    "before the MARKET fallback — using this fill instead."
                )
                return resting_order_id, fill_price
        if _get_filled_quantity(detail) > 0:
            log.warning(
                f"{label}: resting order {resting_order_id} shows a PARTIAL "
                "fill right before the MARKET fallback. NOT cancelling or "
                "placing a MARKET order on top of it. VERIFY ACTUAL POSITION ON GROWW."
            )
            return resting_order_id, None
    except Exception as e:
        log.warning(f"{label}: pre-cancel check failed for {resting_order_id}: {e}")

    # Cancel the resting LIMIT order before placing the MARKET order — never
    # want two live orders for the same exit at once.
    try:
        _call(
            "orders", groww.cancel_order,
            groww_order_id=resting_order_id, segment=groww.SEGMENT_FNO,
        )
        log.info(f"{label}: cancelled resting order {resting_order_id} ahead of MARKET fallback.")
    except Exception as e:
        log.warning(
            f"{label}: failed to cancel {resting_order_id} before the MARKET "
            f"fallback: {e}. Not placing a MARKET order on top of an order "
            "whose state is uncertain. Treating as PENDING/UNCONFIRMED. "
            "VERIFY ACTUAL POSITION ON GROWW."
        )
        return resting_order_id, None

    # Place the MARKET order.
    try:
        resp = _place_market_order(
            groww, trading_symbol, transaction_type, quantity, f"{label} (MARKET fallback)"
        )
        market_order_id = resp.get("groww_order_id", "")
        if not market_order_id:
            log.warning(
                f"{label}: MARKET fallback place_order returned no groww_order_id. "
                f"Response: {resp}. Original LIMIT order {resting_order_id} was "
                "already cancelled. VERIFY ACTUAL POSITION ON GROWW."
            )
            return resting_order_id, None
    except Exception as e:
        log.error(
            f"{label}: MARKET fallback order placement failed: {e}. Original "
            f"LIMIT order {resting_order_id} was already cancelled — this "
            "product/segment may not support MARKET orders. VERIFY ACTUAL "
            "POSITION ON GROWW.",
            exc_info=True,
        )
        return resting_order_id, None

    # Poll the MARKET order for a fill.
    for _ in range(MARKET_FALLBACK_POLL_RETRIES):
        try:
            detail = _call(
                "non_trading", groww.get_order_detail,
                groww_order_id=market_order_id, segment=groww.SEGMENT_FNO,
            )
            status = detail.get("order_status")
            if status in FILLED_STATUSES:
                fill_price = float(detail.get("average_fill_price") or 0)
                if fill_price:
                    log.info(
                        f"{label}: MARKET fallback order {market_order_id} filled "
                        f"avg_fill_price=₹{fill_price:.2f}  status={status}"
                    )
                    return market_order_id, fill_price
            if status in ("REJECTED", "FAILED", "CANCELLED"):
                log.warning(
                    f"{label}: MARKET fallback order {market_order_id} ended "
                    f"status={status} without a confirmed fill. Original LIMIT "
                    "order was already cancelled. VERIFY ACTUAL POSITION ON GROWW."
                )
                return market_order_id, None
            filled_qty = _get_filled_quantity(detail)
            if filled_qty > 0:
                log.warning(
                    f"{label}: MARKET fallback order {market_order_id} shows a "
                    f"PARTIAL fill ({filled_qty}/{quantity}). Returning as "
                    "PENDING/UNCONFIRMED. VERIFY ACTUAL POSITION ON GROWW."
                )
                return market_order_id, None
        except Exception as e:
            log.warning(f"{label}: error polling MARKET fallback order {market_order_id}: {e}")
        time.sleep(MARKET_FALLBACK_POLL_INTERVAL)

    log.warning(
        f"{label}: MARKET fallback order {market_order_id} still not confirmed "
        f"filled after {MARKET_FALLBACK_POLL_RETRIES} polls. Treating as "
        "PENDING/UNCONFIRMED. Manual check on Groww recommended."
    )
    return market_order_id, None


def _execute_with_escalation(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    side: str,          # "SELL" or "BUY"
    label: str,
    use_market_fallback: bool = False,
) -> tuple[str, float | None]:
    """
    Place a LIMIT order, escalating the price offset from LTP across
    ESCALATION_OFFSETS until it fills or all offsets are exhausted.

    For SELL: price = LTP - offset (sell into the bid, more aggressive with larger offset)
    For BUY:  price = LTP + offset (buy through the ask, more aggressive with larger offset)

    On each attempt:
      - place the LIMIT order
      - poll get_order_detail up to FILL_WAIT_RETRIES times
      - if filled (EXECUTED/COMPLETED/DELIVERY_AWAITED with average_fill_price), return (order_id, fill_price)
      - else cancel the resting order and retry with the next, wider offset

    If the final (widest) offset also doesn't fill:
      - if use_market_fallback is True and EXIT_MARKET_FALLBACK_ENABLED (a 4th
        retry, intended for EXITS only): cancel the resting LIMIT order and
        place ONE MARKET order, polling MARKET_FALLBACK_POLL_RETRIES times.
        If that fills, return (order_id, fill_price). If it doesn't fill, is
        rejected, or errors (e.g. MARKET orders unsupported for this
        product), fall back to the same "leave resting / unconfirmed"
        behaviour described below.
      - otherwise (or if the market fallback itself didn't resolve things),
        the order is LEFT RESTING (not cancelled) and (order_id, None) is
        returned — caller must treat the position as "order pending, fill
        unconfirmed" rather than assuming it's flat.

    Safety behavior:
      - If ANY exception occurs AFTER an order has already been placed
        (i.e. we have an order_id from this call), we stop immediately and
        return that order as (order_id, None) — PENDING/UNCONFIRMED —
        instead of letting the exception propagate. Callers rely on getting
        a tuple back to record the position; if we let the exception
        escape here, the caller loses all record of an order that may
        already be live on the exchange, and the next signal could place a
        second, duplicate order on top of it. If NO order has been placed
        yet, the exception is safe to re-raise (nothing to lose track of).
      - If polling shows a PARTIAL fill (some quantity executed but the
        order isn't in a terminal filled/rejected/cancelled status), we
        halt escalation entirely rather than cancelling and placing a new
        order at a wider offset. Cancelling a partially filled order only
        cancels the REMAINING quantity — escalating on top of it would
        risk ending up with two separate positions instead of one.
      - After a successful cancel, we re-check the order once more before
        moving to the next offset, to catch the race where the exchange
        fills the order right at (or just before) the cancel takes effect.
    """
    FILLED_STATUSES = ("EXECUTED", "COMPLETED", "DELIVERY_AWAITED")
    last_order_id = ""

    for i, offset in enumerate(ESCALATION_OFFSETS):
        is_last_offset = (i == len(ESCALATION_OFFSETS) - 1)
        try:
            ltp = get_option_ltp(groww, trading_symbol)
            if side == "SELL":
                price = round(max(0.05, ltp - offset), 2)
            else:
                price = round(ltp + offset, 2)

            resp = _place_limit_order(
                groww, trading_symbol, transaction_type, quantity, price,
                f"{label} (offset=₹{offset})"
            )
            order_id = resp.get("groww_order_id", "")
            last_order_id = order_id or last_order_id

            if not order_id:
                log.warning(f"{label}: place_order returned no groww_order_id. Response: {resp}")
                continue

            partial_fill_seen = False
            for _ in range(FILL_WAIT_RETRIES):
                try:
                    detail = _call(
                        "non_trading", groww.get_order_detail,
                        groww_order_id=order_id, segment=groww.SEGMENT_FNO,
                    )
                    status = detail.get("order_status")
                    if status in FILLED_STATUSES:
                        fill_price = float(detail.get("average_fill_price") or 0)
                        if fill_price:
                            log.info(
                                f"{label}: filled at offset=₹{offset}  "
                                f"avg_fill_price=₹{fill_price:.2f}  status={status}"
                            )
                            return order_id, fill_price
                    if status in ("REJECTED", "FAILED", "CANCELLED"):
                        log.warning(f"{label}: order {order_id} ended status={status} before fill.")
                        break
                    filled_qty = _get_filled_quantity(detail)
                    if filled_qty > 0:
                        partial_fill_seen = True
                        log.warning(
                            f"{label}: order {order_id} shows a PARTIAL fill "
                            f"({filled_qty}/{quantity}) at offset=₹{offset}, status={status}. "
                            "Halting escalation — will NOT cancel/re-place at a new offset, to "
                            "avoid ending up with a second position on top of this partial one. "
                            "VERIFY ACTUAL POSITION ON GROWW."
                        )
                        break
                except Exception as e:
                    log.warning(f"Error polling order {order_id}: {e}")
                time.sleep(FILL_WAIT_INTERVAL)

            if partial_fill_seen:
                return order_id, None

            if is_last_offset:
                if use_market_fallback and EXIT_MARKET_FALLBACK_ENABLED:
                    log.warning(
                        f"{label}: order {order_id} not filled even at widest offset "
                        f"(₹{offset}). Attempting 4th retry — cancel and place a MARKET order."
                    )
                    return _market_fallback(
                        groww, trading_symbol, transaction_type, quantity, label, order_id
                    )

                log.warning(
                    f"{label}: order {order_id} not filled even at widest offset "
                    f"(₹{offset}). LEAVING ORDER RESTING — fill unconfirmed. "
                    "Manual check on Groww recommended."
                )
                return order_id, None

            try:
                _call(
                    "orders", groww.cancel_order,
                    groww_order_id=order_id, segment=groww.SEGMENT_FNO,
                )
                log.info(f"{label}: order {order_id} not filled at offset=₹{offset} — cancelled, retrying wider.")
            except Exception as e:
                log.warning(
                    f"{label}: failed to cancel unfilled order {order_id} (offset=₹{offset}): {e}. "
                    "Order state is now uncertain — halting escalation instead of placing another "
                    "order on top of it. Treating as PENDING/UNCONFIRMED. "
                    "VERIFY ACTUAL POSITION ON GROWW."
                )
                return order_id, None

            # Re-check right after the cancel ack, in case the exchange
            # filled the order in the brief window before/at cancellation.
            try:
                post_cancel = _call(
                    "non_trading", groww.get_order_detail,
                    groww_order_id=order_id, segment=groww.SEGMENT_FNO,
                )
                if post_cancel.get("order_status") in FILLED_STATUSES:
                    fill_price = float(post_cancel.get("average_fill_price") or 0)
                    if fill_price:
                        log.warning(
                            f"{label}: order {order_id} actually FILLED right at cancel time "
                            f"(avg_fill_price=₹{fill_price:.2f}) — using this fill instead of "
                            "escalating to a new order."
                        )
                        return order_id, fill_price
                post_cancel_qty = _get_filled_quantity(post_cancel)
                if post_cancel_qty > 0:
                    log.warning(
                        f"{label}: order {order_id} shows a PARTIAL fill ({post_cancel_qty}/{quantity}) "
                        "right at cancel time. Halting escalation instead of placing a new order. "
                        "VERIFY ACTUAL POSITION ON GROWW."
                    )
                    return order_id, None
            except Exception as e:
                log.warning(f"{label}: post-cancel check failed for order {order_id}: {e}")

        except Exception as exc:
            if last_order_id:
                log.error(
                    f"{label}: unexpected error after order {last_order_id} was already placed "
                    f"(offset attempt {i + 1}/{len(ESCALATION_OFFSETS)}): {exc}. Returning this "
                    "order as PENDING/UNCONFIRMED instead of retrying with a new order, to avoid "
                    "duplicating a possibly-live position. VERIFY ACTUAL POSITION ON GROWW.",
                    exc_info=True,
                )
                return last_order_id, None
            log.error(
                f"{label}: unexpected error before any order was placed "
                f"(offset attempt {i + 1}/{len(ESCALATION_OFFSETS)}): {exc}. "
                "No order exists yet, so re-raising is safe.",
                exc_info=True,
            )
            raise

    return last_order_id, None


def sell_option(groww: GrowwAPI, trading_symbol: str, quantity: int, label: str) -> tuple[str, float | None]:
    """SELL to open, escalating the LIMIT price offset (₹2 → ₹5 → ₹10) until filled."""
    return _execute_with_escalation(
        groww, trading_symbol, groww.TRANSACTION_TYPE_SELL, quantity, "SELL", f"SELL {label}"
    )


def buy_to_cover_option(groww: GrowwAPI, trading_symbol: str, quantity: int, label: str) -> tuple[str, float | None]:
    """
    BUY to cover a short, escalating the LIMIT price offset (₹2 → ₹5 → ₹10)
    until filled. If all three LIMIT offsets fail, a 4th retry cancels the
    resting order and fires one MARKET order (see EXIT_MARKET_FALLBACK_ENABLED).
    """
    return _execute_with_escalation(
        groww, trading_symbol, groww.TRANSACTION_TYPE_BUY, quantity, "BUY", f"BUY (cover) {label}",
        use_market_fallback=True,
    )


# ─── ORDER FILL PRICE POLLER ─────────────────────────────────────────────────
def _await_fill_price(groww: GrowwAPI, order_id: str, retries: int = 6) -> float | None:
    """
    Poll order detail until the order is filled.
    Returns average_fill_price, or None on timeout (caller falls back to LTP).

    NOTE: get_order_status() does NOT return average_fill_price — only
    get_order_detail() / get_order_list() do. Valid terminal "filled"
    statuses per Groww docs are EXECUTED / COMPLETED / DELIVERY_AWAITED.
    """
    FILLED_STATUSES = ("EXECUTED", "COMPLETED", "DELIVERY_AWAITED")
    for _ in range(retries):
        try:
            detail = _call(
                "non_trading", groww.get_order_detail,
                groww_order_id=order_id,
                segment=groww.SEGMENT_FNO,
            )
            if detail.get("order_status") in FILLED_STATUSES:
                price = float(detail.get("average_fill_price") or 0)
                if price:
                    return price
        except Exception as e:
            log.warning(f"Error polling order {order_id}: {e}")
        time.sleep(0.5)
    log.warning(
        f"Order {order_id} not filled after {retries} polls — using LTP as fallback."
    )
    return None


# ─── MARKET STATUS ────────────────────────────────────────────────────────────
def market_status(now_hm: str) -> str:
    """Returns 'CLOSED', 'SQUAREOFF', or 'OPEN'."""
    if now_hm >= MARKET_CLOSE or now_hm < MARKET_OPEN:
        return "CLOSED"
    if now_hm >= SQUARE_OFF_TIME:
        return "SQUAREOFF"
    return "OPEN"


# ─── FORCE SQUARE-OFF ────────────────────────────────────────────────────────
def square_off_position(
    groww: GrowwAPI,
    pos: dict,
    quantity: int,
    leg_label: str,
    reason: str = "force square-off",
) -> dict:
    """
    Buy-to-cover a short option position.
    pos dict: {"symbol": str, "entry_price": float, "order_id": str}

    Returns a trade record dict suitable for the trade journal:
      {"leg": leg_label, "symbol": ..., "quantity": ...,
       "entry_price": ..., "exit_price": ..., "exit_order_id": ...,
       "exit_fill_confirmed": bool, "pnl": float | None, "reason": ...}
    """
    symbol      = pos["symbol"]
    entry_price = pos.get("entry_price", 0.0)
    entry_time  = pos.get("entry_time")
    entry_str   = f"₹{entry_price:.2f}" if entry_price else "unknown"
    log.warning(f"[{reason}] Covering short {leg_label}  entry={entry_str}  symbol={symbol}")

    order_id, fill_price = buy_to_cover_option(groww, symbol, quantity, leg_label)
    exit_time            = datetime.now(IST)
    exit_fill_confirmed  = fill_price is not None
    exit_price           = fill_price if fill_price is not None else get_option_ltp(groww, symbol)

    hold_str = ""
    if entry_time is not None:
        hold_str = f"  held={str(exit_time - entry_time).split('.')[0]}"

    if not exit_fill_confirmed:
        log.warning(
            f"    {leg_label} cover order {order_id} fill UNCONFIRMED — "
            f"using LTP ₹{exit_price:.2f} for logging only. "
            "VERIFY ACTUAL POSITION ON GROWW."
        )

    pnl = (entry_price - exit_price) * quantity if entry_price else None

    if entry_price:
        log.info(
            f"[TRADE] {leg_label} EXIT  symbol={symbol}  qty={quantity}  "
            f"entry=₹{entry_price:.2f}  exit=₹{exit_price:.2f}  "
            f"time={exit_time:%Y-%m-%d %H:%M:%S}{hold_str}  "
            f"P&L=₹{pnl:.2f}  fill_confirmed={exit_fill_confirmed}  "
            f"order_id={order_id}  reason={reason}"
        )
    else:
        log.info(
            f"[TRADE] {leg_label} EXIT  symbol={symbol}  qty={quantity}  "
            f"exit=₹{exit_price:.2f}  time={exit_time:%Y-%m-%d %H:%M:%S}{hold_str}  "
            f"P&L=N/A  fill_confirmed={exit_fill_confirmed}  "
            f"order_id={order_id}  reason={reason}"
        )

    log_trade(
        leg=leg_label, action="EXIT", symbol=symbol, quantity=quantity,
        price=exit_price, fill_confirmed=exit_fill_confirmed, order_id=order_id,
        reason=reason, pnl=pnl, when=exit_time,
    )

    return {
        "leg":                  leg_label,
        "symbol":               symbol,
        "quantity":             quantity,
        "entry_price":          entry_price,
        "entry_time":           entry_time,
        "exit_price":           exit_price,
        "exit_time":            exit_time,
        "exit_order_id":        order_id,
        "exit_fill_confirmed":  exit_fill_confirmed,
        "pnl":                  pnl,
        "reason":               reason,
    }


def attempt_square_off(
    groww: GrowwAPI,
    pos: dict,
    quantity: int,
    leg_label: str,
    reason: str = "force square-off",
) -> dict | None:
    """
    Wraps square_off_position() and decides whether the leg can actually be
    marked flat, instead of the caller blindly clearing its position state.

    Returns:
      - None            → exit fill CONFIRMED. Leg is genuinely flat —
                           caller should set its pos variable to None.
      - a pos dict       → exit fill UNCONFIRMED (order left resting, or an
                           error occurred after the cover order was already
                           placed). The leg is kept OPEN with
                           exit_pending=True so it is never silently
                           "forgotten" while a cover order may still be
                           live — caller should keep this returned dict as
                           the leg's new position instead of clearing it.
                           The main loop's pending-exit resolver checks
                           exit_order_id on each cycle and finalizes the
                           leg (clears it to None) once the cover order's
                           fill is actually confirmed.

    This prevents the earlier bug where a leg was marked flat (pos = None)
    regardless of whether the cover order actually filled — which could
    let a brand-new entry fire on the same leg while the old short was
    still genuinely open on Groww.
    """
    trade = square_off_position(groww, pos, quantity, leg_label, reason=reason)
    if trade["exit_fill_confirmed"]:
        return None

    pos["exit_pending"]  = True
    pos["exit_order_id"] = trade["exit_order_id"]
    pos["exit_reason"]   = reason
    pos["exit_quantity"] = quantity
    log.warning(
        f"{leg_label}: cover order unconfirmed — leg kept OPEN (exit_pending=True) "
        f"instead of being cleared, so it isn't lost or duplicated. Will keep checking "
        f"order {trade['exit_order_id']} until it resolves. VERIFY ACTUAL POSITION ON GROWW."
    )
    return pos


# ─── STRATEGY LOOP ────────────────────────────────────────────────────────────
def run_strategy() -> None:
    """
    Main polling loop — runs every ~60 seconds (aligned to minute boundaries).

    PUT SELL leg (put_pos):
      • Entry  : ST turns BULLISH → SELL ATM PUT of current monthly expiry
      • Exit   : ST turns BEARISH AND |option_LTP − sell_price| >= EXIT_POINTS_THRESHOLD

    CALL SELL leg (call_pos):
      • Entry  : ST turns BEARISH → SELL ATM CALL of current monthly expiry
      • Exit   : ST turns BULLISH AND |option_LTP − sell_price| >= EXIT_POINTS_THRESHOLD

    Both legs are independent.  Force-close at SQUARE_OFF_TIME.
    """
    groww          = authenticate()
    last_auth_time = datetime.now(IST)
    RE_AUTH_INTERVAL = timedelta(hours=2)

    instruments_df = get_cached_instruments_df(groww)
    expiry         = get_monthly_expiry_date(instruments_df)
    fut_symbol, fut_groww_symbol = get_active_banknifty_fut_symbol(instruments_df)

    # State for each leg
    # None → flat; dict → {"symbol": str, "lot_size": int, "entry_price": float, "order_id": str}
    put_pos:  dict | None = None
    call_pos: dict | None = None
    prev_direction: int | None = None

    log.info(
        f"Strategy started.  Underlying: {UNDERLYING}  "
        f"Monthly expiry: {expiry.date()}  "
        f"Supertrend({ST_LENGTH}, {ST_FACTOR})  "
        f"Exit threshold: ±{EXIT_POINTS_THRESHOLD} pts  "
        f"Entry window: {ENTRY_START_TIME}–{ENTRY_END_TIME}  "
        f"Force-close: {SQUARE_OFF_TIME}"
    )

    while True:
        try:
            now_ist    = datetime.now(IST)
            now_ts     = now_ist.strftime("%Y-%m-%d %H:%M:%S")
            now_hm     = now_ist.strftime("%H:%M")
            iter_start = time.monotonic()
            status     = market_status(now_hm)

            # ── Outside market hours ──────────────────────────────────────────
            if status == "CLOSED":
                log.info(f"[{now_ts}] Market CLOSED. Sleeping 60 s…")
                time.sleep(60)
                continue

            # ── Force square-off at SQUARE_OFF_TIME ───────────────────────────
            if status == "SQUAREOFF":
                had_positions = put_pos is not None or call_pos is not None
                if put_pos is not None:
                    try:
                        put_pos = attempt_square_off(
                            groww, put_pos, QUANTITY * put_pos["lot_size"],
                            "PUT SELL", reason="SQUARE-OFF TIME"
                        )
                    except Exception as e:
                        log.error(
                            f"Failed to square off PUT SELL position "
                            f"(symbol={put_pos['symbol']}): {e}. "
                            "MANUAL INTERVENTION REQUIRED — check open positions on Groww."
                        )
                if call_pos is not None:
                    try:
                        call_pos = attempt_square_off(
                            groww, call_pos, QUANTITY * call_pos["lot_size"],
                            "CALL SELL", reason="SQUARE-OFF TIME"
                        )
                    except Exception as e:
                        log.error(
                            f"Failed to square off CALL SELL position "
                            f"(symbol={call_pos['symbol']}): {e}. "
                            "MANUAL INTERVENTION REQUIRED — check open positions on Groww."
                        )
                if put_pos is not None or call_pos is not None:
                    log.warning(
                        "One or more legs could not be confirmed flat at force-close "
                        "(unconfirmed cover order or square-off failure). "
                        "MANUAL INTERVENTION REQUIRED — check open positions on Groww."
                    )
                if not had_positions:
                    log.info(f"[{now_ts}] SQUARE-OFF TIME — no open positions.")
                log.info("Session ended. Exiting strategy.")
                break

            # ── Fetch 1-min candles for BANKNIFTY index ───────────────────────
            df = fetch_1min_candles(groww, fut_groww_symbol)

            # Drop the last row if it's a still-forming (not yet closed) candle.
            # Previously this used a hard-coded assumption (df.iloc[-2] = last
            # completed candle) which could silently be wrong depending on
            # whether the API returns a partial "current" candle or not — if
            # wrong, Supertrend here would be evaluating one candle later/
            # earlier than a TradingView chart (which always plots off closed
            # candles once the bar completes), producing a DIFFERENT trend
            # than what's visible on the chart. We now detect this directly
            # from the candle's own start timestamp instead of guessing.
            last_bar_start = df.iloc[-1]["ts"]           # naive IST, start-of-bar
            now_naive      = now_ist.replace(tzinfo=None)
            if now_naive < last_bar_start + timedelta(minutes=1):
                df = df.iloc[:-1]   # still forming — exclude it

            if len(df) < ST_LENGTH + 5:
                log.warning(
                    f"[{now_ts}] Only {len(df)} completed bars available "
                    f"(need {ST_LENGTH + 5}). Retrying in 60 s…"
                )
                time.sleep(60)
                continue

            if prev_direction is None:
                log.info(
                    f"[DIAG] Last COMPLETED candle used for Supertrend: "
                    f"ts={df.iloc[-1]['ts']}  close={df.iloc[-1]['close']:.2f}  "
                    f"now={now_ist.strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Cross-check this timestamp/close against your TradingView "
                    f"chart's last closed 1-min candle on the SAME symbol "
                    f"(BankNifty FUTURES, not the spot index) to confirm alignment."
                )

            # ── Supertrend direction ──────────────────────────────────────────
            curr_direction = compute_supertrend_direction(df, ST_LENGTH, ST_FACTOR)
            last_close     = float(df.iloc[-1]["close"])
            trend_label    = "BULLISH ▲" if curr_direction == 1 else "BEARISH ▼"

            put_label  = (
                f"PUT SELL @ ₹{put_pos['entry_price']:.2f}"
                if put_pos and put_pos.get("entry_price") is not None
                else "PUT: FLAT"
            )
            call_label = (
                f"CALL SELL @ ₹{call_pos['entry_price']:.2f}"
                if call_pos and call_pos.get("entry_price") is not None
                else "CALL: FLAT"
            )
            log.info(
                f"[{now_ts}]  ST={trend_label}  BankNifty Close={last_close:.2f}  "
                f"|  {put_label}  |  {call_label}"
            )

            # First-run: enter immediately based on current Supertrend — no flip needed.
            # We seed prev_direction to the OPPOSITE of curr_direction so that the
            # flip condition (curr != prev) is satisfied right away in this same iteration,
            # triggering an entry without waiting for a trend change.
            if prev_direction is None:
                prev_direction = -1 if curr_direction == 1 else 1
                log.info(
                    f"First run — current Supertrend is {trend_label}. "
                    "Entering trade immediately based on current trend direction "
                    "(no flip required on startup)."
                )

            # ── Recover missing entry prices ──────────────────────────────────
            # A None entry_price means the entry order from _execute_with_escalation
            # was left resting (unfilled even at the widest offset). Poll again here;
            # if it has since filled, record the confirmed fill price. If still
            # unfilled, fall back to LTP for logging but keep flagging it as
            # unconfirmed so square_off_position will warn appropriately.
            for leg_pos, leg_name in [(put_pos, "PUT"), (call_pos, "CALL")]:
                if leg_pos is not None and leg_pos.get("entry_price") is None:
                    try:
                        confirmed = _await_fill_price(groww, leg_pos["order_id"])
                        if confirmed is not None:
                            leg_pos["entry_price"] = confirmed
                            log.info(
                                f"[TRADE] {leg_name} ENTRY_RECOVERED  "
                                f"symbol={leg_pos['symbol']}  fill=₹{confirmed:.2f}  "
                                f"order_id={leg_pos['order_id']}  reason=late fill confirmed"
                            )
                        else:
                            ltp = get_option_ltp(groww, leg_pos["symbol"])
                            leg_pos["entry_price"] = ltp
                            log.warning(
                                f"{leg_name} entry order {leg_pos['order_id']} still unfilled — "
                                f"using LTP ₹{ltp:.2f} as a provisional entry price. "
                                "VERIFY ACTUAL POSITION ON GROWW."
                            )
                    except Exception as e:
                        log.warning(f"Could not recover {leg_name} entry price: {e}")

            # ── Resolve pending exits ──────────────────────────────────────────
            # exit_pending=True means a previous cover order for this leg came
            # back unconfirmed (attempt_square_off kept the leg OPEN instead of
            # clearing it). Poll that specific order here; if it has since
            # filled, finalize the exit (log it, clear the leg). If it's still
            # unresolved, leave the leg open and do NOT let the exit blocks
            # below fire a second cover order on top of it.
            for leg_pos, leg_name, leg_label in (
                (put_pos, "PUT", "PUT SELL"), (call_pos, "CALL", "CALL SELL")
            ):
                if leg_pos is None or not leg_pos.get("exit_pending"):
                    continue
                try:
                    confirmed = _await_fill_price(groww, leg_pos["exit_order_id"])
                except Exception as e:
                    confirmed = None
                    log.warning(f"Could not resolve pending {leg_name} exit: {e}")

                if confirmed is None:
                    log.warning(
                        f"{leg_label}: exit order {leg_pos['exit_order_id']} still "
                        "unresolved — leg remains OPEN; not re-attempting a new cover "
                        "order until this resolves."
                    )
                    continue

                entry_price = leg_pos.get("entry_price")
                exit_qty    = leg_pos.get("exit_quantity", QUANTITY * leg_pos["lot_size"])
                pnl         = (entry_price - confirmed) * exit_qty if entry_price else None
                entry_str   = f"₹{entry_price:.2f}" if entry_price else "unknown"
                pnl_str     = f"₹{pnl:.2f}" if pnl is not None else "N/A"
                log.info(
                    f"[TRADE] {leg_label} EXIT CONFIRMED (was pending)  "
                    f"symbol={leg_pos['symbol']}  qty={exit_qty}  entry={entry_str}  "
                    f"exit=₹{confirmed:.2f}  P&L={pnl_str}  "
                    f"order_id={leg_pos['exit_order_id']}  reason={leg_pos.get('exit_reason')}"
                )
                log_trade(
                    leg=leg_label, action="EXIT", symbol=leg_pos["symbol"], quantity=exit_qty,
                    price=confirmed, fill_confirmed=True, order_id=leg_pos["exit_order_id"],
                    reason=f"{leg_pos.get('exit_reason')} (confirmed on recheck)", pnl=pnl,
                )
                if leg_name == "PUT":
                    put_pos = None
                else:
                    call_pos = None

            entry_allowed = ENTRY_START_TIME <= now_hm <= ENTRY_END_TIME

            # ═══════════════════════════════════════════════════════════════════
            #  PUT SELL LEG  (enter on BULLISH, exit on BEARISH + threshold)
            # ═══════════════════════════════════════════════════════════════════

            # Entry: ST just flipped BULLISH
            if curr_direction == 1 and prev_direction == -1 and put_pos is None:
                if not entry_allowed:
                    log.info(
                        f"ST flipped BULLISH but PUT SELL entry blocked "
                        f"(outside entry window {ENTRY_START_TIME}–{ENTRY_END_TIME})."
                    )
                else:
                    bn_ltp    = get_banknifty_ltp(groww, fut_symbol)
                    atm       = get_atm_strike(bn_ltp)
                    log.info(
                        f"SIGNAL  ST → BULLISH  BankNifty LTP={bn_ltp:.2f}  "
                        f"ATM strike={atm}  →  Selecting PUT (min premium ₹{MIN_OPTION_PRICE})"
                    )
                    # Reuse cached instruments (refreshed at most every
                    # INSTRUMENTS_CACHE_TTL_SECONDS) instead of a full reload
                    instruments_df = get_cached_instruments_df(groww)
                    expiry         = get_monthly_expiry_date(instruments_df)
                    put_sym, put_lot, put_ltp_check = find_option_with_min_price(
                        groww, instruments_df, expiry, atm, "PE"
                    )
                    trade_qty  = QUANTITY * put_lot
                    order_id, fill_price = sell_option(groww, put_sym, trade_qty, "PUT")
                    entry_time = datetime.now(IST)
                    put_pos    = {
                        "symbol":      put_sym,
                        "lot_size":    put_lot,
                        "entry_price": fill_price,   # None if fill unconfirmed
                        "entry_time":  entry_time,
                        "order_id":    order_id,
                    }
                    if fill_price is not None:
                        log.info(
                            f"[TRADE] PUT SELL ENTRY  symbol={put_sym}  strike_ltp_check=₹{put_ltp_check:.2f}  "
                            f"qty={trade_qty}  fill=₹{fill_price:.2f}  time={entry_time:%Y-%m-%d %H:%M:%S}  "
                            f"order_id={order_id}  reason=ST flip BULLISH"
                        )
                    else:
                        log.warning(
                            f"[TRADE] PUT SELL ENTRY (UNCONFIRMED)  symbol={put_sym}  qty={trade_qty}  "
                            f"time={entry_time:%Y-%m-%d %H:%M:%S}  order_id={order_id}  reason=ST flip BULLISH  "
                            "— fill price unknown. VERIFY ACTUAL POSITION ON GROWW."
                        )
                    log_trade(
                        leg="PUT SELL", action="ENTRY", symbol=put_sym, quantity=trade_qty,
                        price=fill_price if fill_price is not None else put_ltp_check,
                        fill_confirmed=fill_price is not None, order_id=order_id,
                        reason="ST flip BULLISH", when=entry_time,
                    )

            # Exit: ST turned BEARISH AND P/L threshold reached
            elif curr_direction == -1 and put_pos is not None:
                if put_pos.get("exit_pending"):
                    log.info(
                        "    PUT SELL: exit already pending confirmation — "
                        "skipping a new cover attempt this cycle."
                    )
                elif put_pos.get("entry_price") is None:
                    # Cannot evaluate threshold without entry price — skip
                    log.warning("PUT SELL: entry price unknown, skipping exit check.")
                else:
                    put_ltp  = get_option_ltp(groww, put_pos["symbol"])
                    diff     = put_ltp - put_pos["entry_price"]   # positive = loss for seller
                    abs_diff = abs(diff)
                    log.info(
                        f"    PUT exit check: option_LTP=₹{put_ltp:.2f}  "
                        f"Entry=₹{put_pos['entry_price']:.2f}  "
                        f"Diff={diff:+.2f} pts  Threshold={EXIT_POINTS_THRESHOLD} pts"
                    )
                    if abs_diff >= EXIT_POINTS_THRESHOLD:
                        log.info(
                            f"SIGNAL  ST BEARISH + |diff|={abs_diff:.2f} >= "
                            f"{EXIT_POINTS_THRESHOLD}  →  Covering PUT short"
                        )
                        trade_qty = QUANTITY * put_pos["lot_size"]
                        put_pos   = attempt_square_off(
                            groww, put_pos, trade_qty, "PUT SELL",
                            reason="ST flip BEARISH + threshold"
                        )
                    else:
                        log.info(
                            f"    Holding PUT short — "
                            f"|diff|={abs_diff:.2f} < {EXIT_POINTS_THRESHOLD} pts threshold."
                        )

            # ═══════════════════════════════════════════════════════════════════
            #  CALL SELL LEG  (enter on BEARISH, exit on BULLISH + threshold)
            # ═══════════════════════════════════════════════════════════════════

            # Entry: ST just flipped BEARISH
            if curr_direction == -1 and prev_direction == 1 and call_pos is None:
                if not entry_allowed:
                    log.info(
                        f"ST flipped BEARISH but CALL SELL entry blocked "
                        f"(outside entry window {ENTRY_START_TIME}–{ENTRY_END_TIME})."
                    )
                else:
                    bn_ltp    = get_banknifty_ltp(groww, fut_symbol)
                    atm       = get_atm_strike(bn_ltp)
                    log.info(
                        f"SIGNAL  ST → BEARISH  BankNifty LTP={bn_ltp:.2f}  "
                        f"ATM strike={atm}  →  Selecting CALL (min premium ₹{MIN_OPTION_PRICE})"
                    )
                    instruments_df = get_cached_instruments_df(groww)
                    expiry         = get_monthly_expiry_date(instruments_df)
                    call_sym, call_lot, call_ltp_check = find_option_with_min_price(
                        groww, instruments_df, expiry, atm, "CE"
                    )
                    trade_qty  = QUANTITY * call_lot
                    order_id, fill_price = sell_option(groww, call_sym, trade_qty, "CALL")
                    entry_time = datetime.now(IST)
                    call_pos   = {
                        "symbol":      call_sym,
                        "lot_size":    call_lot,
                        "entry_price": fill_price,   # None if fill unconfirmed
                        "entry_time":  entry_time,
                        "order_id":    order_id,
                    }
                    if fill_price is not None:
                        log.info(
                            f"[TRADE] CALL SELL ENTRY  symbol={call_sym}  strike_ltp_check=₹{call_ltp_check:.2f}  "
                            f"qty={trade_qty}  fill=₹{fill_price:.2f}  time={entry_time:%Y-%m-%d %H:%M:%S}  "
                            f"order_id={order_id}  reason=ST flip BEARISH"
                        )
                    else:
                        log.warning(
                            f"[TRADE] CALL SELL ENTRY (UNCONFIRMED)  symbol={call_sym}  qty={trade_qty}  "
                            f"time={entry_time:%Y-%m-%d %H:%M:%S}  order_id={order_id}  reason=ST flip BEARISH  "
                            "— fill price unknown. VERIFY ACTUAL POSITION ON GROWW."
                        )
                    log_trade(
                        leg="CALL SELL", action="ENTRY", symbol=call_sym, quantity=trade_qty,
                        price=fill_price if fill_price is not None else call_ltp_check,
                        fill_confirmed=fill_price is not None, order_id=order_id,
                        reason="ST flip BEARISH", when=entry_time,
                    )

            # Exit: ST turned BULLISH AND P/L threshold reached
            elif curr_direction == 1 and call_pos is not None:
                if call_pos.get("exit_pending"):
                    log.info(
                        "    CALL SELL: exit already pending confirmation — "
                        "skipping a new cover attempt this cycle."
                    )
                elif call_pos.get("entry_price") is None:
                    log.warning("CALL SELL: entry price unknown, skipping exit check.")
                else:
                    call_ltp = get_option_ltp(groww, call_pos["symbol"])
                    diff     = call_ltp - call_pos["entry_price"]
                    abs_diff = abs(diff)
                    log.info(
                        f"    CALL exit check: option_LTP=₹{call_ltp:.2f}  "
                        f"Entry=₹{call_pos['entry_price']:.2f}  "
                        f"Diff={diff:+.2f} pts  Threshold={EXIT_POINTS_THRESHOLD} pts"
                    )
                    if abs_diff >= EXIT_POINTS_THRESHOLD:
                        log.info(
                            f"SIGNAL  ST BULLISH + |diff|={abs_diff:.2f} >= "
                            f"{EXIT_POINTS_THRESHOLD}  →  Covering CALL short"
                        )
                        trade_qty = QUANTITY * call_pos["lot_size"]
                        call_pos  = attempt_square_off(
                            groww, call_pos, trade_qty, "CALL SELL",
                            reason="ST flip BULLISH + threshold"
                        )
                    else:
                        log.info(
                            f"    Holding CALL short — "
                            f"|diff|={abs_diff:.2f} < {EXIT_POINTS_THRESHOLD} pts threshold."
                        )

            # ── Advance state and sleep to next minute ────────────────────────
            prev_direction = curr_direction
            elapsed = time.monotonic() - iter_start
            time.sleep(max(0.0, 60.0 - elapsed))

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — squaring off all open positions…")
            if put_pos is not None:
                try:
                    put_pos = attempt_square_off(
                        groww, put_pos, QUANTITY * put_pos["lot_size"],
                        "PUT SELL", reason="KeyboardInterrupt"
                    )
                except Exception as e:
                    log.error(
                        f"Failed to square off PUT SELL position "
                        f"(symbol={put_pos['symbol']}): {e}. "
                        "MANUAL INTERVENTION REQUIRED — check open positions on Groww."
                    )
            if call_pos is not None:
                try:
                    call_pos = attempt_square_off(
                        groww, call_pos, QUANTITY * call_pos["lot_size"],
                        "CALL SELL", reason="KeyboardInterrupt"
                    )
                except Exception as e:
                    log.error(
                        f"Failed to square off CALL SELL position "
                        f"(symbol={call_pos['symbol']}): {e}. "
                        "MANUAL INTERVENTION REQUIRED — check open positions on Groww."
                    )
            log.info("Strategy stopped by user.")
            break

        except Exception as exc:
            if _is_rate_limit_error(exc):
                # Should be rare — _call() already retries 3x with backoff
                # internally before ever propagating. Reaching here means
                # Groww is still throttling us after that, so back off longer.
                log.warning(
                    f"Groww rate limit persisted past internal retries: {exc}. "
                    "Sleeping 30 s before resuming…"
                )
                time.sleep(30)
            elif "forbidden" in str(exc).lower():
                now_ist = datetime.now(IST)
                age     = now_ist - last_auth_time
                if age >= RE_AUTH_INTERVAL:
                    log.warning(
                        f"Access token {age} old — likely expired. Re-authenticating…"
                    )
                    try:
                        groww          = authenticate()
                        last_auth_time = now_ist
                        log.info("Re-authenticated successfully.")
                    except Exception as auth_exc:
                        log.error(f"Re-authentication failed: {auth_exc}. Sleeping 60 s…")
                        time.sleep(60)
                else:
                    log.warning(
                        f"Forbidden error but token only {age} old — "
                        "likely rate-limited. Sleeping 60 s…"
                    )
                    time.sleep(60)
            elif _is_transient_network_error(exc):
                # Should be rare — _call() already retries network/timeout
                # errors 3x (2s/5s/10s backoff) internally before ever
                # propagating. Reaching here means the connection to
                # api.groww.in is down for longer than ~17s straight.
                log.warning(
                    f"Network/timeout error to Groww API persisted past "
                    f"internal retries: {exc}. Sleeping 20 s before resuming…"
                )
                time.sleep(20)
            else:
                log.error(f"Unhandled error: {exc}", exc_info=True)
                log.info("Sleeping 60 s before retrying…")
                time.sleep(60)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_strategy()
