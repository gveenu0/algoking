"""
Gold Mini Futures Intraday Supertrend Strategy -- Groww API
===========================================================

Instrument  : GOLDM (MCX Gold Mini Futures -- 100 g per lot)
Quantity    : 1 lot per direction
Exchange    : MCX / SEGMENT_COMMODITY

Strategy Overview
-----------------
LONG leg  (long_pos):
  * Entry  : Supertrend flips BULLISH  -> BUY  1 lot of Gold Mini futures
  * Exit   : Supertrend flips BEARISH  AND  |LTP - entry_price| >= EXIT_POINTS_THRESHOLD

SHORT leg (short_pos):
  * Entry  : Supertrend flips BEARISH  -> SELL 1 lot of Gold Mini futures
  * Exit   : Supertrend flips BULLISH  AND  |LTP - entry_price| >= EXIT_POINTS_THRESHOLD

Both legs run independently.  Force-close everything at SQUARE_OFF_TIME (23:05 IST),
10 minutes before Groww MIS auto square-off (~23:15 IST).

Production Features (matching BankNifty v20072026A)
----------------------------------------------------
  * Client-side rate limiting (_throttle / _call) -- stays under Groww's documented
    API ceiling (10 calls/sec, 300/min for live data; similar for orders)
  * Global socket timeout (15 s) -- safety net against hung connections
  * Retry logic for rate-limit errors (exponential backoff) and transient network
    errors (fixed backoff 2 s / 5 s / 10 s), both inside _call()
  * Escalating LIMIT order execution (offsets Rs2->Rs5->Rs10) for entries and exits,
    with a MARKET-order fallback (exit only) if all LIMIT offsets fail
  * Trade journal (CSV) -- one row per fill, written alongside the log file
  * File-based logging (dated .log file) in addition to console output
  * Multi-day candle seeding at startup so Supertrend is already hot at market open
  * Instruments CSV caching (30-minute TTL) to avoid redundant API calls
  * Pending-exit resolver: if a cover order comes back unconfirmed, the leg is kept
    OPEN (exit_pending=True) and polled each cycle until confirmed
  * Re-authentication every 2 hours to handle Groww token expiry

Usage
-----
1. Set TOTP_API_KEY and TOTP_SECRET below.
2. Run:  python v20072026A_goldmini_supertrend_intraday.py
3. Ctrl+C squares off all open positions gracefully before exiting.

DISCLAIMER
----------
This script is for educational and research purposes only.
Trading involves significant risk. Always verify orders on your broker platform.
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
except ImportError:
    class GrowwAPIRateLimitException(Exception):
        pass

try:
    from growwapi.groww.exceptions import GrowwAPITimeoutException
except ImportError:
    class GrowwAPITimeoutException(Exception):
        pass


# --- NETWORK TIMEOUT SAFETY NET -----------------------------------------------
# The growwapi SDK does NOT expose a connect/read timeout. Without a bound, a
# dropped connection can hang for the OS-level TCP timeout (~110 s on Linux).
# socket.setdefaulttimeout() applies to any socket that hasn't set its own
# explicit timeout -- i.e. the SDK's internal requests session. Must be called
# before any network calls are made.
NETWORK_TIMEOUT_SECONDS = 15
socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)


# --- USER CONFIGURATION -------------------------------------------------------
TOTP_API_KEY = "TOTP_API_KEY"   # Replace with your Groww TOTP token
TOTP_SECRET  = "TOTP_SECRET"    # Replace with your Groww TOTP secret

# Supertrend settings
ST_LENGTH = 20    # ATR period
ST_FACTOR = 2.0   # ATR multiplier

# Trade settings
QUANTITY              = 1    # lots per direction (1 lot = 100 g Gold Mini)
EXIT_POINTS_THRESHOLD = 220  # Rs/unit; exit when |LTP - entry| >= this

# Gold Mini instrument prefix on MCX
SYMBOL_PREFIX = "GOLDM"

# MCX commodity session hours (IST)
MARKET_OPEN      = "16:00"
ENTRY_START_TIME = "16:02"   # no entries in first 2 minutes (warm-up)
ENTRY_END_TIME   = "22:55"   # stop taking new entries 10 min before square-off
SQUARE_OFF_TIME  = "23:05"   # force-close 10 min before Groww MIS auto S/O (~23:15)
MARKET_CLOSE     = "23:30"

LOOKBACK_BARS      = 120   # number of 1-min candles to keep in rolling buffer
SEED_LOOKBACK_DAYS = 5     # calendar days to look back when seeding at startup

IST = timezone(timedelta(hours=5, minutes=30))


# --- LOGGING ------------------------------------------------------------------
def _resolve_log_dir() -> str:
    """
    Pick a writable directory for log files.
    Tries (in order): LOG_DIR env var -> ./logs -> <tempdir>/logs
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
    raise last_exc


LOG_DIR = _resolve_log_dir()
_LOG_FILE_PATH = os.path.join(
    LOG_DIR, f"goldmini_{datetime.now(IST).strftime('%Y-%m-%d')}.log"
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


# --- TRADE JOURNAL (CSV) ------------------------------------------------------
TRADE_JOURNAL_PATH = os.path.join(LOG_DIR, "goldmini_trade_journal.csv")
TRADE_JOURNAL_FIELDS = [
    "timestamp",        # IST execution time
    "leg",              # "LONG" or "SHORT"
    "action",           # "ENTRY" or "EXIT"
    "symbol",           # futures trading symbol
    "quantity",         # lots
    "price",            # execution/fill price (or LTP fallback)
    "fill_confirmed",   # True/False
    "order_id",         # Groww order id
    "pnl",              # per-trade P&L for EXIT rows (blank for ENTRY)
    "reason",           # e.g. "ST flip BULLISH", "SQUARE-OFF TIME"
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


# --- RATE LIMITING ------------------------------------------------------------
# Per Groww's published limits:
#   Orders      : 10/sec, 250/min
#   Live Data   : 10/sec, 300/min  (incl. historical candles)
#   Non Trading : 20/sec, 500/min
RATE_LIMITS = {
    "orders":      {"per_sec": 10, "per_min": 250},
    "live_data":   {"per_sec": 10, "per_min": 300},
    "non_trading": {"per_sec": 20, "per_min": 500},
}
RATE_LIMIT_SAFETY_MARGIN = 0.7   # use only ~70% of the documented ceiling

_rate_limit_history: dict[str, deque] = {k: deque() for k in RATE_LIMITS}


def _throttle(category: str) -> None:
    """Block just long enough to stay under the safety-margined rate limits."""
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


# Fixed backoff schedule for transient network/timeout errors.
NETWORK_RETRY_DELAYS = [2, 5, 10]


def _is_transient_network_error(exc: Exception) -> bool:
    """True for connection/timeout failures that are almost always transient."""
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
        "timed out", "timeout", "connection reset", "connection aborted",
        "max retries exceeded", "failed to establish a new connection",
        "connection refused",
    ))


def _call(category: str, fn, *args, **kwargs):
    """
    Throttled, fault-tolerant wrapper around every Groww SDK call.

    1. Paces the call via _throttle() to stay under the rate ceiling.
    2. Retries:
         - Rate limit errors  : exponential backoff (1s, 2s, 4s), up to 3 retries
         - Network/timeout    : fixed backoff (2s, 5s, 10s), up to 3 retries
    3. Any other exception is raised immediately.
    """
    _throttle(category)
    rate_limit_attempt = 0
    network_attempt    = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit_error(e):
                if rate_limit_attempt >= 3:
                    raise
                wait = 2 ** rate_limit_attempt
                log.warning(
                    f"Rate limit ({category}, attempt {rate_limit_attempt+1}/4) "
                    f"-- backing off {wait}s: {e}"
                )
                time.sleep(wait)
                rate_limit_attempt += 1
                _throttle(category)
                continue
            if _is_transient_network_error(e):
                if network_attempt >= len(NETWORK_RETRY_DELAYS):
                    log.error(
                        f"Network error persisted after {len(NETWORK_RETRY_DELAYS)} "
                        f"retries ({category}): {e}"
                    )
                    raise
                wait = NETWORK_RETRY_DELAYS[network_attempt]
                log.warning(
                    f"Network/timeout ({category}, attempt "
                    f"{network_attempt+1}/{len(NETWORK_RETRY_DELAYS)+1}) "
                    f"-- retrying in {wait}s: {e}"
                )
                time.sleep(wait)
                network_attempt += 1
                _throttle(category)
                continue
            raise


# --- AUTHENTICATION -----------------------------------------------------------
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
                log.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
    raise RuntimeError(
        f"Authentication failed after {max_retries} attempts -- "
        "check TOTP credentials and network connectivity to api.groww.in."
    )


# --- INSTRUMENT HELPERS -------------------------------------------------------
def get_all_instruments_df(groww: GrowwAPI) -> pd.DataFrame:
    """Return the full instruments dataframe with normalised column types."""
    df = _call("non_trading", groww.get_all_instruments)
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    return df


_instruments_cache: dict = {"df": None, "ts": 0.0}
INSTRUMENTS_CACHE_TTL_SECONDS = 1800  # 30 min -- instrument list doesn't change intraday


def get_cached_instruments_df(groww: GrowwAPI) -> pd.DataFrame:
    """
    Reuse the instruments dataframe for INSTRUMENTS_CACHE_TTL_SECONDS instead
    of re-fetching Groww's full instruments CSV on every signal cycle.
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


def get_active_goldmini_symbol(df: pd.DataFrame) -> tuple[str, str]:
    """
    Return (trading_symbol, groww_symbol) for the nearest-expiry Gold Mini
    futures contract on MCX COMMODITY.
    groww_symbol is needed by get_historical_candles().
    trading_symbol is used for LTP and order placement.
    """
    today = pd.Timestamp(datetime.now(IST).date())
    mask = (
        (df["exchange"] == "MCX")
        & df["trading_symbol"].str.startswith(SYMBOL_PREFIX, na=False)
        & (df["instrument_type"].str.upper() == "FUT")
        & (df["expiry_date"] >= today)
    )
    active = df[mask].copy().sort_values("expiry_date")
    if active.empty:
        raise RuntimeError(
            f"No active {SYMBOL_PREFIX} contract found on MCX. "
            "Check the instruments CSV or market calendar."
        )
    row          = active.iloc[0]
    sym          = str(row["trading_symbol"])
    groww_symbol = str(row["groww_symbol"]) if "groww_symbol" in row.index else sym
    log.info(
        f"Selected Gold Mini instrument: {sym}  "
        f"(expiry: {row['expiry_date'].date()}  groww_symbol={groww_symbol})"
    )
    return sym, groww_symbol


# --- GOLD MINI LTP ------------------------------------------------------------
def get_ltp(groww: GrowwAPI, trading_symbol: str) -> float:
    """Fetch the last traded price for *trading_symbol* on MCX COMMODITY."""
    key = f"MCX_{trading_symbol}"
    resp = _call(
        "live_data", groww.get_ltp,
        segment=groww.SEGMENT_COMMODITY,
        exchange_trading_symbols=key,
    )
    ltp = resp.get(key)
    if ltp is None:
        raise RuntimeError(f"Gold Mini LTP not found for {trading_symbol}. Response: {resp}")
    return float(ltp)


# --- CANDLE BUFFER ------------------------------------------------------------
_candle_buffer: pd.DataFrame | None = None


def reset_candle_buffer() -> None:
    global _candle_buffer
    _candle_buffer = None


def fetch_1min_candles(groww: GrowwAPI, groww_symbol: str) -> pd.DataFrame:
    """
    Fetch 1-minute candles for Gold Mini futures (MCX COMMODITY).
    Uses get_historical_candles() (the non-deprecated API).
    Candle timestamps come back as "yyyy-MM-ddTHH:mm:ss" strings (IST).

    Initial multi-day seed (SEED_LOOKBACK_DAYS) ensures Supertrend is hot at
    market open. Subsequent calls do incremental 15-minute fetches to stay
    well inside Groww's Live Data rate limit.
    """
    global _candle_buffer

    end_dt = datetime.now(IST).replace(tzinfo=None)
    full_seed_needed = _candle_buffer is None or len(_candle_buffer) < ST_LENGTH + 5

    if full_seed_needed:
        start_dt = end_dt - timedelta(days=SEED_LOOKBACK_DAYS)
        log.info(
            f"Seeding Gold Mini candle buffer with prior-session history "
            f"(from {start_dt.strftime('%Y-%m-%d %H:%M')})..."
        )
    else:
        start_dt = end_dt - timedelta(minutes=15)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        response = _call(
            "live_data", groww.get_historical_candles,
            exchange=groww.EXCHANGE_MCX,
            segment=groww.SEGMENT_COMMODITY,
            groww_symbol=groww_symbol,
            start_time=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            candle_interval=groww.CANDLE_INTERVAL_MIN_1,
        )

    candles = response.get("candles", [])
    if not candles:
        if _candle_buffer is not None and len(_candle_buffer) >= ST_LENGTH + 5:
            log.warning("No new candles returned -- using cached buffer.")
            return _candle_buffer
        raise RuntimeError(
            f"No candle data returned for {groww_symbol} (Gold Mini). "
            "Market may be closed or the symbol may differ."
        )

    ncols     = len(candles[0])
    base_cols = ["ts", "open", "high", "low", "close", "volume"]
    cols      = base_cols + (["oi"] if ncols >= 7 else [])
    df_new    = pd.DataFrame(candles, columns=cols)
    df_new["ts"] = pd.to_datetime(df_new["ts"], errors="coerce")

    if _candle_buffer is None:
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


# --- SUPERTREND INDICATOR -----------------------------------------------------
def compute_supertrend_direction(df: pd.DataFrame, length: int, factor: float) -> int:
    """
    Compute Supertrend with Wilder's ATR on *df*.
    Returns +1 (bullish) or -1 (bearish) for the LAST ROW of df.

    Callers must pass a df that already ends on a fully-closed candle
    (see the completed-candle trim in run_strategy).
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


# --- ORDER HELPERS ------------------------------------------------------------
def _place_limit_order(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    price: float,
    label: str,
) -> dict:
    log.info(f">>> Placing LIMIT {label}  qty={quantity} x {trading_symbol}  @ Rs{price:.2f}")
    resp = _call(
        "orders", groww.place_order,
        trading_symbol=trading_symbol,
        quantity=quantity,
        validity=groww.VALIDITY_DAY,
        exchange=groww.EXCHANGE_MCX,
        segment=groww.SEGMENT_COMMODITY,
        product=groww.PRODUCT_MIS,
        order_type=groww.ORDER_TYPE_LIMIT,
        transaction_type=transaction_type,
        price=price,
    )
    log.info(f"    {label} order response : {resp}")
    return resp


def _place_market_order(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    label: str,
) -> dict:
    """
    Place a MARKET order -- used only as a last-resort EXIT fallback after all
    LIMIT escalation offsets fail. MCX commodity MARKET orders for MIS may or
    may not be supported on your account. If Groww rejects this call, the caller
    falls back to leaving the last LIMIT order resting / unconfirmed.
    """
    log.info(f">>> Placing MARKET {label}  qty={quantity} x {trading_symbol}")
    resp = _call(
        "orders", groww.place_order,
        trading_symbol=trading_symbol,
        quantity=quantity,
        validity=groww.VALIDITY_DAY,
        exchange=groww.EXCHANGE_MCX,
        segment=groww.SEGMENT_COMMODITY,
        product=groww.PRODUCT_MIS,
        order_type=groww.ORDER_TYPE_MARKET,
        transaction_type=transaction_type,
    )
    log.info(f"    {label} order response : {resp}")
    return resp


# Escalating price offsets (Rs) -- Gold Mini tick size is Rs1.
ESCALATION_OFFSETS = (2, 5, 10)
FILL_WAIT_RETRIES  = 6     # polls per offset attempt
FILL_WAIT_INTERVAL = 1.0   # seconds between polls

# 4th-retry fallback (EXIT ONLY): cancel the resting LIMIT and place MARKET.
EXIT_MARKET_FALLBACK_ENABLED  = True
MARKET_FALLBACK_POLL_RETRIES  = 10
MARKET_FALLBACK_POLL_INTERVAL = 1.0


def _get_filled_quantity(detail: dict) -> float:
    """Best-effort extraction of how much quantity has actually executed."""
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
    4th retry (exits only): cancel the resting LIMIT and place one MARKET order.
    Returns (order_id, fill_price_or_None) -- same contract as _execute_with_escalation.
    """
    FILLED_STATUSES = ("EXECUTED", "COMPLETED", "DELIVERY_AWAITED")

    # Re-check the resting LIMIT first -- may have filled in the meantime.
    try:
        detail = _call(
            "non_trading", groww.get_order_detail,
            groww_order_id=resting_order_id, segment=groww.SEGMENT_COMMODITY,
        )
        if detail.get("order_status") in FILLED_STATUSES:
            fill_price = float(detail.get("average_fill_price") or 0)
            if fill_price:
                log.info(f"{label}: resting order {resting_order_id} filled right before MARKET fallback.")
                return resting_order_id, fill_price
        if _get_filled_quantity(detail) > 0:
            log.warning(f"{label}: resting order shows PARTIAL fill -- NOT cancelling. VERIFY POSITION ON GROWW.")
            return resting_order_id, None
    except Exception as e:
        log.warning(f"{label}: pre-cancel check failed for {resting_order_id}: {e}")

    # Cancel the resting LIMIT.
    try:
        _call("orders", groww.cancel_order,
              groww_order_id=resting_order_id, segment=groww.SEGMENT_COMMODITY)
        log.info(f"{label}: cancelled resting order {resting_order_id} ahead of MARKET fallback.")
    except Exception as e:
        log.warning(
            f"{label}: failed to cancel {resting_order_id}: {e}. "
            "NOT placing MARKET order on top. VERIFY POSITION ON GROWW."
        )
        return resting_order_id, None

    # Place the MARKET order.
    try:
        resp = _place_market_order(
            groww, trading_symbol, transaction_type, quantity, f"{label} (MARKET fallback)"
        )
        market_order_id = resp.get("groww_order_id", "")
        if not market_order_id:
            log.warning(f"{label}: MARKET fallback returned no groww_order_id. VERIFY POSITION ON GROWW.")
            return resting_order_id, None
    except Exception as e:
        log.error(
            f"{label}: MARKET fallback placement failed: {e}. "
            "Original LIMIT was cancelled. VERIFY POSITION ON GROWW.",
            exc_info=True,
        )
        return resting_order_id, None

    # Poll the MARKET order for a fill.
    for _ in range(MARKET_FALLBACK_POLL_RETRIES):
        try:
            detail = _call(
                "non_trading", groww.get_order_detail,
                groww_order_id=market_order_id, segment=groww.SEGMENT_COMMODITY,
            )
            status = detail.get("order_status")
            if status in FILLED_STATUSES:
                fill_price = float(detail.get("average_fill_price") or 0)
                if fill_price:
                    log.info(f"{label}: MARKET fallback filled Rs{fill_price:.2f}  status={status}")
                    return market_order_id, fill_price
            if status in ("REJECTED", "FAILED", "CANCELLED"):
                log.warning(f"{label}: MARKET fallback ended status={status} without fill. VERIFY POSITION ON GROWW.")
                return market_order_id, None
            if _get_filled_quantity(detail) > 0:
                log.warning(f"{label}: MARKET fallback PARTIAL fill. VERIFY POSITION ON GROWW.")
                return market_order_id, None
        except Exception as e:
            log.warning(f"{label}: error polling MARKET fallback order {market_order_id}: {e}")
        time.sleep(MARKET_FALLBACK_POLL_INTERVAL)

    log.warning(
        f"{label}: MARKET fallback not confirmed after {MARKET_FALLBACK_POLL_RETRIES} polls. "
        "Manual check recommended."
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

    BUY (entry long / exit short): price = LTP + offset
    SELL (exit long / entry short): price = LTP - offset

    On each attempt:
      - Place the LIMIT order.
      - Poll get_order_detail up to FILL_WAIT_RETRIES times.
      - If filled (EXECUTED/COMPLETED/DELIVERY_AWAITED with average_fill_price),
        return (order_id, fill_price).
      - Else cancel and retry with the next wider offset.

    If the final offset also does not fill:
      - use_market_fallback=True (exits): cancel and fire one MARKET order.
      - Otherwise: leave order resting, return (order_id, None).

    Safety: exception AFTER an order is placed -> return (order_id, None) instead
    of re-raising, so the caller never loses track of a live order.
    """
    FILLED_STATUSES = ("EXECUTED", "COMPLETED", "DELIVERY_AWAITED")
    last_order_id = ""

    for i, offset in enumerate(ESCALATION_OFFSETS):
        is_last_offset = (i == len(ESCALATION_OFFSETS) - 1)
        try:
            ltp   = get_ltp(groww, trading_symbol)
            price = round(ltp + offset if side == "BUY" else max(0.05, ltp - offset), 2)

            resp = _place_limit_order(
                groww, trading_symbol, transaction_type, quantity, price,
                f"{label} (offset=Rs{offset})"
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
                        groww_order_id=order_id, segment=groww.SEGMENT_COMMODITY,
                    )
                    status = detail.get("order_status")
                    if status in FILLED_STATUSES:
                        fill_price = float(detail.get("average_fill_price") or 0)
                        if fill_price:
                            log.info(
                                f"{label}: filled at offset=Rs{offset}  "
                                f"avg_fill=Rs{fill_price:.2f}  status={status}"
                            )
                            return order_id, fill_price
                    if status in ("REJECTED", "FAILED", "CANCELLED"):
                        log.warning(f"{label}: order {order_id} ended status={status} before fill.")
                        break
                    filled_qty = _get_filled_quantity(detail)
                    if filled_qty > 0:
                        partial_fill_seen = True
                        log.warning(
                            f"{label}: order {order_id} PARTIAL fill "
                            f"({filled_qty}/{quantity}) at offset=Rs{offset}. "
                            "Halting escalation. VERIFY POSITION ON GROWW."
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
                        f"{label}: order {order_id} not filled at widest offset "
                        f"(Rs{offset}). Attempting MARKET fallback."
                    )
                    return _market_fallback(
                        groww, trading_symbol, transaction_type, quantity, label, order_id
                    )
                log.warning(
                    f"{label}: order {order_id} not filled at widest offset (Rs{offset}). "
                    "LEAVING ORDER RESTING. Manual check on Groww recommended."
                )
                return order_id, None

            try:
                _call("orders", groww.cancel_order,
                      groww_order_id=order_id, segment=groww.SEGMENT_COMMODITY)
                log.info(f"{label}: order {order_id} not filled at offset=Rs{offset} -- cancelled, retrying wider.")
            except Exception as e:
                log.warning(
                    f"{label}: failed to cancel {order_id} (offset=Rs{offset}): {e}. "
                    "Halting escalation. VERIFY POSITION ON GROWW."
                )
                return order_id, None

            # Re-check right after the cancel ack.
            try:
                post_cancel = _call(
                    "non_trading", groww.get_order_detail,
                    groww_order_id=order_id, segment=groww.SEGMENT_COMMODITY,
                )
                if post_cancel.get("order_status") in FILLED_STATUSES:
                    fill_price = float(post_cancel.get("average_fill_price") or 0)
                    if fill_price:
                        log.warning(
                            f"{label}: order {order_id} FILLED at cancel time "
                            f"(Rs{fill_price:.2f}) -- using this fill."
                        )
                        return order_id, fill_price
                if _get_filled_quantity(post_cancel) > 0:
                    log.warning(
                        f"{label}: order {order_id} PARTIAL fill at cancel time. "
                        "Halting escalation. VERIFY POSITION ON GROWW."
                    )
                    return order_id, None
            except Exception as e:
                log.warning(f"{label}: post-cancel check failed for {order_id}: {e}")

        except Exception as exc:
            if last_order_id:
                log.error(
                    f"{label}: unexpected error after order {last_order_id} was placed "
                    f"(offset attempt {i+1}/{len(ESCALATION_OFFSETS)}): {exc}. "
                    "Returning as PENDING/UNCONFIRMED. VERIFY POSITION ON GROWW.",
                    exc_info=True,
                )
                return last_order_id, None
            log.error(
                f"{label}: unexpected error before any order was placed "
                f"(offset attempt {i+1}/{len(ESCALATION_OFFSETS)}): {exc}. Re-raising.",
                exc_info=True,
            )
            raise

    return last_order_id, None


def buy_futures(groww: GrowwAPI, trading_symbol: str, quantity: int, label: str) -> tuple[str, float | None]:
    """BUY to open a long position, escalating LIMIT offset (Rs2->Rs5->Rs10) until filled."""
    return _execute_with_escalation(
        groww, trading_symbol, groww.TRANSACTION_TYPE_BUY, quantity, "BUY", f"BUY {label}"
    )


def sell_futures(groww: GrowwAPI, trading_symbol: str, quantity: int, label: str) -> tuple[str, float | None]:
    """SELL to open a short position, escalating LIMIT offset (Rs2->Rs5->Rs10) until filled."""
    return _execute_with_escalation(
        groww, trading_symbol, groww.TRANSACTION_TYPE_SELL, quantity, "SELL", f"SELL {label}"
    )


def exit_long(groww: GrowwAPI, trading_symbol: str, quantity: int, label: str) -> tuple[str, float | None]:
    """
    SELL to close a long. Escalates LIMIT; if all offsets fail, cancels and
    fires one MARKET order (EXIT_MARKET_FALLBACK_ENABLED).
    """
    return _execute_with_escalation(
        groww, trading_symbol, groww.TRANSACTION_TYPE_SELL, quantity, "SELL",
        f"EXIT LONG {label}", use_market_fallback=True,
    )


def exit_short(groww: GrowwAPI, trading_symbol: str, quantity: int, label: str) -> tuple[str, float | None]:
    """
    BUY to cover a short. Escalates LIMIT; if all offsets fail, cancels and
    fires one MARKET order (EXIT_MARKET_FALLBACK_ENABLED).
    """
    return _execute_with_escalation(
        groww, trading_symbol, groww.TRANSACTION_TYPE_BUY, quantity, "BUY",
        f"EXIT SHORT {label}", use_market_fallback=True,
    )


# --- ORDER FILL PRICE POLLER --------------------------------------------------
def _await_fill_price(groww: GrowwAPI, order_id: str, retries: int = 6) -> float | None:
    """
    Poll order detail until the order is filled.
    Returns average_fill_price, or None on timeout (caller falls back to LTP).

    NOTE: get_order_status() does NOT return average_fill_price -- only
    get_order_detail() does. Valid filled statuses: EXECUTED / COMPLETED / DELIVERY_AWAITED.
    """
    FILLED_STATUSES = ("EXECUTED", "COMPLETED", "DELIVERY_AWAITED")
    for _ in range(retries):
        try:
            detail = _call(
                "non_trading", groww.get_order_detail,
                groww_order_id=order_id, segment=groww.SEGMENT_COMMODITY,
            )
            if detail.get("order_status") in FILLED_STATUSES:
                price = float(detail.get("average_fill_price") or 0)
                if price:
                    return price
        except Exception as e:
            log.warning(f"Error polling order {order_id}: {e}")
        time.sleep(0.5)
    log.warning(f"Order {order_id} not filled after {retries} polls -- using LTP as fallback.")
    return None


# --- MARKET STATUS ------------------------------------------------------------
def market_status(now_hm: str) -> str:
    """Returns 'CLOSED', 'SQUAREOFF', or 'OPEN'."""
    if now_hm >= MARKET_CLOSE or now_hm < MARKET_OPEN:
        return "CLOSED"
    if now_hm >= SQUARE_OFF_TIME:
        return "SQUAREOFF"
    return "OPEN"


# --- FORCE SQUARE-OFF ---------------------------------------------------------
def square_off_position(
    groww: GrowwAPI,
    pos: dict,
    quantity: int,
    leg_label: str,
    reason: str = "force square-off",
) -> dict:
    """
    Close an open futures position (exit long or cover short).
    pos dict keys: symbol, side ("LONG"|"SHORT"), entry_price, order_id, entry_time.
    Returns a trade record dict suitable for the trade journal.
    """
    symbol      = pos["symbol"]
    side        = pos.get("side", "LONG")
    entry_price = pos.get("entry_price", 0.0)
    entry_time  = pos.get("entry_time")
    entry_str   = f"Rs{entry_price:.2f}" if entry_price else "unknown"
    log.warning(f"[{reason}] Closing {leg_label} {side}  entry={entry_str}  symbol={symbol}")

    if side == "LONG":
        order_id, fill_price = exit_long(groww, symbol, quantity, leg_label)
    else:
        order_id, fill_price = exit_short(groww, symbol, quantity, leg_label)

    exit_time           = datetime.now(IST)
    exit_fill_confirmed = fill_price is not None
    exit_price          = fill_price if fill_price is not None else get_ltp(groww, symbol)

    hold_str = ""
    if entry_time is not None:
        hold_str = f"  held={str(exit_time - entry_time).split('.')[0]}"

    if not exit_fill_confirmed:
        log.warning(
            f"    {leg_label} exit order {order_id} fill UNCONFIRMED -- "
            f"using LTP Rs{exit_price:.2f} for logging only. VERIFY POSITION ON GROWW."
        )

    pnl = None
    if entry_price:
        pnl = (
            (exit_price - entry_price) * quantity
            if side == "LONG"
            else (entry_price - exit_price) * quantity
        )

    if entry_price:
        log.info(
            f"[TRADE] {leg_label} EXIT  symbol={symbol}  qty={quantity}  "
            f"entry=Rs{entry_price:.2f}  exit=Rs{exit_price:.2f}  "
            f"time={exit_time:%Y-%m-%d %H:%M:%S}{hold_str}  "
            f"P&L=Rs{pnl:.2f}  fill_confirmed={exit_fill_confirmed}  "
            f"order_id={order_id}  reason={reason}"
        )
    else:
        log.info(
            f"[TRADE] {leg_label} EXIT  symbol={symbol}  qty={quantity}  "
            f"exit=Rs{exit_price:.2f}  time={exit_time:%Y-%m-%d %H:%M:%S}{hold_str}  "
            f"P&L=N/A  fill_confirmed={exit_fill_confirmed}  "
            f"order_id={order_id}  reason={reason}"
        )

    log_trade(
        leg=leg_label, action="EXIT", symbol=symbol, quantity=quantity,
        price=exit_price, fill_confirmed=exit_fill_confirmed, order_id=order_id,
        reason=reason, pnl=pnl, when=exit_time,
    )
    return {
        "leg": leg_label, "symbol": symbol, "quantity": quantity,
        "entry_price": entry_price, "entry_time": entry_time,
        "exit_price": exit_price, "exit_time": exit_time,
        "exit_order_id": order_id, "exit_fill_confirmed": exit_fill_confirmed,
        "pnl": pnl, "reason": reason,
    }


def attempt_square_off(
    groww: GrowwAPI,
    pos: dict,
    quantity: int,
    leg_label: str,
    reason: str = "force square-off",
) -> dict | None:
    """
    Wraps square_off_position() and decides whether the leg can be marked flat.

    Returns:
      - None        -> exit fill CONFIRMED. Leg is genuinely flat.
      - updated pos -> exit fill UNCONFIRMED (exit_pending=True). The main loop
                        polls each cycle and finalizes the leg once confirmed.
    This prevents the earlier bug (in the older Gold Mini scripts) where a leg
    was blindly marked flat regardless of whether the exit order actually filled.
    """
    trade = square_off_position(groww, pos, quantity, leg_label, reason=reason)
    if trade["exit_fill_confirmed"]:
        return None
    pos["exit_pending"]  = True
    pos["exit_order_id"] = trade["exit_order_id"]
    pos["exit_reason"]   = reason
    pos["exit_quantity"] = quantity
    log.warning(
        f"{leg_label}: exit order unconfirmed -- leg kept OPEN (exit_pending=True). "
        f"Checking order {trade['exit_order_id']} each cycle. VERIFY POSITION ON GROWW."
    )
    return pos


# --- STRATEGY LOOP ------------------------------------------------------------
def run_strategy() -> None:
    """
    Main polling loop -- runs every ~60 seconds (aligned to minute boundaries).

    LONG leg (long_pos):
      * Entry  : ST turns BULLISH  -> BUY  1 lot Gold Mini futures
      * Exit   : ST turns BEARISH  AND  |LTP - entry_price| >= EXIT_POINTS_THRESHOLD

    SHORT leg (short_pos):
      * Entry  : ST turns BEARISH  -> SELL 1 lot Gold Mini futures
      * Exit   : ST turns BULLISH  AND  |LTP - entry_price| >= EXIT_POINTS_THRESHOLD

    Both legs are independent.  Force-close all at SQUARE_OFF_TIME (23:05 IST).
    """
    groww          = authenticate()
    last_auth_time = datetime.now(IST)
    RE_AUTH_INTERVAL = timedelta(hours=2)

    instruments_df = get_cached_instruments_df(groww)
    symbol, groww_symbol = get_active_goldmini_symbol(instruments_df)

    # State -- None = flat; dict = open position details
    long_pos:  dict | None = None
    short_pos: dict | None = None
    prev_direction: int | None = None

    log.info(
        f"Gold Mini Strategy started.  Symbol: {symbol}  "
        f"Supertrend({ST_LENGTH}, {ST_FACTOR})  "
        f"Qty: {QUANTITY} lot(s) per direction  "
        f"Exit threshold: +-Rs{EXIT_POINTS_THRESHOLD}/unit  "
        f"Entry window: {ENTRY_START_TIME}-{ENTRY_END_TIME}  "
        f"Force-close: {SQUARE_OFF_TIME}"
    )

    while True:
        try:
            now_ist    = datetime.now(IST)
            now_ts     = now_ist.strftime("%Y-%m-%d %H:%M:%S")
            now_hm     = now_ist.strftime("%H:%M")
            iter_start = time.monotonic()
            status     = market_status(now_hm)

            # -- Outside market hours -----------------------------------------
            if status == "CLOSED":
                log.info(f"[{now_ts}] Market CLOSED. Sleeping 60 s...")
                time.sleep(60)
                continue

            # -- Force square-off at SQUARE_OFF_TIME --------------------------
            if status == "SQUAREOFF":
                had_positions = long_pos is not None or short_pos is not None
                if long_pos is not None:
                    try:
                        long_pos = attempt_square_off(
                            groww, long_pos, QUANTITY, "LONG", reason="SQUARE-OFF TIME"
                        )
                    except Exception as e:
                        log.error(
                            f"Failed to close LONG (symbol={long_pos['symbol']}): {e}. "
                            "MANUAL INTERVENTION REQUIRED -- check open positions on Groww."
                        )
                if short_pos is not None:
                    try:
                        short_pos = attempt_square_off(
                            groww, short_pos, QUANTITY, "SHORT", reason="SQUARE-OFF TIME"
                        )
                    except Exception as e:
                        log.error(
                            f"Failed to close SHORT (symbol={short_pos['symbol']}): {e}. "
                            "MANUAL INTERVENTION REQUIRED -- check open positions on Groww."
                        )
                if long_pos is not None or short_pos is not None:
                    log.warning(
                        "One or more legs unconfirmed flat at force-close. "
                        "MANUAL INTERVENTION REQUIRED -- check open positions on Groww."
                    )
                if not had_positions:
                    log.info(f"[{now_ts}] SQUARE-OFF TIME -- no open positions.")
                log.info("Session ended. Exiting strategy.")
                break

            # -- Refresh instrument (handles front-month roll) ----------------
            instruments_df   = get_cached_instruments_df(groww)
            symbol, groww_symbol = get_active_goldmini_symbol(instruments_df)

            # -- Fetch 1-min candles ------------------------------------------
            df = fetch_1min_candles(groww, groww_symbol)

            # Drop the last row if it is a still-forming (not yet closed) candle.
            # This ensures Supertrend is computed on fully-closed bars only,
            # matching TradingView's behaviour.
            last_bar_start = df.iloc[-1]["ts"]           # naive IST, start-of-bar
            now_naive      = now_ist.replace(tzinfo=None)
            if now_naive < last_bar_start + timedelta(minutes=1):
                df = df.iloc[:-1]   # still forming -- exclude it

            if len(df) < ST_LENGTH + 5:
                log.warning(
                    f"[{now_ts}] Only {len(df)} completed bars "
                    f"(need {ST_LENGTH + 5}). Retrying in 60 s..."
                )
                time.sleep(60)
                continue

            if prev_direction is None:
                log.info(
                    f"[DIAG] Last COMPLETED candle: ts={df.iloc[-1]['ts']}  "
                    f"close=Rs{df.iloc[-1]['close']:.2f}  now={now_ts}. "
                    "Cross-check against TradingView Gold Mini 1-min chart to confirm alignment."
                )

            # -- Supertrend direction -----------------------------------------
            curr_direction = compute_supertrend_direction(df, ST_LENGTH, ST_FACTOR)
            last_close     = float(df.iloc[-1]["close"])
            trend_label    = "BULLISH UP" if curr_direction == 1 else "BEARISH DOWN"

            def _leg_status_label(leg_pos: dict | None, leg_name: str) -> str:
                """Per-minute status string for one leg, including live LTP."""
                if not leg_pos or leg_pos.get("entry_price") is None:
                    return f"{leg_name}: FLAT"
                entry = leg_pos["entry_price"]
                side  = leg_pos.get("side", "?")
                try:
                    live_ltp = get_ltp(groww, leg_pos["symbol"])
                    diff = (live_ltp - entry) if side == "LONG" else (entry - live_ltp)
                    return (
                        f"{leg_name} {side} entry=Rs{entry:.2f}  "
                        f"LTP=Rs{live_ltp:.2f}  P&L/unit={diff:+.2f}"
                    )
                except Exception as e:
                    return f"{leg_name} {side} entry=Rs{entry:.2f}  LTP=unavailable ({e})"

            log.info(
                f"[{now_ts}]  ST={trend_label}  GoldMini Close=Rs{last_close:.2f}  "
                f"|  {_leg_status_label(long_pos, 'LONG')}  "
                f"|  {_leg_status_label(short_pos, 'SHORT')}"
            )

            # First-run: enter immediately on current trend direction.
            # Seed prev_direction to the OPPOSITE so the flip condition fires
            # in this same iteration -- matching BankNifty v20072026A behaviour.
            if prev_direction is None:
                prev_direction = -1 if curr_direction == 1 else 1
                log.info(
                    f"First run -- current ST is {trend_label}. "
                    "Entering trade immediately (no flip required on startup)."
                )

            # -- Recover missing entry prices ---------------------------------
            for leg_pos, leg_name in [(long_pos, "LONG"), (short_pos, "SHORT")]:
                if leg_pos is not None and leg_pos.get("entry_price") is None:
                    try:
                        confirmed = _await_fill_price(groww, leg_pos["order_id"])
                        if confirmed is not None:
                            leg_pos["entry_price"] = confirmed
                            log.info(
                                f"[TRADE] {leg_name} ENTRY_RECOVERED  "
                                f"fill=Rs{confirmed:.2f}  order_id={leg_pos['order_id']}"
                            )
                        else:
                            ltp = get_ltp(groww, leg_pos["symbol"])
                            leg_pos["entry_price"] = ltp
                            log.warning(
                                f"{leg_name} entry order still unfilled -- "
                                f"using LTP Rs{ltp:.2f} as provisional price. "
                                "VERIFY POSITION ON GROWW."
                            )
                    except Exception as e:
                        log.warning(f"Could not recover {leg_name} entry price: {e}")

            # -- Resolve pending exits ----------------------------------------
            for leg_pos, leg_name, leg_label in (
                (long_pos, "LONG", "LONG"), (short_pos, "SHORT", "SHORT")
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
                        "unresolved -- leg remains OPEN; not re-attempting a new exit."
                    )
                    continue

                entry_price = leg_pos.get("entry_price")
                exit_qty    = leg_pos.get("exit_quantity", QUANTITY)
                pnl = None
                if entry_price:
                    pnl = (
                        (confirmed - entry_price) * exit_qty
                        if leg_pos.get("side") == "LONG"
                        else (entry_price - confirmed) * exit_qty
                    )
                pnl_str   = f"Rs{pnl:.2f}" if pnl is not None else "N/A"
                entry_str = f"Rs{entry_price:.2f}" if entry_price else "unknown"
                log.info(
                    f"[TRADE] {leg_label} EXIT CONFIRMED (was pending)  "
                    f"symbol={leg_pos['symbol']}  qty={exit_qty}  entry={entry_str}  "
                    f"exit=Rs{confirmed:.2f}  P&L={pnl_str}  "
                    f"order_id={leg_pos['exit_order_id']}  "
                    f"reason={leg_pos.get('exit_reason')}"
                )
                log_trade(
                    leg=leg_label, action="EXIT", symbol=leg_pos["symbol"], quantity=exit_qty,
                    price=confirmed, fill_confirmed=True, order_id=leg_pos["exit_order_id"],
                    reason=f"{leg_pos.get('exit_reason')} (confirmed on recheck)", pnl=pnl,
                )
                if leg_name == "LONG":
                    long_pos = None
                else:
                    short_pos = None

            entry_allowed = ENTRY_START_TIME <= now_hm <= ENTRY_END_TIME

            # =================================================================
            # LONG LEG  (enter on BULLISH flip, exit on BEARISH + threshold)
            # =================================================================

            # Entry: ST just flipped BULLISH
            if curr_direction == 1 and prev_direction == -1 and long_pos is None:
                if short_pos is not None:
                    log.info(
                        "ST flipped BULLISH but SHORT position is still open. "
                        "Blocking LONG entry to prevent clashing."
                    )
                elif not entry_allowed:
                    log.info(
                        f"ST flipped BULLISH but LONG entry blocked "
                        f"(outside entry window {ENTRY_START_TIME}-{ENTRY_END_TIME})."
                    )
                else:
                    ltp = get_ltp(groww, symbol)
                    log.info(
                        f"SIGNAL  ST -> BULLISH  GoldMini LTP=Rs{ltp:.2f}  "
                        f"-> BUY {QUANTITY} lot(s)"
                    )
                    order_id, fill_price = buy_futures(groww, symbol, QUANTITY, "LONG")
                    entry_time = datetime.now(IST)
                    long_pos   = {
                        "symbol":      symbol,
                        "side":        "LONG",
                        "entry_price": fill_price,   # None if fill unconfirmed
                        "entry_time":  entry_time,
                        "order_id":    order_id,
                    }
                    if fill_price is not None:
                        log.info(
                            f"[TRADE] LONG ENTRY  symbol={symbol}  qty={QUANTITY}  "
                            f"fill=Rs{fill_price:.2f}  time={entry_time:%Y-%m-%d %H:%M:%S}  "
                            f"order_id={order_id}  reason=ST flip BULLISH"
                        )
                    else:
                        log.warning(
                            f"[TRADE] LONG ENTRY (UNCONFIRMED)  symbol={symbol}  qty={QUANTITY}  "
                            f"time={entry_time:%Y-%m-%d %H:%M:%S}  order_id={order_id}  "
                            "reason=ST flip BULLISH  -- fill unknown. VERIFY POSITION ON GROWW."
                        )
                    log_trade(
                        leg="LONG", action="ENTRY", symbol=symbol, quantity=QUANTITY,
                        price=fill_price if fill_price is not None else ltp,
                        fill_confirmed=fill_price is not None,
                        order_id=order_id, reason="ST flip BULLISH", when=entry_time,
                    )

            # Exit: ST turned BEARISH AND threshold reached
            elif curr_direction == -1 and long_pos is not None:
                if long_pos.get("exit_pending"):
                    log.info("    LONG: exit already pending -- skipping new exit attempt this cycle.")
                elif long_pos.get("entry_price") is None:
                    log.warning("LONG: entry price unknown, skipping exit check.")
                else:
                    ltp      = get_ltp(groww, symbol)
                    diff     = ltp - long_pos["entry_price"]
                    abs_diff = abs(diff)
                    log.info(
                        f"    LONG exit check: LTP=Rs{ltp:.2f}  "
                        f"Entry=Rs{long_pos['entry_price']:.2f}  "
                        f"P&L/unit={diff:+.2f}  Threshold=Rs{EXIT_POINTS_THRESHOLD}"
                    )
                    if abs_diff >= EXIT_POINTS_THRESHOLD:
                        log.info(
                            f"SIGNAL  ST BEARISH + |P&L/unit|=Rs{abs_diff:.2f} >= "
                            f"Rs{EXIT_POINTS_THRESHOLD}  -> Exiting LONG"
                        )
                        long_pos = attempt_square_off(
                            groww, long_pos, QUANTITY, "LONG",
                            reason="ST flip BEARISH + threshold"
                        )
                    else:
                        log.info(
                            f"    Holding LONG -- "
                            f"|P&L/unit|=Rs{abs_diff:.2f} < Rs{EXIT_POINTS_THRESHOLD} threshold."
                        )

            # =================================================================
            # SHORT LEG  (enter on BEARISH flip, exit on BULLISH + threshold)
            # =================================================================

            # Entry: ST just flipped BEARISH
            if curr_direction == -1 and prev_direction == 1 and short_pos is None:
                if long_pos is not None:
                    log.info(
                        "ST flipped BEARISH but LONG position is still open. "
                        "Blocking SHORT entry to prevent clashing."
                    )
                elif not entry_allowed:
                    log.info(
                        f"ST flipped BEARISH but SHORT entry blocked "
                        f"(outside entry window {ENTRY_START_TIME}-{ENTRY_END_TIME})."
                    )
                else:
                    ltp = get_ltp(groww, symbol)
                    log.info(
                        f"SIGNAL  ST -> BEARISH  GoldMini LTP=Rs{ltp:.2f}  "
                        f"-> SELL {QUANTITY} lot(s)"
                    )
                    order_id, fill_price = sell_futures(groww, symbol, QUANTITY, "SHORT")
                    entry_time = datetime.now(IST)
                    short_pos  = {
                        "symbol":      symbol,
                        "side":        "SHORT",
                        "entry_price": fill_price,   # None if fill unconfirmed
                        "entry_time":  entry_time,
                        "order_id":    order_id,
                    }
                    if fill_price is not None:
                        log.info(
                            f"[TRADE] SHORT ENTRY  symbol={symbol}  qty={QUANTITY}  "
                            f"fill=Rs{fill_price:.2f}  time={entry_time:%Y-%m-%d %H:%M:%S}  "
                            f"order_id={order_id}  reason=ST flip BEARISH"
                        )
                    else:
                        log.warning(
                            f"[TRADE] SHORT ENTRY (UNCONFIRMED)  symbol={symbol}  qty={QUANTITY}  "
                            f"time={entry_time:%Y-%m-%d %H:%M:%S}  order_id={order_id}  "
                            "reason=ST flip BEARISH  -- fill unknown. VERIFY POSITION ON GROWW."
                        )
                    log_trade(
                        leg="SHORT", action="ENTRY", symbol=symbol, quantity=QUANTITY,
                        price=fill_price if fill_price is not None else ltp,
                        fill_confirmed=fill_price is not None,
                        order_id=order_id, reason="ST flip BEARISH", when=entry_time,
                    )

            # Exit: ST turned BULLISH AND threshold reached
            elif curr_direction == 1 and short_pos is not None:
                if short_pos.get("exit_pending"):
                    log.info("    SHORT: exit already pending -- skipping new exit attempt this cycle.")
                elif short_pos.get("entry_price") is None:
                    log.warning("SHORT: entry price unknown, skipping exit check.")
                else:
                    ltp      = get_ltp(groww, symbol)
                    diff     = short_pos["entry_price"] - ltp
                    abs_diff = abs(diff)
                    log.info(
                        f"    SHORT exit check: LTP=Rs{ltp:.2f}  "
                        f"Entry=Rs{short_pos['entry_price']:.2f}  "
                        f"P&L/unit={diff:+.2f}  Threshold=Rs{EXIT_POINTS_THRESHOLD}"
                    )
                    if abs_diff >= EXIT_POINTS_THRESHOLD:
                        log.info(
                            f"SIGNAL  ST BULLISH + |P&L/unit|=Rs{abs_diff:.2f} >= "
                            f"Rs{EXIT_POINTS_THRESHOLD}  -> Covering SHORT"
                        )
                        short_pos = attempt_square_off(
                            groww, short_pos, QUANTITY, "SHORT",
                            reason="ST flip BULLISH + threshold"
                        )
                    else:
                        log.info(
                            f"    Holding SHORT -- "
                            f"|P&L/unit|=Rs{abs_diff:.2f} < Rs{EXIT_POINTS_THRESHOLD} threshold."
                        )

            # -- Advance state and sleep to next minute ------------------------
            prev_direction = curr_direction
            elapsed = time.monotonic() - iter_start
            time.sleep(max(0.0, 60.0 - elapsed))

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt -- closing all open positions...")
            if long_pos is not None:
                try:
                    long_pos = attempt_square_off(
                        groww, long_pos, QUANTITY, "LONG", reason="KeyboardInterrupt"
                    )
                except Exception as e:
                    log.error(
                        f"Failed to close LONG (symbol={long_pos['symbol']}): {e}. "
                        "MANUAL INTERVENTION REQUIRED."
                    )
            if short_pos is not None:
                try:
                    short_pos = attempt_square_off(
                        groww, short_pos, QUANTITY, "SHORT", reason="KeyboardInterrupt"
                    )
                except Exception as e:
                    log.error(
                        f"Failed to close SHORT (symbol={short_pos['symbol']}): {e}. "
                        "MANUAL INTERVENTION REQUIRED."
                    )
            log.info("Strategy stopped by user.")
            break

        except Exception as exc:
            if _is_rate_limit_error(exc):
                log.warning(
                    f"Rate limit persisted past internal retries: {exc}. Sleeping 30 s..."
                )
                time.sleep(30)
            elif "forbidden" in str(exc).lower():
                now_ist = datetime.now(IST)
                age     = now_ist - last_auth_time
                if age >= RE_AUTH_INTERVAL:
                    log.warning(f"Access token {age} old -- likely expired. Re-authenticating...")
                    try:
                        groww          = authenticate()
                        last_auth_time = now_ist
                        log.info("Re-authenticated successfully.")
                    except Exception as auth_exc:
                        log.error(f"Re-authentication failed: {auth_exc}. Sleeping 60 s...")
                        time.sleep(60)
                else:
                    log.warning(f"Forbidden error but token only {age} old. Sleeping 60 s...")
                    time.sleep(60)
            elif _is_transient_network_error(exc):
                log.warning(
                    f"Network/timeout error persisted past internal retries: {exc}. "
                    "Sleeping 20 s..."
                )
                time.sleep(20)
            else:
                log.error(f"Unhandled error: {exc}", exc_info=True)
                log.info("Sleeping 60 s before retrying...")
                time.sleep(60)


# --- ENTRY POINT --------------------------------------------------------------
if __name__ == "__main__":
    run_strategy()
