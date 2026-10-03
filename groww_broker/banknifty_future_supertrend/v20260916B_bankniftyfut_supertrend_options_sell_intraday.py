"""
================================================================================
BankNifty Futures Supertrend Intraday Options Selling Strategy
================================================================================

OVERVIEW:
    This algorithm executes an automated, rule-based intraday options selling
    strategy on BankNifty (NSE FNO) powered by the Groww Trading API.
    It tracks trend momentum using a 1-minute Supertrend indicator calculated
    on the nearest BankNifty Futures contract and systematically takes short
    option positions (selling Put options during bullish phases and Call options
    during bearish phases) to harvest premium decay and directional momentum.

--------------------------------------------------------------------------------
KEY STRATEGY PARAMETERS:
--------------------------------------------------------------------------------
    • Underlying Index        : BANKNIFTY (via NSE FNO active Futures contract)
    • Strike Step             : 100 points
    • Expiry Selection        : Current monthly expiry (last Tuesday of the month)
    • Trade Quantity          : 2 lots per direction
    • Supertrend Length (ATR) : 20 periods (Wilder's ATR smoothing)
    • Supertrend Multiplier   : 1.8 factor
    • Min Option Premium      : ₹400 (walks ITM if ATM premium < ₹400)
    • Exit Point Threshold    : 111 points (|Option LTP - Entry Price| >= 111)
    • Candle Timeframe        : 1-minute completed bars (forming candles excluded)
    • Historical Pre-warm     : 5 calendar days lookback to prime indicator at open

--------------------------------------------------------------------------------
SESSION TIMING & EXECUTION GATES (IST):
--------------------------------------------------------------------------------
    • 09:15          : Market opens. Strategy connects and warms historical buffer.
    • 09:23          : ENTRY_START_TIME. Initial trade allowed on active trend.
    • 09:23 - 15:05  : Active entry window for trend flips (BULLISH <-> BEARISH).
    • 15:05          : ENTRY_END_TIME. No new positions entered after this cutoff.
    • 15:08          : SQUARE_OFF_TIME. Hard intraday force-exit for all open legs.
    • 15:30          : Market close.

--------------------------------------------------------------------------------
TRADE ENTRY LOGIC:
--------------------------------------------------------------------------------
    1. First Trade of Day:
       - As soon as the clock reaches 09:23 IST (ENTRY_START_TIME), the algorithm
         enters immediately according to the prevailing Supertrend direction
         without requiring a fresh flip.
    2. Subsequent Trades (Flip Signals):
       - Bullish Signal (ST flips to +1):
           * Sell Put (PE) of the monthly expiry.
       - Bearish Signal (ST flips to -1):
           * Sell Call (CE) of the monthly expiry.
    3. Dynamic Strike Selection & ITM Premium Walk:
       - Calculates the At-The-Money (ATM) strike: round(BankNifty LTP / 100) * 100.
       - Fetches live LTP of the ATM strike.
       - If ATM LTP >= MIN_OPTION_PRICE (₹400), sells the ATM contract.
       - If ATM LTP < MIN_OPTION_PRICE, walks In-The-Money (ITM) one strike step
         at a time (strikes decrease for CE, strikes increase for PE) up to
         10 steps until an option with premium >= ₹400 is found.

--------------------------------------------------------------------------------
TRADE EXIT & RISK MANAGEMENT:
--------------------------------------------------------------------------------
    1. Profit-Based Strike Switch (180-pt Rule):
       - When an open short leg earns >= PROFIT_SWITCH_THRESHOLD (180 points)
         from its entry fill price (entry_price - option_LTP >= 180 for a short),
         the strategy exits the current strike and immediately re-sells the nearest
         ATM strike of the same type (CE/PE) based on live BankNifty futures LTP.
         The profit baseline resets to the new fill price after each switch.
         This check runs every minute and operates independently per leg.
    2. Trend Reversal + Point Threshold Rule:
       - An open short leg is covered IF AND ONLY IF both conditions are met:
         a) Supertrend changes to the opposite direction (e.g., BEARISH for a Put short), AND
         b) Absolute movement from entry price meets or exceeds EXIT_POINTS_THRESHOLD:
            |Current Option LTP - Entry Price| >= 111 points.
       - This protects against false whipsaws by requiring an adverse price move
         confirmation before taking a loss or closing profits.
    3. End-of-Day Square-Off:
       - At 15:08 IST (SQUARE_OFF_TIME), any open positions are immediately
         covered via escalating limit/market orders to guarantee zero overnight risk.
    4. Independent Leg Tracking:
       - PUT and CALL positions are tracked separately (`put_pos` and `call_pos`),
         preventing race conditions or accidental double execution across legs.

--------------------------------------------------------------------------------
EXECUTION & RESILIENCE ARCHITECTURE:
--------------------------------------------------------------------------------
    • Smart LIMIT Price Escalation:
      Orders are placed as LIMIT orders at current LTP and escalated by
      price offsets (+₹2, +₹5, +₹10) if not filled immediately, ensuring
      quick fills without paying excessive bid-ask spreads.
    • Emergency MARKET Fallback:
      For exit cover orders, if all 3 LIMIT escalation tiers fail, an emergency
      MARKET order fallback triggers to ensure urgent risk neutralization.
    • Idempotency & Duplicate Order Prevention:
      Generates client-side `order_reference_id` per order. On network or
      timeout errors, checks order status before retrying to prevent duplicate orders.
    • Rate Limiting & Auth Token Caching:
      Enforces client-side throttling (70% safety margin) for Groww API quotas
      (orders: 10/s, live data: 10/s, non-trading: 20/s). Caches daily access
      tokens in `groww_token_cache.json` to prevent hitting the 150 auth calls/day quota.
    • Robust Error & State Handling:
      Handles transient network blips with backoff schedules, enforces hard
      call timeouts via ThreadPoolExecutor, and marks unconfirmed fills as
      pending rather than dropping state.
    • Structured Auditing:
      Streams events to console and dated log files (`strategy_YYYY-MM-DD.log`),
      and appends executed fills to a structured CSV file (`trade_journal.csv`).
================================================================================
"""

import os
import csv
import json
import time
import uuid
import socket
import logging
import tempfile
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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

try:
    from growwapi.groww.exceptions import GrowwAPIAuthenticationException
except ImportError:  # pragma: no cover - defensive against SDK path changes
    class GrowwAPIAuthenticationException(Exception):
        """Fallback stand-in if the SDK's exception module path changes.
        Real authentication errors are still caught via message-text matching
        in _is_auth_error() below."""
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
NETWORK_TIMEOUT_SECONDS = 60
socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)


# ─── USER CONFIGURATION ──────────────────────────────────────────────────────
TOTP_API_KEY = "TOTP_API_KEY"   # Replace with your Groww TOTP token
TOTP_SECRET  = "TOTP_SECRET"    # Replace with your Groww TOTP secret

# Supertrend settings
ST_LENGTH = 20
ST_FACTOR = 1.8

# Trade settings
QUANTITY              = 2    # lots per direction
EXIT_POINTS_THRESHOLD = 111  # points; exit when |option_LTP − sell_price| >= this

# Profit-based strike switch: once an open leg's profit crosses this many points
# (entry_price − current_LTP >= threshold for a short), exit the current strike
# and immediately re-sell the nearest ATM strike of the same type.  The profit
# baseline resets to the new fill price after each switch.
PROFIT_SWITCH_THRESHOLD = 180  # points

UNDERLYING = "BANKNIFTY"
STRIKE_STEP = 100  # BankNifty option strike interval

# Candle data — we compute ST on BANKNIFTY index (NSE CASH), not on the option itself
# The underlying trading symbol used for the index is:
BANKNIFTY_INDEX_SYMBOL = "NIFTY BANK"  # NSE CASH segment

# Time gates (IST, HH:MM strings)
ENTRY_START_TIME   = "09:23"
ENTRY_END_TIME     = "15:05"
SQUARE_OFF_TIME    = "15:08"
MARKET_OPEN        = "09:15"
MARKET_CLOSE       = "15:30"
# Time to wake up after being asleep overnight. Set before MARKET_OPEN to
# allow authentication, instrument loading, and candle-buffer seeding to
# complete before the first tradeable candle arrives at 09:15.
STRATEGY_WAKE_TIME = "09:00"

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
TOKEN_CACHE_FILE = os.path.join(LOG_DIR, "groww_token_cache.json")

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


def _is_auth_error(exc: Exception) -> bool:
    """
    True for Groww API authentication / token errors — i.e. the token has
    expired or been revoked and the request was rejected with HTTP 401.
    These are NOT transient network blips; they require a forced re-auth.
    """
    if isinstance(exc, GrowwAPIAuthenticationException):
        return True
    msg = str(exc).lower()
    return (
        "authentication failed" in msg
        or "api token has either expired" in msg
        or "invalid token" in msg
        or "token expired" in msg
        or "unauthorized" in msg
        or "http 401" in msg
        or "status 401" in msg
        or "status_code: 401" in msg
        # NOTE: plain "401" is intentionally NOT matched here — too broad;
        # a strike price like 40100 could appear in error messages and
        # incorrectly trigger a forced re-authentication.
    )


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


_HARD_TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="groww-call")


def _call_with_hard_timeout(fn, *args, timeout_seconds: float = 20.0, **kwargs):
    """
    Run fn in a worker thread and enforce a real wall-clock timeout on it.

    socket.setdefaulttimeout() only helps if the SDK's underlying HTTP client
    actually honours the global default socket timeout. Some SDKs construct
    their own requests.Session with an explicit timeout=None (or no timeout
    kwarg at all passed through to the connection), which silently overrides
    the global default -- in that case a stalled TCP connection or a slow
    server can hang the call indefinitely with zero exception raised, so
    _call()'s retry/backoff logic never triggers (it only reacts to raised
    exceptions). Running the call in a separate thread and bounding it with
    .result(timeout=...) guarantees we regain control even if the call itself
    never returns or raises, so the outer loop's error handling can still run.
    """
    future = _HARD_TIMEOUT_EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        raise TimeoutError(
            f"Call to {getattr(fn, '__name__', fn)} did not return within "
            f"{timeout_seconds}s (hard timeout) -- treating as a hung network call."
        )


def _call(category: str, fn, *args, hard_timeout_seconds: float = 20.0, **kwargs):
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

    `hard_timeout_seconds` lets a specific call site ask for more time before
    being treated as hung — e.g. the multi-day candle seed fetch legitimately
    moves more data than a quick LTP check and shouldn't share the same
    20s ceiling.
    """
    _throttle(category)
    rate_limit_attempt = 0
    network_attempt = 0

    while True:
        try:
            return _call_with_hard_timeout(
                fn, *args, timeout_seconds=hard_timeout_seconds, **kwargs
            )
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
def authenticate(force_refresh: bool = False) -> GrowwAPI:
    """Authenticate via cached token or TOTP with retry logic.

    Checks TOKEN_CACHE_FILE for an existing valid token generated today.
    If valid, reuses it immediately to avoid hitting Groww's strict auth
    rate limit (150 requests / 24 hours).
    """
    today_str = datetime.now(IST).strftime("%Y-%m-%d")

    # 1. Try loading cached token from today
    # Groww tokens have a shorter TTL than a calendar day. A token generated
    # at 20:00 the previous evening may expire by 06:00 the next morning,
    # even though it passes the "date == today" check. We store and check
    # generated_at so we proactively get a fresh token rather than discovering
    # expiry at 09:15 on the first API call of the trading day.
    MAX_TOKEN_AGE_HOURS = 7  # conservative: Groww tokens typically valid ~8 h

    if not force_refresh and os.path.exists(TOKEN_CACHE_FILE):
        try:
            with open(TOKEN_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            cached_date  = cache.get("date")
            cached_token = cache.get("token")

            if cached_date == today_str and cached_token:
                # Check token age against MAX_TOKEN_AGE_HOURS when available
                generated_at_str = cache.get("generated_at")
                token_age_ok = True
                if generated_at_str:
                    try:
                        generated_at = datetime.fromisoformat(generated_at_str)
                        if generated_at.tzinfo is None:
                            generated_at = generated_at.replace(tzinfo=IST)
                        token_age = datetime.now(IST) - generated_at
                        token_age_ok = token_age < timedelta(hours=MAX_TOKEN_AGE_HOURS)
                        if not token_age_ok:
                            log.info(
                                f"Cached token is {token_age} old (limit: {MAX_TOKEN_AGE_HOURS}h) "
                                "— treating as expired, requesting fresh token."
                            )
                    except (ValueError, TypeError) as ts_err:
                        log.warning(f"Could not parse cached token generated_at ({ts_err}); skipping age check.")

                if token_age_ok:
                    log.info("Found cached Groww access token for today. Verifying token...")
                    groww = GrowwAPI(cached_token)
                    # Quick lightweight verification.
                    # No timeout= kwarg here — socket.setdefaulttimeout already caps
                    # all connections to NETWORK_TIMEOUT_SECONDS globally.
                    groww.get_user_profile()
                    log.info("Cached Groww access token verified successfully. Skipping auth endpoint.")
                    return groww
            else:
                log.info(f"Cached token is from date {cached_date} (today: {today_str}). Requesting fresh token.")
        except Exception as e:
            log.warning(f"Cached token check failed ({e}). Proceeding with TOTP authentication...")
            try:
                if os.path.exists(TOKEN_CACHE_FILE):
                    os.remove(TOKEN_CACHE_FILE)
            except OSError:
                pass
    elif force_refresh:
        log.info("Force-refresh requested: invalidating cached token.")
        try:
            if os.path.exists(TOKEN_CACHE_FILE):
                os.remove(TOKEN_CACHE_FILE)
        except OSError:
            pass

    # 2. Authenticate via TOTP
    max_retries = 5
    secret = TOTP_SECRET.replace(" ", "").replace("-", "").upper()

    for attempt in range(1, max_retries + 1):
        try:
            totp = pyotp.TOTP(secret).now()
            access_token = GrowwAPI.get_access_token(api_key=TOTP_API_KEY, totp=totp)
            log.info(f"Authenticated with Groww API (TOTP, attempt {attempt}).")

            # Persist token to disk for reuse on restart
            try:
                with open(TOKEN_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump({
                        "date":         today_str,
                        "token":        access_token,
                        "generated_at": datetime.now(IST).isoformat(),
                    }, f)
                log.info(f"Groww access token cached to {TOKEN_CACHE_FILE}.")
            except Exception as save_err:
                log.warning(f"Could not cache access token to disk: {save_err}")

            return GrowwAPI(access_token)
        except Exception as e:
            is_rate_limit = _is_rate_limit_error(e)
            if is_rate_limit:
                retry_delay = 60 * attempt  # Exponential/gradual backoff: 60s, 120s, 180s...
                log.error(
                    f"Authentication attempt {attempt}/{max_retries} hit Groww API Rate Limit (HTTP 429): {e}. "
                    "Auth endpoint (/v1/token/api/access) quota (150 calls/24h) or burst limit exceeded."
                )
            else:
                retry_delay = 30
                log.error(f"Authentication attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                log.info(f"Retrying authentication in {retry_delay}s…")
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Authentication failed after {max_retries} attempts — "
        "check TOTP credentials, NTP clock synchronization, and rate-limit status on api.groww.in."
    )


# ─── INSTRUMENT HELPERS ───────────────────────────────────────────────────────
def get_all_instruments_df(groww: GrowwAPI) -> pd.DataFrame:
    """Return the full instruments dataframe with normalised column types."""
    df = _call("non_trading", groww.get_all_instruments, hard_timeout_seconds=45.0)
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
    On the expiry day itself, today's expiring contract is excluded so the
    strategy rolls straight to the next available expiry instead of trading
    the contract that is expiring today.
    The actual date is always derived from the live instruments CSV so
    the correct expiry is resolved automatically regardless of day changes.
    """
    today = pd.Timestamp(datetime.now(IST).date())

    mask = (
        (df["exchange"] == "NSE")
        & (df["underlying_symbol"] == UNDERLYING)
        & (df["instrument_type"].str.upper().isin(["CE", "PE"]))
        & (df["segment"].str.upper() == "FNO")
        & (df["expiry_date"] > today)
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
            hard_timeout_seconds=60.0 if full_seed_needed else 20.0,
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
# ─── SAFE (NON-DUPLICATING) ORDER PLACEMENT ──────────────────────────────────
# BUG FIX (found after a BankNifty 57700 CE cover got bought twice, 28-Aug-2026):
#
# place_order() is NOT idempotent, but it used to go through the generic
# _call() wrapper — the same wrapper used for read-only calls like
# get_order_detail. _call() treats network/timeout errors (including our own
# 20s hard-timeout, which raises plain TimeoutError) as safe to blindly
# retry by just calling place_order() again with the same arguments.
#
# That's fine for read-only calls, but dangerous for placing an order: if the
# original request actually reached Groww and was accepted, but the response
# was lost/delayed on our side (dropped connection, or the hard-timeout
# thread giving up on a call that was still completing server-side), a
# blind retry submits a brand-new, SEPARATE order — a real duplicate on the
# exchange. That's what produced the extra BUY fills.
#
# Fix: every placement attempt carries its own client-generated
# order_reference_id. On a transient network/timeout error we do NOT
# immediately resubmit — we first call get_order_status_by_reference() to
# check whether Groww actually created an order for that reference id. If it
# did, we use that order instead of placing a new one. Only when the lookup
# positively confirms no such order exists do we retry, with a fresh
# reference id. If the lookup itself can't be confirmed either way, we raise
# rather than guess — callers (e.g. _execute_with_escalation) already treat
# "unexpected error, order state unknown" conservatively (order left
# resting / position marked exit_pending) instead of duplicating.
def _generate_order_reference_id() -> str:
    """Short unique id passed to place_order so a retry can look the attempt
    up on Groww instead of blindly resubmitting it."""
    return uuid.uuid4().hex[:20]


def _lookup_order_by_reference(
    groww: GrowwAPI, segment: str, order_reference_id: str, retries: int = 3
) -> dict | None:
    """
    Checks whether a place_order attempt with this reference id actually
    reached Groww.

    Returns:
      - a dict (with groww_order_id populated) if Groww shows an order for
        this reference id — the original attempt succeeded.
      - None if the lookup POSITIVELY confirms no such order exists — safe
        to retry.

    Raises if the lookup itself can't be confirmed either way after
    `retries` attempts — deliberately does NOT return None in that case,
    since that would make an inconclusive check look identical to a
    confirmed "no order" and could let a real duplicate slip through.
    """
    for attempt in range(retries):
        try:
            detail = groww.get_order_status_by_reference(
                segment=segment, order_reference_id=order_reference_id,
            )
            order_id = detail.get("groww_order_id") or detail.get("order_id")
            if order_id:
                detail.setdefault("groww_order_id", order_id)
                return detail
            return None  # lookup succeeded — genuinely no order for this ref
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "404" in msg or "no order" in msg:
                return None  # lookup succeeded — genuinely no order for this ref
            if attempt < retries - 1:
                time.sleep(1.5)
                continue
            raise RuntimeError(
                f"Could not confirm whether order ref={order_reference_id} "
                f"was placed — reconciliation lookup itself failed: {e}"
            ) from e


def _place_order_safe(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    order_type,
    label: str,
    price: float = 0.0,
    hard_timeout_seconds: float = 20.0,
) -> dict:
    """
    Places an order via groww.place_order(), replacing the old blind-retry
    _call("orders", groww.place_order, ...) path. See the module comment
    above for why: this checks-before-retrying instead of resubmitting on
    every network/timeout error, so a slow/lost response can never turn
    into a duplicate order on the exchange.
    """
    _throttle("orders")
    rate_limit_attempt = 0
    network_attempt = 0

    while True:
        ref_id = _generate_order_reference_id()
        try:
            resp = _call_with_hard_timeout(
                groww.place_order,
                trading_symbol=trading_symbol,
                quantity=quantity,
                validity=groww.VALIDITY_DAY,
                exchange=groww.EXCHANGE_NSE,
                segment=groww.SEGMENT_FNO,
                product=groww.PRODUCT_MIS,
                order_type=order_type,
                transaction_type=transaction_type,
                price=price,
                order_reference_id=ref_id,
                timeout_seconds=hard_timeout_seconds,
            )
            log.info(f"    {label} order response (ref={ref_id}) : {resp}")
            return resp

        except Exception as e:
            if _is_rate_limit_error(e):
                if rate_limit_attempt >= 3:
                    raise
                wait = 2 ** rate_limit_attempt
                log.warning(
                    f"{label}: Groww rate limit hit placing order (attempt "
                    f"{rate_limit_attempt + 1}/4) — backing off {wait}s: {e}"
                )
                time.sleep(wait)
                rate_limit_attempt += 1
                _throttle("orders")
                continue

            if _is_transient_network_error(e):
                if network_attempt >= len(NETWORK_RETRY_DELAYS):
                    log.error(
                        f"{label}: network/timeout error placing order "
                        f"persisted after {len(NETWORK_RETRY_DELAYS)} retries: {e}"
                    )
                    raise
                wait = NETWORK_RETRY_DELAYS[network_attempt]
                log.warning(
                    f"{label}: network/timeout error placing order "
                    f"(ref={ref_id}, attempt {network_attempt + 1}/"
                    f"{len(NETWORK_RETRY_DELAYS) + 1}) — will confirm on Groww "
                    f"before deciding whether to retry: {e}"
                )
                time.sleep(wait)

                existing = _lookup_order_by_reference(groww, groww.SEGMENT_FNO, ref_id)
                if existing is not None:
                    log.warning(
                        f"{label}: order WAS placed on Groww despite the "
                        f"network error (ref={ref_id}, "
                        f"groww_order_id={existing.get('groww_order_id')}) — "
                        "using it instead of submitting a duplicate."
                    )
                    return existing

                log.warning(
                    f"{label}: confirmed ref={ref_id} was never created on "
                    "Groww — safe to retry."
                )
                network_attempt += 1
                _throttle("orders")
                continue

            raise


def _place_limit_order(
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
    return _place_order_safe(
        groww, trading_symbol, transaction_type, quantity,
        groww.ORDER_TYPE_LIMIT, label, price=price,
    )


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
    return _place_order_safe(
        groww, trading_symbol, transaction_type, quantity,
        groww.ORDER_TYPE_MARKET, label,
    )


# Escalating price offsets (₹) tried in order until the order fills.
# Groww's API does not support MARKET orders for this product, so we widen
# the LIMIT price step by step to chase a fill without crossing the full
# spread blindly on the first attempt.
ESCALATION_OFFSETS = (2, 5, 10, 15)
FILL_WAIT_RETRIES  = 6     # polls per offset attempt
FILL_WAIT_INTERVAL = 1.0   # seconds between polls

# Fallback (EXIT ONLY): if all LIMIT offsets above fail to fill,
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
    for key in (
        "filled_quantity", "quantity_filled", "executed_quantity",
        "filled_qty", "cumulative_quantity", "traded_quantity", "cum_quantity",
    ):
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
) -> tuple[str, float | None, int]:
    """
    The 4th retry, used only when _execute_with_escalation is called with
    use_market_fallback=True (i.e. exits) and all 3 LIMIT offsets failed to
    fill. Cancels the still-resting LIMIT order from the last offset, then
    places ONE MARKET order and polls up to MARKET_FALLBACK_POLL_RETRIES
    times for a fill.

    resting_order_id may be "" — meaning the caller has ALREADY confirmed
    (via a REJECTED/FAILED/CANCELLED order_status) that there is nothing
    live to check or cancel. In that case both pre-checks below are skipped
    entirely and we go straight to placing the MARKET order — trying to
    cancel an already-dead order is what caused the original bug (Groww
    correctly refuses with "Cancellation not allowed", which the old code
    misread as "state uncertain" and aborted on).

    Always returns a (order_id, fill_price_or_None, executed_qty) tuple —
    same contract as _execute_with_escalation.
    """
    FILLED_STATUSES = ("EXECUTED", "COMPLETED", "DELIVERY_AWAITED")

    if resting_order_id:
        # Re-check the resting LIMIT order first — it may have filled in the
        # moments since the last poll, in which case there's nothing to cancel.
        try:
            detail = _call(
                "non_trading", groww.get_order_detail,
                groww_order_id=resting_order_id, segment=groww.SEGMENT_FNO,
            )
            filled_qty = int(round(_get_filled_quantity(detail)))
            fill_price = float(detail.get("average_fill_price") or 0)
            status = detail.get("order_status")

            if filled_qty >= quantity or (status in FILLED_STATUSES and filled_qty == 0):
                if fill_price:
                    log.info(
                        f"{label}: resting order {resting_order_id} fully filled right "
                        "before the MARKET fallback — using this fill instead."
                    )
                    return resting_order_id, fill_price, quantity

            if filled_qty > 0:
                log.warning(
                    f"{label}: resting order {resting_order_id} shows a PARTIAL "
                    f"fill ({filled_qty}/{quantity}) right before MARKET fallback. "
                    "NOT cancelling or placing a MARKET order on top of it. "
                    "VERIFY ACTUAL POSITION ON GROWW."
                )
                return resting_order_id, fill_price or None, filled_qty
        except Exception as e:
            log.warning(f"{label}: pre-cancel check failed for {resting_order_id}: {e}")

        # Cancel the resting LIMIT order before placing the MARKET order —
        # never want two live orders for the same exit at once.
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
            return resting_order_id, None, 0
    else:
        log.info(f"{label}: no resting order to check/cancel — going straight to MARKET.")

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
            return resting_order_id, None, 0
    except Exception as e:
        log.error(
            f"{label}: MARKET fallback order placement failed: {e}. Original "
            f"LIMIT order {resting_order_id} was already cancelled — this "
            "product/segment may not support MARKET orders. VERIFY ACTUAL "
            "POSITION ON GROWW.",
            exc_info=True,
        )
        return resting_order_id, None, 0

    # Poll the MARKET order for a fill.
    for _ in range(MARKET_FALLBACK_POLL_RETRIES):
        try:
            detail = _call(
                "non_trading", groww.get_order_detail,
                groww_order_id=market_order_id, segment=groww.SEGMENT_FNO,
            )
            status = detail.get("order_status")
            filled_qty = int(round(_get_filled_quantity(detail)))
            fill_price = float(detail.get("average_fill_price") or 0)

            if filled_qty >= quantity or (status in FILLED_STATUSES and filled_qty == 0):
                if fill_price:
                    log.info(
                        f"{label}: MARKET fallback order {market_order_id} filled "
                        f"qty={quantity}  avg_fill_price=₹{fill_price:.2f}  status={status}"
                    )
                    return market_order_id, fill_price, quantity

            if status in ("REJECTED", "FAILED", "CANCELLED"):
                if filled_qty > 0:
                    log.warning(
                        f"{label}: MARKET fallback order {market_order_id} ended status={status} "
                        f"with PARTIAL fill ({filled_qty}/{quantity}) @ ₹{fill_price:.2f}."
                    )
                    return market_order_id, fill_price or None, filled_qty
                log.warning(
                    f"{label}: MARKET fallback order {market_order_id} ended "
                    f"status={status} without a confirmed fill. Original LIMIT "
                    "order was already cancelled. VERIFY ACTUAL POSITION ON GROWW."
                )
                return market_order_id, None, 0

            if 0 < filled_qty < quantity:
                log.info(
                    f"{label}: MARKET fallback order {market_order_id} shows a "
                    f"partial fill ({filled_qty}/{quantity}) — waiting for remainder to execute..."
                )
        except Exception as e:
            log.warning(f"{label}: error polling MARKET fallback order {market_order_id}: {e}")
        time.sleep(MARKET_FALLBACK_POLL_INTERVAL)

    try:
        final_detail = _call(
            "non_trading", groww.get_order_detail,
            groww_order_id=market_order_id, segment=groww.SEGMENT_FNO,
        )
        final_qty = int(round(_get_filled_quantity(final_detail)))
        final_price = float(final_detail.get("average_fill_price") or 0)
        if final_qty > 0:
            log.warning(
                f"{label}: MARKET fallback order {market_order_id} finished polling with "
                f"partial fill ({final_qty}/{quantity}) @ ₹{final_price:.2f}."
            )
            return market_order_id, final_price or None, final_qty
    except Exception:
        pass

    log.warning(
        f"{label}: MARKET fallback order {market_order_id} still not confirmed "
        f"filled after {MARKET_FALLBACK_POLL_RETRIES} polls. Treating as "
        "PENDING/UNCONFIRMED. Manual check on Groww recommended."
    )
    return market_order_id, None, 0


def _execute_with_escalation(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    side: str,          # "SELL" or "BUY"
    label: str,
    use_market_fallback: bool = False,
) -> tuple[str, float | None, int]:
    """
    Place a LIMIT order, escalating the price offset from LTP across
    ESCALATION_OFFSETS until the entire target quantity fills or all offsets are exhausted.

    If an order partially fills (e.g. 30 of 60), the unfilled resting portion is cancelled,
    and a NEW order is immediately placed for the remaining quantity (e.g. 30) at the
    next escalation offset instead of continuing to trade with only partial quantity.

    For SELL: price = LTP - offset (sell into the bid, more aggressive with larger offset)
    For BUY:  price = LTP + offset (buy through the ask, more aggressive with larger offset)

    Returns (order_id, weighted_fill_price_or_None, total_executed_quantity).
    """
    FILLED_STATUSES = ("EXECUTED", "COMPLETED", "DELIVERY_AWAITED")
    DEAD_STATUSES   = ("REJECTED", "FAILED", "CANCELLED")

    target_quantity = quantity
    remaining_qty   = quantity
    fills: list[tuple[int, float, str]] = []  # [(qty, fill_price, order_id), ...]
    last_order_id = ""
    last_live_order_id = ""

    for i, offset in enumerate(ESCALATION_OFFSETS):
        if remaining_qty <= 0:
            break

        is_last_offset = (i == len(ESCALATION_OFFSETS) - 1)
        order_qty = remaining_qty
        try:
            ltp = get_option_ltp(groww, trading_symbol)
            if side == "SELL":
                price = round(max(0.05, ltp - offset), 2)
            else:
                price = round(ltp + offset, 2)

            resp = _place_limit_order(
                groww, trading_symbol, transaction_type, order_qty, price,
                f"{label} (offset=₹{offset}, attempt={i + 1}/{len(ESCALATION_OFFSETS)}, qty={order_qty}/{target_quantity})"
            )
            order_id = resp.get("groww_order_id", "")
            last_order_id = order_id or last_order_id
            last_live_order_id = order_id or last_live_order_id

            if not order_id:
                log.warning(f"{label}: place_order returned no groww_order_id. Response: {resp}")
                continue

            partial_fill_seen    = False
            partial_fill_qty     = 0
            order_confirmed_dead = False

            for _ in range(FILL_WAIT_RETRIES):
                try:
                    detail = _call(
                        "non_trading", groww.get_order_detail,
                        groww_order_id=order_id, segment=groww.SEGMENT_FNO,
                    )
                    status     = detail.get("order_status")
                    filled_qty = int(round(_get_filled_quantity(detail)))
                    fill_price = float(detail.get("average_fill_price") or 0)

                    # 1. Full fill of this order chunk
                    if filled_qty >= order_qty or (status in FILLED_STATUSES and filled_qty == 0):
                        actual_price = fill_price or price
                        fills.append((order_qty, actual_price, order_id))
                        remaining_qty -= order_qty
                        last_live_order_id = ""
                        log.info(
                            f"{label}: order {order_id} filled {order_qty} @ ₹{actual_price:.2f} "
                            f"(offset=₹{offset}). Total filled: {target_quantity - remaining_qty}/{target_quantity}"
                        )
                        break

                    # 2. Terminal dead status
                    if status in DEAD_STATUSES:
                        if filled_qty > 0:
                            actual_price = fill_price or price
                            fills.append((filled_qty, actual_price, order_id))
                            remaining_qty -= filled_qty
                            log.warning(
                                f"{label}: order {order_id} ended status={status} with partial fill "
                                f"({filled_qty}/{order_qty}) @ ₹{actual_price:.2f}. "
                                f"Remaining quantity to fill: {remaining_qty}"
                            )
                        else:
                            log.warning(f"{label}: order {order_id} ended status={status} before fill.")
                        order_confirmed_dead = True
                        last_live_order_id = ""
                        break

                    # 3. Terminal executed status on broker with partial quantity
                    if status in FILLED_STATUSES and 0 < filled_qty < order_qty:
                        actual_price = fill_price or price
                        fills.append((filled_qty, actual_price, order_id))
                        remaining_qty -= filled_qty
                        order_confirmed_dead = True
                        last_live_order_id = ""
                        log.warning(
                            f"{label}: order {order_id} marked {status} with partial fill "
                            f"({filled_qty}/{order_qty}) @ ₹{actual_price:.2f}. "
                            f"Remaining quantity to fill: {remaining_qty}"
                        )
                        break

                    # 4. Partial fill in-flight (order still open/active)
                    if 0 < filled_qty < order_qty:
                        partial_fill_seen = True
                        partial_fill_qty  = filled_qty
                        log.info(
                            f"{label}: order {order_id} has partial fill ({filled_qty}/{order_qty}), "
                            "waiting for remaining quantity to execute..."
                        )

                except Exception as e:
                    log.warning(f"Error polling order {order_id}: {e}")
                time.sleep(FILL_WAIT_INTERVAL)

            # If remaining_qty is now 0 (fully filled during polling), done!
            if remaining_qty <= 0:
                break

            # If partial fill was seen during polling and order is still resting:
            # Cancel the remaining unfilled portion and immediately escalate for remaining_qty
            if partial_fill_seen and not order_confirmed_dead:
                try:
                    _call(
                        "orders", groww.cancel_order,
                        groww_order_id=order_id, segment=groww.SEGMENT_FNO,
                    )
                    log.info(
                        f"{label}: cancelled remaining unfilled portion of partially filled order {order_id}."
                    )
                except Exception as e:
                    log.warning(f"{label}: failed to cancel remaining unfilled portion of {order_id}: {e}")

                try:
                    final_detail = _call(
                        "non_trading", groww.get_order_detail,
                        groww_order_id=order_id, segment=groww.SEGMENT_FNO,
                    )
                    final_qty = int(round(_get_filled_quantity(final_detail))) or partial_fill_qty
                    final_price = float(final_detail.get("average_fill_price") or 0) or price
                except Exception:
                    final_qty = partial_fill_qty
                    final_price = price

                fills.append((final_qty, final_price, order_id))
                remaining_qty -= final_qty
                last_live_order_id = ""

                log.warning(
                    f"{label}: order {order_id} partially filled ({final_qty}/{order_qty}) @ ₹{final_price:.2f}. "
                    f"Remaining quantity needed: {remaining_qty}. Immediately placing new order at next offset..."
                )

                if remaining_qty <= 0:
                    break
                continue

            if order_confirmed_dead:
                if remaining_qty <= 0:
                    break
                if is_last_offset:
                    if use_market_fallback and EXIT_MARKET_FALLBACK_ENABLED:
                        log.warning(
                            f"{label}: order {order_id} was dead at widest offset (₹{offset}). "
                            f"Going straight to MARKET fallback for remaining {remaining_qty}."
                        )
                        m_id, m_price, m_qty = _market_fallback(
                            groww, trading_symbol, transaction_type, remaining_qty, label, ""
                        )
                        if m_qty > 0:
                            fills.append((m_qty, m_price or price, m_id))
                            remaining_qty -= m_qty
                        break
                    log.warning(
                        f"{label}: order {order_id} was dead at widest offset (₹{offset}). "
                        f"Escalation exhausted. Total filled: {target_quantity - remaining_qty}/{target_quantity}."
                    )
                    break
                log.info(
                    f"{label}: order {order_id} dead at offset=₹{offset} — moving to next offset for remaining {remaining_qty}."
                )
                continue

            # If not filled and not dead at last offset:
            if is_last_offset:
                if use_market_fallback and EXIT_MARKET_FALLBACK_ENABLED:
                    log.warning(
                        f"{label}: order {order_id} not filled even at widest offset (₹{offset}). "
                        f"Attempting fallback — cancel and place MARKET order for remaining {remaining_qty}."
                    )
                    m_id, m_price, m_qty = _market_fallback(
                        groww, trading_symbol, transaction_type, remaining_qty, label, order_id
                    )
                    if m_qty > 0:
                        fills.append((m_qty, m_price or price, m_id))
                        remaining_qty -= m_qty
                    break

                # For entry on last offset: cancel resting order to avoid stale fills later
                try:
                    _call(
                        "orders", groww.cancel_order,
                        groww_order_id=order_id, segment=groww.SEGMENT_FNO,
                    )
                    log.info(f"{label}: cancelled unfilled order {order_id} at widest offset (₹{offset}).")
                    last_live_order_id = ""
                except Exception as e:
                    log.warning(f"{label}: failed to cancel unfilled order {order_id} at widest offset: {e}")

                try:
                    post_cancel = _call(
                        "non_trading", groww.get_order_detail,
                        groww_order_id=order_id, segment=groww.SEGMENT_FNO,
                    )
                    post_qty = int(round(_get_filled_quantity(post_cancel)))
                    post_price = float(post_cancel.get("average_fill_price") or 0)
                    if post_qty > 0:
                        p_price = post_price or price
                        fills.append((post_qty, p_price, order_id))
                        remaining_qty -= post_qty
                        log.warning(f"{label}: order {order_id} filled {post_qty} right at cancel time @ ₹{p_price:.2f}.")
                except Exception:
                    pass
                break

            # Not last offset: cancel resting order and proceed to next offset for remaining_qty
            try:
                _call(
                    "orders", groww.cancel_order,
                    groww_order_id=order_id, segment=groww.SEGMENT_FNO,
                )
                log.info(f"{label}: order {order_id} not filled at offset=₹{offset} — cancelled, retrying wider.")
                last_live_order_id = ""
            except Exception as e:
                log.warning(
                    f"{label}: failed to cancel unfilled order {order_id} (offset=₹{offset}): {e}. "
                    "Order state is uncertain — halting escalation to avoid duplicate orders. "
                    "VERIFY ACTUAL POSITION ON GROWW."
                )
                break

            # Re-check right after cancel ack in case order executed right at cancel time
            try:
                post_cancel = _call(
                    "non_trading", groww.get_order_detail,
                    groww_order_id=order_id, segment=groww.SEGMENT_FNO,
                )
                post_cancel_qty = int(round(_get_filled_quantity(post_cancel)))
                post_fill_price = float(post_cancel.get("average_fill_price") or 0)
                if post_cancel_qty > 0:
                    p_price = post_fill_price or price
                    fills.append((post_cancel_qty, p_price, order_id))
                    remaining_qty -= post_cancel_qty
                    log.warning(
                        f"{label}: order {order_id} actually filled {post_cancel_qty} right at cancel time @ ₹{p_price:.2f}. "
                        f"Remaining quantity to fill: {remaining_qty}."
                    )
                    if remaining_qty <= 0:
                        break
            except Exception as e:
                log.warning(f"{label}: post-cancel check failed for order {order_id}: {e}")

        except Exception as exc:
            if last_live_order_id:
                log.error(
                    f"{label}: unexpected error while order {last_live_order_id} may still be "
                    f"live (offset attempt {i + 1}/{len(ESCALATION_OFFSETS)}): {exc}. Halting "
                    "escalation to avoid duplicate positions. VERIFY ACTUAL POSITION ON GROWW.",
                    exc_info=True,
                )
                break
            log.error(
                f"{label}: unexpected error with no order currently live "
                f"(offset attempt {i + 1}/{len(ESCALATION_OFFSETS)}): {exc}.",
                exc_info=True,
            )
            raise

    # ── Aggregate fills across all attempts ───────────────────────────
    total_filled = sum(q for q, p, o in fills)
    if total_filled > 0:
        valid_prices = [(q, p) for q, p, o in fills if p and p > 0]
        avg_price = round(sum(q * p for q, p in valid_prices) / sum(q for q, p in valid_prices), 2) if valid_prices else None
        primary_order_id = fills[0][2] if fills else last_order_id
        if total_filled >= target_quantity:
            price_str = f"₹{avg_price:.2f}" if avg_price else "unknown"
            log.info(
                f"{label}: FULLY FILLED across escalation attempts — "
                f"qty={total_filled}/{target_quantity}  avg_fill_price={price_str}"
            )
        else:
            price_str = f"₹{avg_price:.2f}" if avg_price else "unknown"
            log.warning(
                f"{label}: PARTIALLY FILLED across escalation attempts — "
                f"qty={total_filled}/{target_quantity}  avg_fill_price={price_str}. "
                "Remaining quantity will be topped up if conditions permit."
            )
        return primary_order_id, avg_price, total_filled

    return last_order_id, None, 0


def sell_option(groww: GrowwAPI, trading_symbol: str, quantity: int, label: str) -> tuple[str, float | None, int]:
    """SELL to open, escalating the LIMIT price offset (₹2 → ₹5 → ₹10 → ₹15) until filled.
    If an order partially fills, a new order is immediately placed for the remaining quantity.
    Returns (order_id, fill_price, executed_quantity)."""
    return _execute_with_escalation(
        groww, trading_symbol, groww.TRANSACTION_TYPE_SELL, quantity, "SELL", f"SELL {label}"
    )


def buy_to_cover_option(groww: GrowwAPI, trading_symbol: str, quantity: int, label: str) -> tuple[str, float | None, int]:
    """
    BUY to cover a short, escalating the LIMIT price offset (₹2 → ₹5 → ₹10 → ₹15)
    until filled. If an order partially fills, a new order is immediately placed for the
    remaining quantity. If all LIMIT offsets fail, a fallback cancels resting order and fires
    a MARKET order (see EXIT_MARKET_FALLBACK_ENABLED).
    Returns (order_id, fill_price, executed_quantity).
    """
    return _execute_with_escalation(
        groww, trading_symbol, groww.TRANSACTION_TYPE_BUY, quantity, "BUY", f"BUY (cover) {label}",
        use_market_fallback=True,
    )


def switch_strike_on_profit(
    groww: GrowwAPI,
    pos: dict,
    option_type: str,   # "PE" or "CE"
    leg_label: str,     # "PUT SELL" or "CALL SELL"
    instruments_df: pd.DataFrame,
    expiry: pd.Timestamp,
    fut_symbol: str,
) -> dict | None:
    """
    Called when an open short leg has accumulated >= PROFIT_SWITCH_THRESHOLD
    points of profit (entry_price − current_LTP for a short position).

    Workflow:
      1. Buy-to-cover the current strike (exit).
      2. Fetch the live BankNifty futures LTP and compute the new ATM strike.
      3. Sell the new ATM strike (same option_type) to re-enter.

    Returns:
      - A new pos dict (entry_price = new fill, or None if unconfirmed) if
        the re-entry order was placed (even if fill is unconfirmed).
      - None if the cover fill itself was unconfirmed (leg kept "exit_pending")
        or if the re-entry definitively failed.  Callers must keep the
        returned value to track the new (or pending-exit) position.

    If the cover order fill is UNCONFIRMED, the old pos is returned with
    exit_pending=True (same as attempt_square_off), and NO re-entry is
    attempted — we must not open a new short on top of an unresolved one.
    """
    symbol      = pos["symbol"]
    entry_price = pos.get("entry_price", 0.0)
    quantity    = pos.get("quantity") or (QUANTITY * pos["lot_size"])

    log.info(
        f"[PROFIT SWITCH] {leg_label}  symbol={symbol}  qty={quantity}  "
        f"entry=₹{entry_price:.2f}  profit >= {PROFIT_SWITCH_THRESHOLD} pts  "
        "→ Exiting current strike and re-entering new ATM strike."
    )

    # ── Step 1: Buy-to-cover current strike ─────────────────────────────────
    cover_order_id, cover_fill, cover_qty = buy_to_cover_option(groww, symbol, quantity, leg_label)
    exit_time = datetime.now(IST)
    actual_cover_qty = cover_qty if cover_qty > 0 else quantity

    if cover_fill is None and cover_order_id:
        # Cover fill is unconfirmed — keep leg open with exit_pending.
        # Do NOT attempt a re-entry on top of an unresolved short.
        pnl_est = None
        try:
            exit_ltp = get_option_ltp(groww, symbol)
            pnl_est  = (entry_price - exit_ltp) * actual_cover_qty if entry_price else None
        except Exception:
            pass
        log.warning(
            f"[PROFIT SWITCH] {leg_label}: cover order {cover_order_id} fill UNCONFIRMED — "
            "leg kept OPEN (exit_pending=True). Re-entry skipped until cover resolves. "
            "VERIFY ACTUAL POSITION ON GROWW."
        )
        log_trade(
            leg=leg_label, action="EXIT", symbol=symbol, quantity=actual_cover_qty,
            price=None, fill_confirmed=False, order_id=cover_order_id,
            reason=f"Profit switch (>= {PROFIT_SWITCH_THRESHOLD} pts) — cover UNCONFIRMED",
            pnl=pnl_est, when=exit_time,
        )
        pos["exit_pending"]  = True
        pos["exit_order_id"] = cover_order_id
        pos["exit_reason"]   = f"profit switch >= {PROFIT_SWITCH_THRESHOLD} pts"
        pos["exit_quantity"] = actual_cover_qty
        return pos

    if not cover_order_id:
        # Every cover attempt definitively rejected — unclear state, bail.
        log.error(
            f"[PROFIT SWITCH] {leg_label}: cover order was rejected for all offsets — "
            "cannot safely re-enter. Leaving leg unchanged."
        )
        return pos

    # Cover confirmed
    cover_price = cover_fill  # confirmed fill
    pnl = (entry_price - cover_price) * actual_cover_qty if entry_price and cover_price else None
    log.info(
        f"[PROFIT SWITCH] {leg_label}: cover confirmed  symbol={symbol}  qty={actual_cover_qty}  "
        f"entry=₹{entry_price:.2f}  cover=₹{cover_price:.2f}  "
        f"P&L=₹{pnl:.2f}  time={exit_time:%Y-%m-%d %H:%M:%S}"
    )
    log_trade(
        leg=leg_label, action="EXIT", symbol=symbol, quantity=actual_cover_qty,
        price=cover_price, fill_confirmed=True, order_id=cover_order_id,
        reason=f"Profit switch (>= {PROFIT_SWITCH_THRESHOLD} pts)",
        pnl=pnl, when=exit_time,
    )

    # ── Step 2: Re-enter on new ATM strike ──────────────────────────────────
    try:
        bn_ltp  = get_banknifty_ltp(groww, fut_symbol)
        new_atm = get_atm_strike(bn_ltp)
        log.info(
            f"[PROFIT SWITCH] {leg_label}: BankNifty LTP={bn_ltp:.2f}  "
            f"new ATM strike={new_atm}  →  Re-selling {option_type} (min premium ₹{MIN_OPTION_PRICE})"
        )
        new_sym, new_lot, new_ltp_check = find_option_with_min_price(
            groww, instruments_df, expiry, new_atm, option_type
        )
    except Exception as e:
        log.error(
            f"[PROFIT SWITCH] {leg_label}: failed to resolve new ATM strike after cover: {e}. "
            "Re-entry skipped — leg is now FLAT."
        )
        return None

    new_qty = QUANTITY * new_lot
    new_order_id, new_fill, new_filled_qty = sell_option(groww, new_sym, new_qty, leg_label)
    re_entry_time = datetime.now(IST)
    actual_re_entry_qty = new_filled_qty if new_filled_qty > 0 else new_qty

    if not new_order_id:
        log.error(
            f"[PROFIT SWITCH] {leg_label}: re-entry sell order FAILED for all offsets — "
            f"symbol={new_sym}  qty={new_qty}. Leg is FLAT after the cover."
        )
        log_trade(
            leg=leg_label, action="ENTRY", symbol=new_sym, quantity=new_qty,
            price=new_ltp_check, fill_confirmed=False, order_id="",
            reason=f"Profit switch re-entry — FAILED", when=re_entry_time,
        )
        return None

    new_pos = {
        "symbol":      new_sym,
        "lot_size":    new_lot,
        "quantity":    actual_re_entry_qty,
        "entry_price": new_fill,    # None if unconfirmed
        "entry_time":  re_entry_time,
        "order_id":    new_order_id,
    }
    if new_fill is not None:
        log.info(
            f"[PROFIT SWITCH] {leg_label}: re-entry confirmed  symbol={new_sym}  "
            f"strike={new_atm}  fill=₹{new_fill:.2f}  qty={actual_re_entry_qty}  "
            f"time={re_entry_time:%Y-%m-%d %H:%M:%S}  order_id={new_order_id}"
        )
    else:
        log.warning(
            f"[PROFIT SWITCH] {leg_label}: re-entry order placed but fill UNCONFIRMED  "
            f"symbol={new_sym}  qty={actual_re_entry_qty}  time={re_entry_time:%Y-%m-%d %H:%M:%S}  "
            f"order_id={new_order_id}. VERIFY ACTUAL POSITION ON GROWW."
        )
    log_trade(
        leg=leg_label, action="ENTRY", symbol=new_sym, quantity=actual_re_entry_qty,
        price=new_fill if new_fill is not None else new_ltp_check,
        fill_confirmed=new_fill is not None, order_id=new_order_id,
        reason=f"Profit switch re-entry (>= {PROFIT_SWITCH_THRESHOLD} pts on prev strike)",
        when=re_entry_time,
    )
    return new_pos


# ─── ORDER FILL PRICE POLLER ─────────────────────────────────────────────────
def _await_fill_price(groww: GrowwAPI, order_id: str, retries: int = 6) -> tuple[float | None, int]:
    """
    Poll order detail until the order is filled or partially filled.
    Returns (average_fill_price, filled_quantity).
    """
    FILLED_STATUSES = ("EXECUTED", "COMPLETED", "DELIVERY_AWAITED")
    for _ in range(retries):
        try:
            detail = _call(
                "non_trading", groww.get_order_detail,
                groww_order_id=order_id,
                segment=groww.SEGMENT_FNO,
            )
            filled_qty = int(round(_get_filled_quantity(detail)))
            fill_price = float(detail.get("average_fill_price") or 0)
            status     = detail.get("order_status")
            if (status in FILLED_STATUSES or filled_qty > 0) and fill_price > 0:
                return fill_price, filled_qty
        except Exception as e:
            log.warning(f"Error polling order {order_id}: {e}")
        time.sleep(0.5)
    log.warning(
        f"Order {order_id} not filled after {retries} polls — using LTP as fallback."
    )
    return None, 0


# ─── MARKET STATUS ────────────────────────────────────────────────────────────
def market_status(now_hm: str, weekday: int | None = None) -> str:
    """Returns 'CLOSED', 'SQUAREOFF', or 'OPEN'.

    weekday: datetime.weekday() value (0=Mon … 6=Sun).  If None, computed
    from the current IST time.  Saturday (5) and Sunday (6) always return
    'CLOSED' regardless of the clock time.
    """
    if weekday is None:
        weekday = datetime.now(IST).weekday()
    if weekday >= 5:          # 5 = Saturday, 6 = Sunday
        return "CLOSED"
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
    pos dict: {"symbol": str, "quantity": int, "entry_price": float, "order_id": str}

    Returns a trade record dict suitable for the trade journal:
      {"leg": leg_label, "symbol": ..., "quantity": ...,
       "entry_price": ..., "exit_price": ..., "exit_order_id": ...,
       "exit_fill_confirmed": bool, "pnl": float | None, "reason": ...}
    """
    symbol      = pos["symbol"]
    entry_price = pos.get("entry_price", 0.0)
    entry_time  = pos.get("entry_time")
    entry_str   = f"₹{entry_price:.2f}" if entry_price else "unknown"
    log.warning(f"[{reason}] Covering short {leg_label}  qty={quantity}  entry={entry_str}  symbol={symbol}")

    order_id, fill_price, filled_qty = buy_to_cover_option(groww, symbol, quantity, leg_label)
    exit_time            = datetime.now(IST)
    exit_fill_confirmed  = fill_price is not None and (filled_qty >= quantity or filled_qty == 0)
    exit_price           = fill_price if fill_price is not None else get_option_ltp(groww, symbol)
    actual_exit_qty      = filled_qty if filled_qty > 0 else quantity

    hold_str = ""
    if entry_time is not None:
        hold_str = f"  held={str(exit_time - entry_time).split('.')[0]}"

    if not exit_fill_confirmed:
        log.warning(
            f"    {leg_label} cover order {order_id} fill UNCONFIRMED (filled {filled_qty}/{quantity}) — "
            f"using LTP ₹{exit_price:.2f} for logging only. "
            "VERIFY ACTUAL POSITION ON GROWW."
        )

    pnl = (entry_price - exit_price) * actual_exit_qty if entry_price else None

    if entry_price:
        log.info(
            f"[TRADE] {leg_label} EXIT  symbol={symbol}  qty={actual_exit_qty}  "
            f"entry=₹{entry_price:.2f}  exit=₹{exit_price:.2f}  "
            f"time={exit_time:%Y-%m-%d %H:%M:%S}{hold_str}  "
            f"P&L=₹{pnl:.2f}  fill_confirmed={exit_fill_confirmed}  "
            f"order_id={order_id}  reason={reason}"
        )
    else:
        log.info(
            f"[TRADE] {leg_label} EXIT  symbol={symbol}  qty={actual_exit_qty}  "
            f"exit=₹{exit_price:.2f}  time={exit_time:%Y-%m-%d %H:%M:%S}{hold_str}  "
            f"P&L=N/A  fill_confirmed={exit_fill_confirmed}  "
            f"order_id={order_id}  reason={reason}"
        )

    log_trade(
        leg=leg_label, action="EXIT", symbol=symbol, quantity=actual_exit_qty,
        price=exit_price, fill_confirmed=exit_fill_confirmed, order_id=order_id,
        reason=reason, pnl=pnl, when=exit_time,
    )

    return {
        "leg":                  leg_label,
        "symbol":               symbol,
        "quantity":             actual_exit_qty,
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

    # If a partial fill occurred during exit, reduce the open quantity accordingly
    covered_qty = trade.get("quantity", 0)
    current_qty = pos.get("quantity", quantity)
    if 0 < covered_qty < current_qty:
        pos["quantity"] = current_qty - covered_qty
        log.warning(
            f"{leg_label}: partial exit executed ({covered_qty} covered). "
            f"Remaining open position quantity: {pos['quantity']}."
        )

    pos["exit_pending"]  = True
    pos["exit_order_id"] = trade["exit_order_id"]
    pos["exit_reason"]   = reason
    pos["exit_quantity"] = covered_qty
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
    # Guard against off-market crash-restart loops: if market is CLOSED and
    # authentication fails (e.g. rate-limited), wait gracefully instead of crashing
    # and being immediately restarted by a watchdog or service.
    while True:
        now_ist = datetime.now(IST)
        now_hm  = now_ist.strftime("%H:%M")
        status  = market_status(now_hm)

        try:
            groww = authenticate()
            break
        except Exception as auth_err:
            # Never raise here regardless of market status — crashing during
            # market hours causes an immediate watchdog restart, which retries
            # auth again instantly and can exhaust Groww's 150 calls/day limit
            # within minutes. Always wait before retrying; use a shorter sleep
            # during market hours so we recover quickly without hammering auth.
            sleep_s = 300 if status == "CLOSED" else 60
            log.warning(
                f"Authentication failed ({auth_err}). "
                f"Sleeping {sleep_s} s before retrying "
                f"(market status: {status})…"
            )
            time.sleep(sleep_s)

    last_auth_time = datetime.now(IST)
    # RE_AUTH_INTERVAL removed — auth errors now trigger immediate re-auth
    # via _is_auth_error() in the main except handler (no age gate needed).

    instruments_df = get_cached_instruments_df(groww)
    expiry         = get_monthly_expiry_date(instruments_df)
    fut_symbol, fut_groww_symbol = get_active_banknifty_fut_symbol(instruments_df)

    # State for each leg
    # None → flat; dict → {"symbol": str, "lot_size": int, "entry_price": float, "order_id": str}
    put_pos:  dict | None = None
    call_pos: dict | None = None
    prev_direction: int | None = None
    # Tracks whether the "enter immediately, no flip required" first-trade
    # logic has fired yet. Deliberately separate from `prev_direction is None`
    # -- that condition is only ever true on the single very first loop
    # iteration, which usually happens before ENTRY_START_TIME (script starts
    # ~9:00, entry window opens 09:23). If the seed were tied to that first
    # iteration, it gets silently discarded when the immediate entry attempt
    # is blocked by the entry-window check, and prev_direction is overwritten
    # with the real direction at the end of that same iteration -- so by the
    # time the entry window actually opens, prev_direction is no longer None
    # and the strategy ends up waiting for a genuine flip instead of entering
    # on the current trend as intended.
    first_trade_done: bool = False

    log.info(
        f"Strategy started.  Underlying: {UNDERLYING}  "
        f"Monthly expiry: {expiry.date()}  "
        f"Supertrend({ST_LENGTH}, {ST_FACTOR})  "
        f"Exit threshold: ±{EXIT_POINTS_THRESHOLD} pts  "
        f"Entry window: {ENTRY_START_TIME}–{ENTRY_END_TIME}  "
        f"Force-close: {SQUARE_OFF_TIME}"
    )

    def _leg_status_label(groww_client: GrowwAPI, leg_pos: dict | None, leg_name: str) -> str:
        """Build a per-minute status string for one leg, including the
        LIVE LTP whenever a position is open.

        Defined once outside the main loop so Python doesn't recreate the
        function object on every 60-second iteration (~900 times/day).
        groww_client is passed explicitly so this works correctly after a
        mid-session re-authentication (where groww may be reassigned).
        Never lets an LTP-fetch hiccup break the per-minute summary log —
        falls back to entry-only text on any error."""
        if not leg_pos or leg_pos.get("entry_price") is None:
            return f"{leg_name}: FLAT"
        entry = leg_pos["entry_price"]
        qty_str = f" (qty={leg_pos['quantity']})" if leg_pos.get("quantity") else ""
        try:
            live_ltp = get_option_ltp(groww_client, leg_pos["symbol"])
            diff     = live_ltp - entry   # positive = loss for the seller
            return (
                f"{leg_name} SELL{qty_str} entry=₹{entry:.2f}  LTP=₹{live_ltp:.2f}  "
                f"diff={diff:+.2f}"
            )
        except Exception as e:
            return f"{leg_name} SELL{qty_str} entry=₹{entry:.2f}  LTP=unavailable ({e})"

    while True:
        try:
            now_ist    = datetime.now(IST)
            now_ts     = now_ist.strftime("%Y-%m-%d %H:%M:%S")
            now_hm     = now_ist.strftime("%H:%M")
            now_wd     = now_ist.weekday()   # 0=Mon … 6=Sun
            iter_start = time.monotonic()
            status     = market_status(now_hm, weekday=now_wd)

            # ── Outside market hours ──────────────────────────────────────────
            if status == "CLOSED":
                now_ist_c = datetime.now(IST)

                # Pre-market warm-up window: strategy has woken (>= STRATEGY_WAKE_TIME)
                # but the exchange is not yet open (< MARKET_OPEN, typically 09:00–09:14).
                # In this window we do a short 30-second sleep and keep looping so we
                # are ready to trade the moment MARKET_OPEN is reached — NOT a 24-hour sleep.
                if now_wd < 5 and now_hm >= STRATEGY_WAKE_TIME and now_hm < MARKET_OPEN:
                    log.info(
                        f"[{now_ts}] Pre-market ({now_hm}). "
                        f"Waiting for market open at {MARKET_OPEN} IST — sleeping 30 s."
                    )
                    time.sleep(30)
                    continue

                # Weekend or genuine overnight closure: sleep until STRATEGY_WAKE_TIME
                # on the next trading day (Monday if today is Fri/Sat/Sun).
                # This means zero API calls over the entire weekend.
                wh, wm     = (int(p) for p in STRATEGY_WAKE_TIME.split(":"))
                wake_today = now_ist_c.replace(hour=wh, minute=wm, second=0, microsecond=0)
                # Move forward day-by-day until we land on a weekday whose wake
                # time is still in the future.
                candidate  = wake_today if now_ist_c < wake_today else wake_today + timedelta(days=1)
                while candidate.weekday() >= 5:   # skip Saturday (5) and Sunday (6)
                    candidate += timedelta(days=1)
                wake_at    = candidate
                sleep_s    = max(1.0, (wake_at - now_ist_c).total_seconds())
                log.info(
                    f"[{now_ts}] Market CLOSED. "
                    f"Sleeping {sleep_s / 3600:.1f} h until "
                    f"{wake_at.strftime('%Y-%m-%d %H:%M')} IST — "
                    "no API calls until then."
                )
                time.sleep(sleep_s)
                # Reset candle buffer so the next cycle after waking does a
                # full multi-day seed fetch rather than a stale incremental one.
                reset_candle_buffer()
                continue

            # ── Force square-off at SQUARE_OFF_TIME ───────────────────────────
            if status == "SQUAREOFF":
                had_positions = put_pos is not None or call_pos is not None
                if put_pos is not None:
                    try:
                        trade_qty = put_pos.get("quantity") or (QUANTITY * put_pos["lot_size"])
                        put_pos = attempt_square_off(
                            groww, put_pos, trade_qty,
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
                        trade_qty = call_pos.get("quantity") or (QUANTITY * call_pos["lot_size"])
                        call_pos = attempt_square_off(
                            groww, call_pos, trade_qty,
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

            put_label  = _leg_status_label(groww, put_pos, "PUT")
            call_label = _leg_status_label(groww, call_pos, "CALL")
            log.info(
                f"[{now_ts}]  ST={trend_label}  BankNifty Close={last_close:.2f}  "
                f"|  {put_label}  |  {call_label}"
            )

            entry_allowed = ENTRY_START_TIME <= now_hm <= ENTRY_END_TIME

            # First trade of the day: enter immediately on the current Supertrend
            # direction once the entry window is actually open — no flip required.
            # Gated on entry_allowed (not on prev_direction is None) so that if the
            # script starts before ENTRY_START_TIME, the immediate-entry attempt
            # waits until the window opens instead of firing early, getting
            # blocked, and being silently lost for the rest of the day.
            if not first_trade_done:
                if entry_allowed:
                    prev_direction   = -1 if curr_direction == 1 else 1
                    first_trade_done = True
                    log.info(
                        f"Entry window open — current Supertrend is {trend_label}. "
                        "Entering trade immediately based on current trend direction "
                        "(no flip required for the first trade of the day)."
                    )
                elif prev_direction is None:
                    # Before the entry window opens, just start tracking direction —
                    # don't let this look like a "flip" once first_trade_done fires.
                    prev_direction = curr_direction

            # ── Recover missing entry prices ──────────────────────────────────
            # A None entry_price means the entry order from _execute_with_escalation
            # was left resting (unfilled even at the widest offset). Poll again here;
            # if it has since filled, record the confirmed fill price. If still
            # unfilled, fall back to LTP for logging but keep flagging it as
            # unconfirmed so square_off_position will warn appropriately.
            for leg_pos, leg_name in [(put_pos, "PUT"), (call_pos, "CALL")]:
                if leg_pos is not None and leg_pos.get("entry_price") is None:
                    try:
                        confirmed_price, confirmed_qty = _await_fill_price(groww, leg_pos["order_id"])
                        if confirmed_price is not None:
                            leg_pos["entry_price"] = confirmed_price
                            if confirmed_qty > 0:
                                leg_pos["quantity"] = confirmed_qty
                            log.info(
                                f"[TRADE] {leg_name} ENTRY_RECOVERED  "
                                f"symbol={leg_pos['symbol']}  qty={leg_pos.get('quantity')}  "
                                f"fill=₹{confirmed_price:.2f}  "
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
                    confirmed_price, confirmed_qty = _await_fill_price(groww, leg_pos["exit_order_id"])
                except Exception as e:
                    confirmed_price, confirmed_qty = None, 0
                    log.warning(f"Could not resolve pending {leg_name} exit: {e}")

                if confirmed_price is None:
                    log.warning(
                        f"{leg_label}: exit order {leg_pos['exit_order_id']} still "
                        "unresolved — leg remains OPEN; not re-attempting a new cover "
                        "order until this resolves."
                    )
                    continue

                entry_price = leg_pos.get("entry_price")
                exit_qty    = confirmed_qty if confirmed_qty > 0 else leg_pos.get("exit_quantity", leg_pos.get("quantity", QUANTITY * leg_pos["lot_size"]))
                pnl         = (entry_price - confirmed_price) * exit_qty if entry_price else None
                entry_str   = f"₹{entry_price:.2f}" if entry_price else "unknown"
                pnl_str     = f"₹{pnl:.2f}" if pnl is not None else "N/A"
                log.info(
                    f"[TRADE] {leg_label} EXIT CONFIRMED (was pending)  "
                    f"symbol={leg_pos['symbol']}  qty={exit_qty}  entry={entry_str}  "
                    f"exit=₹{confirmed_price:.2f}  P&L={pnl_str}  "
                    f"order_id={leg_pos['exit_order_id']}  reason={leg_pos.get('exit_reason')}"
                )
                log_trade(
                    leg=leg_label, action="EXIT", symbol=leg_pos["symbol"], quantity=exit_qty,
                    price=confirmed_price, fill_confirmed=True, order_id=leg_pos["exit_order_id"],
                    reason=f"{leg_pos.get('exit_reason')} (confirmed on recheck)", pnl=pnl,
                )
                remaining_open = leg_pos.get("quantity", 0) - exit_qty
                if remaining_open <= 0:
                    if leg_name == "PUT":
                        put_pos = None
                    else:
                        call_pos = None
                else:
                    leg_pos["quantity"] = remaining_open
                    leg_pos["exit_pending"] = False
                    log.warning(
                        f"{leg_label}: exit confirmed {exit_qty} shares, but {remaining_open} still remain open."
                    )

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
                    order_id, fill_price, filled_qty = sell_option(groww, put_sym, trade_qty, "PUT")
                    entry_time = datetime.now(IST)
                    if not order_id:
                        # BUG FIX (found 27-Aug-2026): an empty order_id here
                        # means _execute_with_escalation confirmed every
                        # offset attempt was REJECTED/FAILED/CANCELLED — NOT
                        # "unconfirmed", DEFINITIVELY no position exists on
                        # the exchange. The old code still built a put_pos
                        # dict in this case (order_id was always truthy —
                        # it held the id of the rejected order), so the
                        # strategy believed it held a short PUT that was
                        # never actually opened, and would later fire a real
                        # BUY to "cover" a position that didn't exist.
                        # put_pos is deliberately left untouched (still
                        # None) so the strategy waits for the next flip
                        # instead of tracking a phantom position.
                        log.error(
                            f"PUT SELL ENTRY FAILED  symbol={put_sym}  qty={trade_qty}  "
                            f"time={entry_time:%Y-%m-%d %H:%M:%S}  reason=ST flip BULLISH  "
                            "— every offset attempt was rejected/failed. NO POSITION was "
                            "opened. Not tracking a position for this signal."
                        )
                        log_trade(
                            leg="PUT SELL", action="ENTRY", symbol=put_sym, quantity=trade_qty,
                            price=put_ltp_check, fill_confirmed=False, order_id="",
                            reason="ST flip BULLISH — ENTRY FAILED, no position opened",
                            when=entry_time,
                        )
                    else:
                        actual_qty = filled_qty if filled_qty > 0 else trade_qty
                        is_partial = 0 < filled_qty < trade_qty
                        put_pos = {
                            "symbol":      put_sym,
                            "lot_size":    put_lot,
                            "quantity":    actual_qty,
                            "entry_price": fill_price,   # None if fill unconfirmed
                            "entry_time":  entry_time,
                            "order_id":    order_id,
                        }
                        partial_tag = " (PARTIAL)" if is_partial else ""
                        if fill_price is not None:
                            log.info(
                                f"[TRADE] PUT SELL ENTRY{partial_tag}  symbol={put_sym}  strike_ltp_check=₹{put_ltp_check:.2f}  "
                                f"qty={actual_qty}  fill=₹{fill_price:.2f}  time={entry_time:%Y-%m-%d %H:%M:%S}  "
                                f"order_id={order_id}  reason=ST flip BULLISH"
                            )
                        else:
                            log.warning(
                                f"[TRADE] PUT SELL ENTRY (UNCONFIRMED)  symbol={put_sym}  qty={actual_qty}  "
                                f"time={entry_time:%Y-%m-%d %H:%M:%S}  order_id={order_id}  reason=ST flip BULLISH  "
                                "— fill price unknown. VERIFY ACTUAL POSITION ON GROWW."
                            )
                        log_trade(
                            leg="PUT SELL", action="ENTRY", symbol=put_sym, quantity=actual_qty,
                            price=fill_price if fill_price is not None else put_ltp_check,
                            fill_confirmed=fill_price is not None, order_id=order_id,
                            reason=f"ST flip BULLISH{partial_tag}", when=entry_time,
                        )

            # ── Top-up partial quantity: PUT leg ──────────────────────────────
            # If the open PUT short holds less than the target quantity (QUANTITY * lot_size),
            # Supertrend is still BULLISH, entry window is open, and no exit is pending:
            # immediately place a new order for the remaining quantity instead of trading partial.
            if (
                curr_direction == 1
                and put_pos is not None
                and not put_pos.get("exit_pending")
                and entry_allowed
            ):
                target_put_qty = QUANTITY * put_pos.get("lot_size", 30)
                current_put_qty = put_pos.get("quantity", 0)
                if 0 < current_put_qty < target_put_qty:
                    missing_put_qty = target_put_qty - current_put_qty
                    log.warning(
                        f"[TOP-UP] PUT SELL: currently holding partial quantity ({current_put_qty}/{target_put_qty}). "
                        f"Placing top-up order for remaining quantity {missing_put_qty} x {put_pos['symbol']}..."
                    )
                    top_id, top_fill, top_filled_qty = sell_option(
                        groww, put_pos["symbol"], missing_put_qty, "PUT TOP-UP"
                    )
                    if top_filled_qty > 0:
                        old_qty   = current_put_qty
                        old_price = put_pos.get("entry_price") or 0.0
                        new_total = old_qty + top_filled_qty
                        if top_fill and old_price:
                            new_avg_entry = round((old_qty * old_price + top_filled_qty * top_fill) / new_total, 2)
                        else:
                            new_avg_entry = top_fill or old_price
                        put_pos["quantity"]    = new_total
                        put_pos["entry_price"] = new_avg_entry
                        log.info(
                            f"[TOP-UP SUCCESS] PUT SELL: top-up filled {top_filled_qty}/{missing_put_qty} @ ₹{top_fill:.2f}. "
                            f"New total quantity={new_total}/{target_put_qty}, new weighted entry=₹{new_avg_entry:.2f}"
                        )
                        log_trade(
                            leg="PUT SELL", action="TOP-UP", symbol=put_pos["symbol"],
                            quantity=top_filled_qty, price=top_fill, fill_confirmed=top_fill is not None,
                            order_id=top_id, reason=f"Top-up remaining quantity ({new_total}/{target_put_qty})",
                            when=datetime.now(IST),
                        )
                    else:
                        log.warning(
                            f"[TOP-UP FAILED] PUT SELL: could not fill remaining {missing_put_qty} for {put_pos['symbol']}. "
                            "Will retry next cycle if trend persists."
                        )

            # ── Profit-based strike switch: PUT leg ───────────────────────────
            # If the open PUT short has >= PROFIT_SWITCH_THRESHOLD pts of profit
            # (entry_price - LTP >= threshold), exit and re-sell a new ATM PUT.
            # This runs every cycle regardless of Supertrend direction, before
            # the ST-flip exit check below.
            if put_pos is not None and not put_pos.get("exit_pending") and put_pos.get("entry_price") is not None:
                _put_ltp_switch = get_option_ltp(groww, put_pos["symbol"])
                _put_profit_pts = put_pos["entry_price"] - _put_ltp_switch  # positive = profit for seller
                if _put_profit_pts >= PROFIT_SWITCH_THRESHOLD:
                    log.info(
                        f"[PROFIT SWITCH] PUT SELL: profit={_put_profit_pts:.2f} pts >= "
                        f"{PROFIT_SWITCH_THRESHOLD} pts threshold  "
                        f"(entry=₹{put_pos['entry_price']:.2f}  LTP=₹{_put_ltp_switch:.2f})  "
                        "→ Switching to new nearest ATM PUT strike."
                    )
                    instruments_df = get_cached_instruments_df(groww)
                    expiry         = get_monthly_expiry_date(instruments_df)
                    put_pos = switch_strike_on_profit(
                        groww, put_pos, "PE", "PUT SELL",
                        instruments_df, expiry, fut_symbol
                    )

            # Exit: ST turned BEARISH AND P/L threshold reached
            # NOTE: deliberately `if`, not `elif` off the profit-switch block above —
            # that block's outer condition only tests `put_pos is not None` (independent
            # of curr_direction), so as `elif` it silently swallowed this exit check
            # every cycle a position was open but under the 180-pt switch threshold,
            # even when the ST-flip + 111-pt exit condition was clearly met.
            if curr_direction == -1 and put_pos is not None:
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
                        trade_qty = put_pos.get("quantity") or (QUANTITY * put_pos["lot_size"])
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
                    order_id, fill_price, filled_qty = sell_option(groww, call_sym, trade_qty, "CALL")
                    entry_time = datetime.now(IST)
                    if not order_id:
                        # See matching PUT SELL comment above: empty order_id
                        # means every offset was confirmed REJECTED/FAILED/
                        # CANCELLED — definitively no position, not merely
                        # "unconfirmed". Leave call_pos as None.
                        log.error(
                            f"CALL SELL ENTRY FAILED  symbol={call_sym}  qty={trade_qty}  "
                            f"time={entry_time:%Y-%m-%d %H:%M:%S}  reason=ST flip BEARISH  "
                            "— every offset attempt was rejected/failed. NO POSITION was "
                            "opened. Not tracking a position for this signal."
                        )
                        log_trade(
                            leg="CALL SELL", action="ENTRY", symbol=call_sym, quantity=trade_qty,
                            price=call_ltp_check, fill_confirmed=False, order_id="",
                            reason="ST flip BEARISH — ENTRY FAILED, no position opened",
                            when=entry_time,
                        )
                    else:
                        actual_qty = filled_qty if filled_qty > 0 else trade_qty
                        is_partial = 0 < filled_qty < trade_qty
                        call_pos = {
                            "symbol":      call_sym,
                            "lot_size":    call_lot,
                            "quantity":    actual_qty,
                            "entry_price": fill_price,   # None if fill unconfirmed
                            "entry_time":  entry_time,
                            "order_id":    order_id,
                        }
                        partial_tag = " (PARTIAL)" if is_partial else ""
                        if fill_price is not None:
                            log.info(
                                f"[TRADE] CALL SELL ENTRY{partial_tag}  symbol={call_sym}  strike_ltp_check=₹{call_ltp_check:.2f}  "
                                f"qty={actual_qty}  fill=₹{fill_price:.2f}  time={entry_time:%Y-%m-%d %H:%M:%S}  "
                                f"order_id={order_id}  reason=ST flip BEARISH"
                            )
                        else:
                            log.warning(
                                f"[TRADE] CALL SELL ENTRY (UNCONFIRMED)  symbol={call_sym}  qty={actual_qty}  "
                                f"time={entry_time:%Y-%m-%d %H:%M:%S}  order_id={order_id}  reason=ST flip BEARISH  "
                                "— fill price unknown. VERIFY ACTUAL POSITION ON GROWW."
                            )
                        log_trade(
                            leg="CALL SELL", action="ENTRY", symbol=call_sym, quantity=actual_qty,
                            price=fill_price if fill_price is not None else call_ltp_check,
                            fill_confirmed=fill_price is not None, order_id=order_id,
                            reason=f"ST flip BEARISH{partial_tag}", when=entry_time,
                        )

            # ── Top-up partial quantity: CALL leg ─────────────────────────────
            # If the open CALL short holds less than the target quantity (QUANTITY * lot_size),
            # Supertrend is still BEARISH, entry window is open, and no exit is pending:
            # immediately place a new order for the remaining quantity instead of trading partial.
            if (
                curr_direction == -1
                and call_pos is not None
                and not call_pos.get("exit_pending")
                and entry_allowed
            ):
                target_call_qty = QUANTITY * call_pos.get("lot_size", 30)
                current_call_qty = call_pos.get("quantity", 0)
                if 0 < current_call_qty < target_call_qty:
                    missing_call_qty = target_call_qty - current_call_qty
                    log.warning(
                        f"[TOP-UP] CALL SELL: currently holding partial quantity ({current_call_qty}/{target_call_qty}). "
                        f"Placing top-up order for remaining quantity {missing_call_qty} x {call_pos['symbol']}..."
                    )
                    top_id, top_fill, top_filled_qty = sell_option(
                        groww, call_pos["symbol"], missing_call_qty, "CALL TOP-UP"
                    )
                    if top_filled_qty > 0:
                        old_qty   = current_call_qty
                        old_price = call_pos.get("entry_price") or 0.0
                        new_total = old_qty + top_filled_qty
                        if top_fill and old_price:
                            new_avg_entry = round((old_qty * old_price + top_filled_qty * top_fill) / new_total, 2)
                        else:
                            new_avg_entry = top_fill or old_price
                        call_pos["quantity"]    = new_total
                        call_pos["entry_price"] = new_avg_entry
                        log.info(
                            f"[TOP-UP SUCCESS] CALL SELL: top-up filled {top_filled_qty}/{missing_call_qty} @ ₹{top_fill:.2f}. "
                            f"New total quantity={new_total}/{target_call_qty}, new weighted entry=₹{new_avg_entry:.2f}"
                        )
                        log_trade(
                            leg="CALL SELL", action="TOP-UP", symbol=call_pos["symbol"],
                            quantity=top_filled_qty, price=top_fill, fill_confirmed=top_fill is not None,
                            order_id=top_id, reason=f"Top-up remaining quantity ({new_total}/{target_call_qty})",
                            when=datetime.now(IST),
                        )
                    else:
                        log.warning(
                            f"[TOP-UP FAILED] CALL SELL: could not fill remaining {missing_call_qty} for {call_pos['symbol']}. "
                            "Will retry next cycle if trend persists."
                        )

            # ── Profit-based strike switch: CALL leg ──────────────────────────
            # If the open CALL short has >= PROFIT_SWITCH_THRESHOLD pts of profit
            # (entry_price - LTP >= threshold), exit and re-sell a new ATM CALL.
            # This runs every cycle regardless of Supertrend direction, before
            # the ST-flip exit check below.
            if call_pos is not None and not call_pos.get("exit_pending") and call_pos.get("entry_price") is not None:
                _call_ltp_switch = get_option_ltp(groww, call_pos["symbol"])
                _call_profit_pts = call_pos["entry_price"] - _call_ltp_switch  # positive = profit for seller
                if _call_profit_pts >= PROFIT_SWITCH_THRESHOLD:
                    log.info(
                        f"[PROFIT SWITCH] CALL SELL: profit={_call_profit_pts:.2f} pts >= "
                        f"{PROFIT_SWITCH_THRESHOLD} pts threshold  "
                        f"(entry=₹{call_pos['entry_price']:.2f}  LTP=₹{_call_ltp_switch:.2f})  "
                        "→ Switching to new nearest ATM CALL strike."
                    )
                    instruments_df = get_cached_instruments_df(groww)
                    expiry         = get_monthly_expiry_date(instruments_df)
                    call_pos = switch_strike_on_profit(
                        groww, call_pos, "CE", "CALL SELL",
                        instruments_df, expiry, fut_symbol
                    )

            # Exit: ST turned BULLISH AND P/L threshold reached
            # NOTE: deliberately `if`, not `elif` — see matching PUT SELL comment above.
            if curr_direction == 1 and call_pos is not None:
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
                        trade_qty = call_pos.get("quantity") or (QUANTITY * call_pos["lot_size"])
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
                    trade_qty = put_pos.get("quantity") or (QUANTITY * put_pos["lot_size"])
                    put_pos = attempt_square_off(
                        groww, put_pos, trade_qty,
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
                    trade_qty = call_pos.get("quantity") or (QUANTITY * call_pos["lot_size"])
                    call_pos = attempt_square_off(
                        groww, call_pos, trade_qty,
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
            elif _is_auth_error(exc) or "forbidden" in str(exc).lower():
                # API token has expired or been invalidated — force a fresh
                # re-authentication immediately.  Sleeping-and-retrying (the
                # old generic else branch) never recovers from this because
                # every subsequent API call will also fail with the same
                # 401/auth error until a new token is obtained.
                now_ist = datetime.now(IST)
                age     = now_ist - last_auth_time
                log.warning(
                    f"Groww API authentication error (token age={age}): {exc}. "
                    "Forcing token refresh and re-authenticating…"
                )
                # Delete the cached token so authenticate() does a fresh TOTP login
                # rather than loading the same expired token from disk.
                try:
                    if os.path.exists(TOKEN_CACHE_FILE):
                        os.remove(TOKEN_CACHE_FILE)
                        log.info(f"Deleted stale cached token: {TOKEN_CACHE_FILE}")
                except OSError as cache_err:
                    log.warning(f"Could not delete token cache file: {cache_err}")
                try:
                    groww          = authenticate(force_refresh=True)
                    last_auth_time = datetime.now(IST)
                    # Reset candle buffer so the next cycle does a full re-seed
                    # rather than an incremental 15-min fetch that could miss
                    # candles accumulated during the auth outage.
                    reset_candle_buffer()
                    log.info(
                        "Re-authenticated successfully after token expiry. "
                        "Candle buffer reset — full re-seed on next cycle. "
                        "Resuming strategy on the next cycle."
                    )
                except Exception as auth_exc:
                    log.error(
                        f"Re-authentication failed after token expiry: {auth_exc}. "
                        "Sleeping 60 s before retrying…"
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
