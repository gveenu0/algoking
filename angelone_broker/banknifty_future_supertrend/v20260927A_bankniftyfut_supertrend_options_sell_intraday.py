"""
================================================================================
BankNifty Futures Supertrend Intraday Options Selling Strategy (AngelOne)
================================================================================

OVERVIEW:
    This algorithm executes an automated, rule-based intraday options selling
    strategy on BankNifty (NSE FNO) powered by the AngelOne SmartAPI.
    It tracks trend momentum using a 1-minute Supertrend indicator calculated
    on the nearest BankNifty Futures contract and systematically takes short
    option positions (selling Put options during bullish phases and Call options
    during bearish phases) to harvest premium decay and directional momentum.

    This is a port of the Groww broker version
    (groww_broker/banknifty_future_supertrend/v20260916B_bankniftyfut_supertrend_options_sell_intraday.py)
    onto the AngelOne SmartAPI (smartapi-python). The trading logic, risk
    management, and resilience architecture are functionally identical —
    only the broker SDK plumbing (auth, instruments, LTP, candles, orders)
    has been swapped out.

--------------------------------------------------------------------------------
KEY STRATEGY PARAMETERS:
--------------------------------------------------------------------------------
    • Underlying Index        : BANKNIFTY (via NSE FNO active Futures contract)
    • Strike Step             : 100 points
    • Expiry Selection        : Current monthly expiry (last available in month)
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
      Generates a client-side `ordertag` per order (AngelOne has no direct
      "get order by client reference" endpoint like Groww, so on a network/
      timeout error we search the day's order book for a matching ordertag
      before ever retrying, instead of blindly resubmitting).
    • Rate Limiting & Auth Token Caching:
      Enforces client-side throttling (70% safety margin) against AngelOne's
      published per-second/per-minute API limits (orders: 9/s, LTP: 10/s,
      historical candles: 3/s, order lookups: 10/s). Caches the day's
      jwtToken/refreshToken/feedToken in `angelone_token_cache.json` and
      prefers a cheap refresh-token renewal over a full TOTP login.
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
from SmartApi import SmartConnect

try:
    from SmartApi.smartExceptions import (
        TokenException as AngelTokenException,
        PermissionException as AngelPermissionException,
        NetworkException as AngelNetworkException,
        DataException as AngelDataException,
        GeneralException as AngelGeneralException,
    )
except ImportError:  # pragma: no cover - defensive against SDK path changes
    class AngelTokenException(Exception):
        """Fallback stand-in if the SDK's exception module path changes.
        Real auth errors are still caught via message-text matching in
        _is_auth_error() below."""
        pass

    class AngelPermissionException(Exception):
        pass

    class AngelNetworkException(Exception):
        pass

    class AngelDataException(Exception):
        pass

    class AngelGeneralException(Exception):
        pass


# ─── NETWORK TIMEOUT SAFETY NET ───────────────────────────────────────────────
# smartapi-python builds its own `requests` calls internally with a default
# 7s timeout for most routes, but individual_order_details() makes a raw
# `requests.get()` call with NO timeout kwarg at all. Without a bound, a
# dropped/slow connection to apiconnect.angelone.in can hang for the OS-level
# TCP timeout before failing, which is far too long for a strategy that needs
# to check exits every minute.
#
# socket.setdefaulttimeout() is a process-wide safety net: any socket that
# doesn't already have an explicit timeout set will now raise `socket.timeout`
# after this many seconds instead of hanging indefinitely. This must be set
# once, before any network calls are made.
NETWORK_TIMEOUT_SECONDS = 60
socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)


# ─── USER CONFIGURATION ──────────────────────────────────────────────────────
API_KEY       = "ANGELONE_API_KEY"        # Replace with your AngelOne SmartAPI key
CLIENT_CODE   = "ANGELONE_CLIENT_CODE"    # Replace with your AngelOne client code
CLIENT_PIN    = "ANGELONE_PIN"            # Replace with your AngelOne trading PIN/password
TOTP_SECRET   = "ANGELONE_TOTP_SECRET"    # Replace with your AngelOne TOTP secret (QR seed)

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

UNDERLYING  = "BANKNIFTY"
STRIKE_STEP = 100  # BankNifty option strike interval
EXCHANGE_NFO = "NFO"

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
# AngelOne's historical API allows up to 30 days per request for ONE_MINUTE
# candles, so this comfortably fits in a single call.
SEED_LOOKBACK_DAYS = 5

IST = timezone(timedelta(hours=5, minutes=30))

# Minimum acceptable option premium for entry.
# If ATM price < this, walk ITM (one strike at a time) until premium >= threshold.
MIN_OPTION_PRICE = 400


# ─── LOGGING ─────────────────────────────────────────────────────────────────
# Everything still prints to stdout, but is ALSO written to a dated file
# under LOG_DIR so you have a persistent, downloadable record even after the
# console scrolls/refreshes/restarts.
def _resolve_log_dir() -> str:
    """
    Pick a writable directory for the log file / trade journal.

    Some deployments run the code from a read-only directory, so a plain
    relative "logs" folder can fail with PermissionError even though the box
    has plenty of writable space elsewhere. Try, in order:
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
    LOG_DIR, f"strategy_angelone_{datetime.now(IST).strftime('%Y-%m-%d')}.log"
)
TOKEN_CACHE_FILE = os.path.join(LOG_DIR, "angelone_token_cache.json")
SCRIP_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

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
TRADE_JOURNAL_PATH = os.path.join(LOG_DIR, "trade_journal_angelone.csv")
TRADE_JOURNAL_FIELDS = [
    "timestamp",        # IST execution time this row was logged
    "leg",              # "PUT SELL" or "CALL SELL"
    "action",           # "ENTRY" or "EXIT"
    "symbol",           # option trading symbol
    "quantity",         # total quantity (lots x lot_size)
    "price",            # execution/fill price (or LTP fallback if unconfirmed)
    "fill_confirmed",   # True/False — False means price is an LTP estimate
    "order_id",         # AngelOne orderid
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
# Per AngelOne's published limits (smartapi.angelbroking.com/docs/RateLimit):
#   orders       (place/modify/cancel, combined)        : 9/sec,  500/min
#   live_data    (getLtpData)                           : 10/sec, 500/min
#   historical   (getCandleData)                        : 3/sec,  150/min
#   non_trading  (individual order details)             : 10/sec, 500/min
#   order_book   (getOrderBook — no documented per-min ceiling, kept tight)   : 1/sec
# We pace ourselves comfortably under these ceilings so we essentially never
# trigger a 403 rate-limit response in the first place, and we still handle
# it gracefully (with backoff) if it ever fires anyway.
RATE_LIMITS = {
    "orders":      {"per_sec": 9,  "per_min": 500},
    "live_data":   {"per_sec": 10, "per_min": 500},
    "historical":  {"per_sec": 3,  "per_min": 150},
    "non_trading": {"per_sec": 10, "per_min": 500},
    "order_book":  {"per_sec": 1,  "per_min": 50},
}
RATE_LIMIT_SAFETY_MARGIN = 0.7  # only ever use ~70% of the documented ceiling

_rate_limit_history: dict[str, deque] = {k: deque() for k in RATE_LIMITS}


def _throttle(category: str) -> None:
    """Block just long enough to stay under the safety-margined per-second
    and per-minute limits for `category`, based on this process's own
    rolling call history. Purely client-side pacing — doesn't talk to AngelOne."""
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
    if isinstance(exc, AngelPermissionException):
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "access denied" in msg or "too many requests" in msg or "403" in msg


def _is_auth_error(exc: Exception) -> bool:
    """
    True for AngelOne SmartAPI authentication / token errors — i.e. the
    jwtToken has expired or been revoked and the request was rejected.
    These are NOT transient network blips; they require a forced re-auth.
    """
    if isinstance(exc, AngelTokenException):
        return True
    msg = str(exc).lower()
    return (
        "invalid token" in msg
        or "token expired" in msg
        or "session expired" in msg
        or "unauthorized" in msg
        or "login" in msg and "again" in msg
        or "http 401" in msg
        or "status 401" in msg
    )


# Fixed backoff schedule (seconds) for transient network/timeout errors.
# Deliberately short and fixed (not exponential) — these are brief connection
# blips to apiconnect.angelone.in, not sustained outages, so we want to
# recover fast rather than back off aggressively. If all retries are
# exhausted, the error still propagates up to run_strategy()'s top-level
# handler as a final safety net (60s sleep + resume next cycle).
NETWORK_RETRY_DELAYS = [2, 5, 10]


def _is_transient_network_error(exc: Exception) -> bool:
    """
    True for connection/timeout failures to apiconnect.angelone.in that are
    almost always transient (a brief network blip) rather than a fault in
    the request itself.
    """
    if isinstance(exc, AngelNetworkException):
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


_HARD_TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="angelone-call")


def _call_with_hard_timeout(fn, *args, timeout_seconds: float = 20.0, **kwargs):
    """
    Run fn in a worker thread and enforce a real wall-clock timeout on it.

    Some smartapi-python calls (e.g. individual_order_details) issue a raw
    requests.get() with no explicit timeout, so a stalled TCP connection can
    hang indefinitely with zero exception raised — in that case _call()'s
    retry/backoff logic never triggers (it only reacts to raised
    exceptions). Running the call in a separate thread and bounding it with
    .result(timeout=...) guarantees we regain control even if the call
    itself never returns or raises.
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
    Throttled, fault-tolerant wrapper around every AngelOne SmartAPI call.

    1. Paces the call via _throttle() so we stay under AngelOne's documented
       ceiling for `category` before we even attempt the request.
    2. Retries two distinct classes of *transient* failure, each on its own
       backoff schedule, before propagating to the caller:
         - Rate limit errors      : exponential backoff (1s, 2s, 4s), 3 retries
         - Network/timeout errors : fixed backoff (2s, 5s, 10s), 3 retries
       Any other exception is raised immediately — it's not a class of
       error we know is safe to blindly retry.
    3. If retries are exhausted, the original exception is raised to the
       caller. run_strategy()'s top-level handler is still there as a final
       safety net (logs, sleeps 60s, resumes on the next cycle).
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
                    f"AngelOne rate limit hit ({category}, attempt "
                    f"{rate_limit_attempt + 1}/4) — backing off {wait}s: {e}"
                )
                time.sleep(wait)
                rate_limit_attempt += 1
                _throttle(category)
                continue

            if _is_transient_network_error(e):
                if network_attempt >= len(NETWORK_RETRY_DELAYS):
                    log.error(
                        f"AngelOne API network/timeout error persisted after "
                        f"{len(NETWORK_RETRY_DELAYS)} retries ({category}): {e}"
                    )
                    raise
                wait = NETWORK_RETRY_DELAYS[network_attempt]
                log.warning(
                    f"Network/timeout error calling AngelOne API ({category}, "
                    f"attempt {network_attempt + 1}/{len(NETWORK_RETRY_DELAYS) + 1}) "
                    f"— retrying in {wait}s: {e}"
                )
                time.sleep(wait)
                network_attempt += 1
                _throttle(category)
                continue

            raise


# ─── AUTHENTICATION ──────────────────────────────────────────────────────────
def _new_smart_connect() -> SmartConnect:
    return SmartConnect(api_key=API_KEY)


def authenticate(force_refresh: bool = False) -> SmartConnect:
    """Authenticate via cached session, refresh-token renewal, or full TOTP login.

    Checks TOKEN_CACHE_FILE for an existing session generated today.
      1. If present and not too old, verify it with a lightweight getProfile()
         call before trusting it.
      2. If verification fails (or the cache is stale), try to mint a fresh
         jwtToken from the cached refreshToken via generateToken() — this is
         a cheap call that does NOT require a fresh TOTP code.
      3. Only if both of the above fail do we fall back to a full TOTP login
         via generateSession(), which is the most rate-limited/sensitive
         call and should be avoided when possible.
    """
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    MAX_TOKEN_AGE_HOURS = 6  # conservative: AngelOne jwtTokens are short-lived

    cache = None
    if not force_refresh and os.path.exists(TOKEN_CACHE_FILE):
        try:
            with open(TOKEN_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            log.warning(f"Could not read cached AngelOne session ({e}). Ignoring cache.")
            cache = None
    elif force_refresh:
        log.info("Force-refresh requested: invalidating cached session.")
        try:
            if os.path.exists(TOKEN_CACHE_FILE):
                os.remove(TOKEN_CACHE_FILE)
        except OSError:
            pass

    def _persist_cache(jwt_token: str, refresh_token: str, feed_token: str) -> None:
        try:
            with open(TOKEN_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "date":          today_str,
                    "jwt_token":     jwt_token,
                    "refresh_token": refresh_token,
                    "feed_token":    feed_token,
                    "generated_at":  datetime.now(IST).isoformat(),
                }, f)
            log.info(f"AngelOne session cached to {TOKEN_CACHE_FILE}.")
        except Exception as save_err:
            log.warning(f"Could not cache AngelOne session to disk: {save_err}")

    # 1. Try the cached session first (same-day and not too old)
    if cache and cache.get("date") == today_str and cache.get("jwt_token"):
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
                        f"Cached AngelOne token is {token_age} old (limit: "
                        f"{MAX_TOKEN_AGE_HOURS}h) — treating as expired."
                    )
            except (ValueError, TypeError) as ts_err:
                log.warning(f"Could not parse cached generated_at ({ts_err}); skipping age check.")

        if token_age_ok:
            try:
                log.info("Found cached AngelOne session for today. Verifying…")
                smart_api = _new_smart_connect()
                smart_api.setAccessToken(cache["jwt_token"])
                smart_api.setRefreshToken(cache["refresh_token"])
                smart_api.setFeedToken(cache.get("feed_token"))
                smart_api.getProfile(cache["refresh_token"])
                log.info("Cached AngelOne session verified successfully. Skipping login.")
                return smart_api
            except Exception as e:
                log.warning(f"Cached session verification failed ({e}). Trying refresh-token renewal…")

        # 2. Cheap renewal via refreshToken (no TOTP required)
        if cache.get("refresh_token"):
            try:
                smart_api = _new_smart_connect()
                token_set = smart_api.generateToken(cache["refresh_token"])
                new_jwt = token_set["data"]["jwtToken"]
                new_feed = token_set["data"].get("feedToken", cache.get("feed_token"))
                smart_api.setAccessToken(new_jwt)
                smart_api.setRefreshToken(cache["refresh_token"])
                smart_api.setFeedToken(new_feed)
                log.info("Renewed AngelOne jwtToken via refreshToken — no TOTP login needed.")
                _persist_cache(new_jwt, cache["refresh_token"], new_feed)
                return smart_api
            except Exception as e:
                log.warning(f"Refresh-token renewal failed ({e}). Falling back to full TOTP login…")

    # 3. Full TOTP login
    max_retries = 5
    secret = TOTP_SECRET.replace(" ", "").replace("-", "").upper()

    for attempt in range(1, max_retries + 1):
        try:
            totp = pyotp.TOTP(secret).now()
            smart_api = _new_smart_connect()
            session = smart_api.generateSession(CLIENT_CODE, CLIENT_PIN, totp)
            if not session.get("status", False):
                raise RuntimeError(f"generateSession returned status=False: {session}")

            jwt_token     = session["data"]["jwtToken"]
            refresh_token = session["data"]["refreshToken"]
            feed_token    = session["data"]["feedToken"]
            smart_api.setAccessToken(jwt_token)
            smart_api.setRefreshToken(refresh_token)
            smart_api.setFeedToken(feed_token)
            log.info(f"Authenticated with AngelOne SmartAPI (TOTP, attempt {attempt}).")

            _persist_cache(jwt_token, refresh_token, feed_token)
            return smart_api
        except Exception as e:
            is_rate_limit = _is_rate_limit_error(e)
            if is_rate_limit:
                retry_delay = 60 * attempt  # gradual backoff: 60s, 120s, 180s...
                log.error(
                    f"Authentication attempt {attempt}/{max_retries} hit AngelOne "
                    f"rate limit: {e}."
                )
            else:
                retry_delay = 30
                log.error(f"Authentication attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                log.info(f"Retrying authentication in {retry_delay}s…")
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Authentication failed after {max_retries} attempts — "
        "check TOTP credentials, NTP clock synchronization, and rate-limit status on AngelOne."
    )


# ─── INSTRUMENT HELPERS ───────────────────────────────────────────────────────
def get_all_instruments_df(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download (or reuse a same-day local copy of) AngelOne's full instrument
    master and return it as a normalised DataFrame.

    The master is a public, unauthenticated JSON dump regenerated once a day
    (per AngelOne's docs), so we cache it to disk once per calendar day
    instead of re-downloading the ~30MB file on every restart.
    """
    today_str  = datetime.now(IST).strftime("%Y-%m-%d")
    cache_path = os.path.join(LOG_DIR, f"angelone_scripmaster_{today_str}.json")

    if force_refresh or not os.path.exists(cache_path):
        log.info(f"Downloading AngelOne instrument master from {SCRIP_MASTER_URL}…")
        resp = _call(
            "non_trading",
            lambda: requests.get(SCRIP_MASTER_URL, timeout=NETWORK_TIMEOUT_SECONDS),
            hard_timeout_seconds=90.0,
        )
        resp.raise_for_status()
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        log.info(f"Instrument master downloaded and cached to {cache_path}.")
    else:
        log.info(f"Reusing today's cached instrument master: {cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df["expiry_date"] = pd.to_datetime(df["expiry"], format="%d%b%Y", errors="coerce")
    df["strike"]      = pd.to_numeric(df["strike"], errors="coerce") / 100.0
    df["lotsize"]     = pd.to_numeric(df["lotsize"], errors="coerce")
    df["exch_seg"]    = df["exch_seg"].astype(str).str.upper()
    df["instrumenttype"] = df["instrumenttype"].astype(str).str.upper()
    return df


_instruments_cache: dict = {"df": None, "ts": 0.0}
INSTRUMENTS_CACHE_TTL_SECONDS = 1800  # 30 min — instrument list doesn't change intraday


def get_cached_instruments_df(force_refresh: bool = False) -> pd.DataFrame:
    """
    Reuse the in-memory instruments dataframe for INSTRUMENTS_CACHE_TTL_SECONDS
    instead of re-parsing the ~30MB scrip master JSON on every PUT/CALL entry
    signal.
    """
    now = time.monotonic()
    stale = (
        force_refresh
        or _instruments_cache["df"] is None
        or (now - _instruments_cache["ts"]) > INSTRUMENTS_CACHE_TTL_SECONDS
    )
    if stale:
        _instruments_cache["df"] = get_all_instruments_df(force_refresh=force_refresh)
        _instruments_cache["ts"] = now
    return _instruments_cache["df"]


def get_monthly_expiry_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    Find the BankNifty monthly expiry for the current/next month.
    We pick the latest expiry within the current calendar month whose
    expiry date has not yet passed today.  If none remain this month,
    we use the next month's monthly expiry.
    On the expiry day itself, today's expiring contract is excluded so the
    strategy rolls straight to the next available expiry instead of trading
    the contract that is expiring today.
    The actual date is always derived from the live instrument master so
    the correct expiry is resolved automatically regardless of day changes.
    """
    today = pd.Timestamp(datetime.now(IST).date())

    mask = (
        (df["exch_seg"] == EXCHANGE_NFO)
        & (df["name"] == UNDERLYING)
        & (df["instrumenttype"] == "OPTIDX")
        & (df["expiry_date"] > today)
    )
    expiry_dates = df[mask]["expiry_date"].dropna().unique()
    expiry_dates = sorted(expiry_dates)

    if not expiry_dates:
        raise RuntimeError("No future BankNifty OPTIDX expiry dates found in instrument master.")

    current_month = today.month
    current_year  = today.year

    # Pick expiry dates that fall in the current month
    this_month = [e for e in expiry_dates if pd.Timestamp(e).month == current_month and pd.Timestamp(e).year == current_year]

    if this_month:
        # Latest expiry in current month = monthly expiry
        chosen = max(this_month)
    else:
        # None left this month — pick the latest in the next calendar month
        next_month = current_month % 12 + 1
        next_year  = current_year + (1 if current_month == 12 else 0)
        next_month_dates = [
            e for e in expiry_dates if pd.Timestamp(e).month == next_month and pd.Timestamp(e).year == next_year
        ]
        if next_month_dates:
            chosen = max(next_month_dates)
        else:
            # Fallback: nearest available expiry
            chosen = expiry_dates[0]
            log.warning(
                f"Could not determine monthly expiry; falling back to nearest: {pd.Timestamp(chosen).date()}"
            )

    chosen = pd.Timestamp(chosen)
    log.info(f"Selected BankNifty monthly expiry: {chosen.date()}")
    return chosen


def get_active_banknifty_fut_symbol(df: pd.DataFrame) -> tuple[str, str]:
    """
    Return (trading_symbol, symbol_token) of the nearest-expiry BANKNIFTY
    futures contract on NFO. Used for historical candle fetching and LTP —
    futures prices track the index closely.
    """
    today = pd.Timestamp(datetime.now(IST).date())
    mask = (
        (df["exch_seg"] == EXCHANGE_NFO)
        & (df["name"] == UNDERLYING)
        & (df["instrumenttype"] == "FUTIDX")
        & (df["expiry_date"] >= today)
    )
    active = df[mask].copy().sort_values("expiry_date")
    if active.empty:
        raise RuntimeError(
            "No active BANKNIFTY FUT contract found on NFO. "
            "Check the instrument master or market calendar."
        )
    row     = active.iloc[0]
    sym     = str(row["symbol"])
    token   = str(row["token"])
    log.info(f"Using BANKNIFTY futures symbol for candle/LTP data: {sym} (token={token})")
    return sym, token


def get_atm_strike(banknifty_ltp: float) -> int:
    """Round LTP to the nearest STRIKE_STEP to get ATM strike."""
    return int(round(banknifty_ltp / STRIKE_STEP) * STRIKE_STEP)


def find_option_symbol(
    df: pd.DataFrame,
    expiry: pd.Timestamp,
    strike: int,
    option_type: str,   # "CE" or "PE"
) -> tuple[str, str, int]:
    """
    Return (trading_symbol, symbol_token, lot_size) for the BankNifty option
    matching the given expiry, strike and option type.
    """
    option_type = option_type.upper()
    mask = (
        (df["exch_seg"] == EXCHANGE_NFO)
        & (df["name"] == UNDERLYING)
        & (df["instrumenttype"] == "OPTIDX")
        & (df["expiry_date"].dt.normalize() == pd.Timestamp(expiry.date()))
        & (df["strike"].round().fillna(-1).astype(int) == strike)
        & (df["symbol"].str.upper().str.endswith(option_type))
    )
    rows = df[mask]
    if rows.empty:
        raise RuntimeError(
            f"No instrument found for BANKNIFTY {option_type} strike={strike} "
            f"expiry={expiry.date()}. ATM strike may not be listed — "
            "check the instrument master."
        )
    row      = rows.iloc[0]
    raw_lot  = row.get("lotsize")
    if pd.notna(raw_lot) and raw_lot:
        lot_size = int(raw_lot)
    else:
        lot_size = 30
        log.warning(
            f"lotsize missing/invalid for {option_type} strike={strike} "
            f"expiry={expiry.date()} — falling back to default lot_size=30. "
            "Verify this matches the current exchange lot size before trading."
        )
    sym   = str(row["symbol"])
    token = str(row["token"])
    log.info(
        f"Resolved {option_type} instrument: {sym}  token={token}  "
        f"strike={strike}  expiry={expiry.date()}  lot_size={lot_size}"
    )
    return sym, token, lot_size


def find_option_with_min_price(
    smart_api: SmartConnect,
    df: pd.DataFrame,
    expiry: pd.Timestamp,
    atm_strike: int,
    option_type: str,   # "CE" or "PE"
    min_price: float = MIN_OPTION_PRICE,
    max_itm_steps: int = 10,
) -> tuple[str, str, int, float]:
    """
    Return (trading_symbol, symbol_token, lot_size, ltp) for the cheapest
    strike whose LTP meets the minimum premium threshold.

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
    best_token = None
    best_lot  = None
    best_ltp  = None

    for step in range(max_itm_steps + 1):
        label = "ATM" if step == 0 else f"ITM+{step}"
        try:
            sym, token, lot = find_option_symbol(df, expiry, strike, option_type)
        except RuntimeError as e:
            log.warning(f"  {label} strike={strike} not found in instruments: {e}. Stopping ITM walk.")
            break

        ltp = get_option_ltp(smart_api, sym, token)
        log.info(
            f"  {label} {option_type} strike={strike}  LTP=₹{ltp:.2f}  "
            f"(threshold=₹{min_price:.2f})"
        )

        if best_sym is None:
            # Always keep at least the ATM as fallback
            best_sym, best_token, best_lot, best_ltp = sym, token, lot, ltp

        if ltp >= min_price:
            if step > 0:
                log.info(
                    f"  ATM LTP was below ₹{min_price:.2f} — "
                    f"selected {label} {option_type} strike={strike} @ ₹{ltp:.2f}"
                )
            return sym, token, lot, ltp

        # Not enough premium yet — go deeper ITM
        best_sym, best_token, best_lot, best_ltp = sym, token, lot, ltp
        strike += direction

    # No strike met the threshold
    log.warning(
        f"No {option_type} strike found with LTP >= ₹{min_price:.2f} after "
        f"{max_itm_steps} ITM steps. Using deepest tried: "
        f"strike={strike - direction}  LTP=₹{best_ltp:.2f}. Proceeding anyway."
    )
    return best_sym, best_token, best_lot, best_ltp


# ─── LTP HELPERS ──────────────────────────────────────────────────────────────
def get_banknifty_ltp(smart_api: SmartConnect, fut_symbol: str, fut_token: str) -> float:
    """Fetch the last traded price of the nearest BANKNIFTY futures contract (NFO)."""
    resp = _call(
        "live_data", smart_api.ltpData,
        exchange=EXCHANGE_NFO, tradingsymbol=fut_symbol, symboltoken=fut_token,
    )
    ltp = (resp or {}).get("data", {}).get("ltp")
    if ltp is None:
        raise RuntimeError(f"BankNifty LTP not found for {fut_symbol}. Response: {resp}")
    return float(ltp)


def get_option_ltp(smart_api: SmartConnect, trading_symbol: str, symbol_token: str) -> float:
    """Fetch the last traded price of an FNO option contract."""
    resp = _call(
        "live_data", smart_api.ltpData,
        exchange=EXCHANGE_NFO, tradingsymbol=trading_symbol, symboltoken=symbol_token,
    )
    ltp = (resp or {}).get("data", {}).get("ltp")
    if ltp is None:
        raise RuntimeError(f"Option LTP not found for {trading_symbol}. Response: {resp}")
    return float(ltp)


# ─── CANDLE BUFFER FOR BANKNIFTY FUTURES ──────────────────────────────────────
_candle_buffer: pd.DataFrame | None = None


def reset_candle_buffer() -> None:
    global _candle_buffer
    _candle_buffer = None


def fetch_1min_candles(smart_api: SmartConnect, fut_symbol: str, fut_token: str) -> pd.DataFrame:
    """
    Fetch 1-minute candles for the given BANKNIFTY futures contract (NFO) —
    used for Supertrend.

    Uses getCandleData(). Candle timestamps are returned as
    "yyyy-MM-ddTHH:mm:ss+05:30" strings.

    Incremental fetch after initial seed to stay within API rate limits.
    The initial seed deliberately reaches back several CALENDAR days (not just
    LOOKBACK_BARS minutes) so that if the strategy is started at/soon after
    market open, the buffer is already backfilled with the previous trading
    session's candles.
    """
    global _candle_buffer

    end_dt = datetime.now(IST).replace(tzinfo=None)
    full_seed_needed = _candle_buffer is None or len(_candle_buffer) < ST_LENGTH + 5

    if full_seed_needed:
        # AngelOne allows up to 30 calendar days per ONE_MINUTE request, so
        # SEED_LOOKBACK_DAYS comfortably fits in a single call.
        start_dt = end_dt - timedelta(days=SEED_LOOKBACK_DAYS)
        log.info(
            f"Seeding BANKNIFTY candle buffer with prior-session history "
            f"(from {start_dt.strftime('%Y-%m-%d %H:%M')})…"
        )
    else:
        start_dt = end_dt - timedelta(minutes=15)

    historic_params = {
        "exchange":   EXCHANGE_NFO,
        "symboltoken": fut_token,
        "interval":   "ONE_MINUTE",
        "fromdate":   start_dt.strftime("%Y-%m-%d %H:%M"),
        "todate":     end_dt.strftime("%Y-%m-%d %H:%M"),
    }
    response = _call(
        "historical", smart_api.getCandleData,
        hard_timeout_seconds=60.0 if full_seed_needed else 20.0,
        historicDataParams=historic_params,
    )

    candles = (response or {}).get("data") or []
    if not candles:
        if _candle_buffer is not None and len(_candle_buffer) >= ST_LENGTH + 5:
            log.warning("No new candles returned — using cached buffer.")
            return _candle_buffer
        raise RuntimeError(
            f"No candle data returned for {fut_symbol} (BANKNIFTY futures). "
            f"Market may be closed or the symbol may differ. Response: {response}"
        )

    df_new = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    # Timestamps come back as "yyyy-MM-ddTHH:mm:ss+05:30" — strip the tz
    # offset since we work entirely in naive IST throughout this script.
    df_new["ts"] = pd.to_datetime(df_new["ts"], errors="coerce", utc=True).dt.tz_convert(IST).dt.tz_localize(None)

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
    the completed-candle trim in run_strategy).
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
# placeOrderFullResponse() is NOT idempotent. Unlike Groww, AngelOne has no
# "get order status by client reference" endpoint — the closest equivalent
# is tagging every order with a client-generated `ordertag` (max 20 chars)
# and, on a network/timeout error where we don't know if the request reached
# the exchange, searching the day's order book for a matching tag instead of
# blindly resubmitting. If the original request actually reached AngelOne but
# the response was lost on our side, a blind retry would submit a brand-new,
# SEPARATE order — a real duplicate on the exchange.
VARIETY_NORMAL      = "NORMAL"
PRODUCT_INTRADAY    = "INTRADAY"
DURATION_DAY        = "DAY"
ORDER_TYPE_LIMIT    = "LIMIT"
ORDER_TYPE_MARKET   = "MARKET"
TRANSACTION_TYPE_BUY  = "BUY"
TRANSACTION_TYPE_SELL = "SELL"

FILLED_STATUSES = ("complete",)
DEAD_STATUSES   = ("rejected", "cancelled")


def _generate_order_tag() -> str:
    """Short unique tag passed to placeOrder so a retry can look the attempt
    up in the order book instead of blindly resubmitting it. Must stay
    under AngelOne's 20-character ordertag limit."""
    return uuid.uuid4().hex[:16]


def _lookup_order_by_tag(
    smart_api: SmartConnect, order_tag: str, retries: int = 3
) -> dict | None:
    """
    Checks whether a place_order attempt with this ordertag actually
    reached AngelOne, by scanning today's order book.

    Returns:
      - a dict (the matching order-book row) if AngelOne shows an order
        for this tag — the original attempt succeeded.
      - None if the lookup POSITIVELY confirms no such order exists — safe
        to retry.

    Raises if the lookup itself can't be confirmed either way after
    `retries` attempts — deliberately does NOT return None in that case,
    since that would make an inconclusive check look identical to a
    confirmed "no order" and could let a real duplicate slip through.
    """
    for attempt in range(retries):
        try:
            resp = _call("order_book", smart_api.orderBook, hard_timeout_seconds=20.0)
            if not resp or not resp.get("status", False):
                raise RuntimeError(f"orderBook() returned status=False: {resp}")
            orders = resp.get("data") or []
            for order in orders:
                if order.get("ordertag") == order_tag:
                    return order
            return None  # lookup succeeded — genuinely no order for this tag
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5)
                continue
            raise RuntimeError(
                f"Could not confirm whether order tag={order_tag} "
                f"was placed — reconciliation lookup itself failed: {e}"
            ) from e


def _get_order_detail(smart_api: SmartConnect, unique_order_id: str) -> dict | None:
    """Fetch the current status of a single order via its uniqueorderid."""
    resp = _call(
        "non_trading", smart_api.individual_order_details, unique_order_id,
    )
    if not resp or not resp.get("status", False):
        return None
    return resp.get("data")


def _get_filled_quantity(detail: dict) -> float:
    """
    Best-effort extraction of how much quantity has actually executed on an
    order. Returns 0.0 if the field is missing/unparseable — we'd rather
    under-detect a partial fill than crash on an unexpected response schema.
    """
    val = detail.get("filledshares")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    return 0.0


def _place_order_safe(
    smart_api: SmartConnect,
    trading_symbol: str,
    symbol_token: str,
    transaction_type: str,
    quantity: int,
    order_type: str,
    label: str,
    price: float = 0.0,
    hard_timeout_seconds: float = 20.0,
) -> dict:
    """
    Places an order via smart_api.placeOrderFullResponse(), checking the
    order book by ordertag before retrying on a network/timeout error so a
    slow/lost response can never turn into a duplicate order on the exchange.

    Returns a dict with at least "orderid" and "uniqueorderid" keys (may be
    empty strings if the placement definitively failed and was rejected
    before an order was ever created — callers must handle that).
    """
    _throttle("orders")
    rate_limit_attempt = 0
    network_attempt = 0

    while True:
        order_tag = _generate_order_tag()
        order_params = {
            "variety":         VARIETY_NORMAL,
            "tradingsymbol":   trading_symbol,
            "symboltoken":     symbol_token,
            "transactiontype": transaction_type,
            "exchange":        EXCHANGE_NFO,
            "ordertype":       order_type,
            "producttype":     PRODUCT_INTRADAY,
            "duration":        DURATION_DAY,
            "price":           f"{price:.2f}" if order_type == ORDER_TYPE_LIMIT else "0",
            "quantity":        str(quantity),
            "ordertag":        order_tag,
        }
        try:
            resp = _call_with_hard_timeout(
                smart_api.placeOrderFullResponse, order_params,
                timeout_seconds=hard_timeout_seconds,
            )
            if resp and resp.get("status", False) and resp.get("data"):
                order_id        = resp["data"].get("orderid", "")
                unique_order_id = resp["data"].get("uniqueorderid", "")
                log.info(f"    {label} order response (tag={order_tag}) : {resp}")
                return {"orderid": order_id, "uniqueorderid": unique_order_id, "ordertag": order_tag}

            log.warning(f"{label}: place_order rejected. Response: {resp}")
            return {"orderid": "", "uniqueorderid": "", "ordertag": order_tag}

        except Exception as e:
            if _is_rate_limit_error(e):
                if rate_limit_attempt >= 3:
                    raise
                wait = 2 ** rate_limit_attempt
                log.warning(
                    f"{label}: AngelOne rate limit hit placing order (attempt "
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
                    f"(tag={order_tag}, attempt {network_attempt + 1}/"
                    f"{len(NETWORK_RETRY_DELAYS) + 1}) — will confirm via order "
                    f"book before deciding whether to retry: {e}"
                )
                time.sleep(wait)

                existing = _lookup_order_by_tag(smart_api, order_tag)
                if existing is not None:
                    log.warning(
                        f"{label}: order WAS placed on AngelOne despite the "
                        f"network error (tag={order_tag}, "
                        f"orderid={existing.get('orderid')}) — "
                        "using it instead of submitting a duplicate."
                    )
                    return {
                        "orderid":        existing.get("orderid", ""),
                        "uniqueorderid":  existing.get("uniqueorderid", ""),
                        "ordertag":       order_tag,
                    }

                log.warning(
                    f"{label}: confirmed tag={order_tag} was never created on "
                    "AngelOne — safe to retry."
                )
                network_attempt += 1
                _throttle("orders")
                continue

            raise


def _place_limit_order(
    smart_api: SmartConnect,
    trading_symbol: str,
    symbol_token: str,
    transaction_type: str,
    quantity: int,
    price: float,
    label: str,
) -> dict:
    log.info(
        f">>> Placing LIMIT {label}  qty={quantity} x {trading_symbol}  @ ₹{price:.2f}"
    )
    return _place_order_safe(
        smart_api, trading_symbol, symbol_token, transaction_type, quantity,
        ORDER_TYPE_LIMIT, label, price=price,
    )


def _place_market_order(
    smart_api: SmartConnect,
    trading_symbol: str,
    symbol_token: str,
    transaction_type: str,
    quantity: int,
    label: str,
) -> dict:
    """
    Place a MARKET order (no price). Used only as a last-resort EXIT
    fallback after all LIMIT escalation offsets fail to fill — see
    EXIT_MARKET_FALLBACK_ENABLED / _execute_with_escalation.
    """
    log.info(f">>> Placing MARKET {label}  qty={quantity} x {trading_symbol}")
    return _place_order_safe(
        smart_api, trading_symbol, symbol_token, transaction_type, quantity,
        ORDER_TYPE_MARKET, label,
    )


# Escalating price offsets (₹) tried in order until the order fills.
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


def _market_fallback(
    smart_api: SmartConnect,
    trading_symbol: str,
    symbol_token: str,
    transaction_type: str,
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

    resting_order_id may be "" — meaning the caller has ALREADY confirmed
    (via a rejected/cancelled order status) that there is nothing live to
    check or cancel. In that case both pre-checks below are skipped
    entirely and we go straight to placing the MARKET order.

    Always returns a (order_id, fill_price_or_None) tuple — same contract as
    _execute_with_escalation — so callers don't need special-casing. A
    fill_price of None means "treat as PENDING/UNCONFIRMED, verify on
    AngelOne", exactly as with the existing LIMIT-only behavior.
    """
    if resting_order_id:
        # Re-check the resting LIMIT order first — it may have filled in the
        # moments since the last poll, in which case there's nothing to cancel.
        try:
            detail = _get_order_detail(smart_api, resting_order_id)
            if detail and detail.get("orderstatus") in FILLED_STATUSES:
                fill_price = float(detail.get("averageprice") or 0)
                if fill_price:
                    log.info(
                        f"{label}: resting order {resting_order_id} filled right "
                        "before the MARKET fallback — using this fill instead."
                    )
                    return resting_order_id, fill_price
            if detail and _get_filled_quantity(detail) > 0:
                log.warning(
                    f"{label}: resting order {resting_order_id} shows a PARTIAL "
                    "fill right before the MARKET fallback. NOT cancelling or "
                    "placing a MARKET order on top of it. VERIFY ACTUAL POSITION ON ANGELONE."
                )
                return resting_order_id, None
        except Exception as e:
            log.warning(f"{label}: pre-cancel check failed for {resting_order_id}: {e}")

        # Cancel the resting LIMIT order before placing the MARKET order —
        # never want two live orders for the same exit at once.
        try:
            _call(
                "orders", smart_api.cancelOrder,
                order_id=resting_order_id, variety=VARIETY_NORMAL,
            )
            log.info(f"{label}: cancelled resting order {resting_order_id} ahead of MARKET fallback.")
        except Exception as e:
            log.warning(
                f"{label}: failed to cancel {resting_order_id} before the MARKET "
                f"fallback: {e}. Not placing a MARKET order on top of an order "
                "whose state is uncertain. Treating as PENDING/UNCONFIRMED. "
                "VERIFY ACTUAL POSITION ON ANGELONE."
            )
            return resting_order_id, None
    else:
        log.info(f"{label}: no resting order to check/cancel — going straight to MARKET.")

    # Place the MARKET order.
    try:
        resp = _place_market_order(
            smart_api, trading_symbol, symbol_token, transaction_type, quantity,
            f"{label} (MARKET fallback)"
        )
        market_order_id  = resp.get("orderid", "")
        market_unique_id = resp.get("uniqueorderid", "")
        if not market_order_id:
            log.warning(
                f"{label}: MARKET fallback place_order returned no orderid. "
                f"Response: {resp}. Original LIMIT order {resting_order_id} was "
                "already cancelled. VERIFY ACTUAL POSITION ON ANGELONE."
            )
            return resting_order_id, None
    except Exception as e:
        log.error(
            f"{label}: MARKET fallback order placement failed: {e}. Original "
            f"LIMIT order {resting_order_id} was already cancelled. VERIFY ACTUAL "
            "POSITION ON ANGELONE.",
            exc_info=True,
        )
        return resting_order_id, None

    # Poll the MARKET order for a fill.
    for _ in range(MARKET_FALLBACK_POLL_RETRIES):
        try:
            detail = _get_order_detail(smart_api, market_unique_id or market_order_id)
            if detail:
                status = detail.get("orderstatus")
                if status in FILLED_STATUSES:
                    fill_price = float(detail.get("averageprice") or 0)
                    if fill_price:
                        log.info(
                            f"{label}: MARKET fallback order {market_order_id} filled "
                            f"avg_fill_price=₹{fill_price:.2f}  status={status}"
                        )
                        return market_order_id, fill_price
                if status in DEAD_STATUSES:
                    log.warning(
                        f"{label}: MARKET fallback order {market_order_id} ended "
                        f"status={status} without a confirmed fill. Original LIMIT "
                        "order was already cancelled. VERIFY ACTUAL POSITION ON ANGELONE."
                    )
                    return market_order_id, None
                filled_qty = _get_filled_quantity(detail)
                if filled_qty > 0:
                    log.warning(
                        f"{label}: MARKET fallback order {market_order_id} shows a "
                        f"PARTIAL fill ({filled_qty}/{quantity}). Returning as "
                        "PENDING/UNCONFIRMED. VERIFY ACTUAL POSITION ON ANGELONE."
                    )
                    return market_order_id, None
        except Exception as e:
            log.warning(f"{label}: error polling MARKET fallback order {market_order_id}: {e}")
        time.sleep(MARKET_FALLBACK_POLL_INTERVAL)

    log.warning(
        f"{label}: MARKET fallback order {market_order_id} still not confirmed "
        f"filled after {MARKET_FALLBACK_POLL_RETRIES} polls. Treating as "
        "PENDING/UNCONFIRMED. Manual check on AngelOne recommended."
    )
    return market_order_id, None


def _execute_with_escalation(
    smart_api: SmartConnect,
    trading_symbol: str,
    symbol_token: str,
    transaction_type: str,
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
      - poll individual_order_details up to FILL_WAIT_RETRIES times
      - if filled ("complete" with averageprice), return (order_id, fill_price)
      - else cancel the resting order and retry with the next, wider offset

    If the final (widest) offset also doesn't fill:
      - if use_market_fallback is True and EXIT_MARKET_FALLBACK_ENABLED (a 4th
        retry, intended for EXITS only): cancel the resting LIMIT order and
        place ONE MARKET order, polling MARKET_FALLBACK_POLL_RETRIES times.
      - otherwise the order is LEFT RESTING (not cancelled) and
        (order_id, None) is returned — caller must treat the position as
        "order pending, fill unconfirmed" rather than assuming it's flat.

    Safety behavior mirrors the Groww-broker version: any exception AFTER an
    order has already been placed returns that order as PENDING/UNCONFIRMED
    instead of letting the exception propagate and lose track of a possibly-
    live order. Partial fills halt escalation entirely rather than risking a
    second position on top of the partial one.
    """
    last_order_id = ""
    last_live_order_id = ""

    for i, offset in enumerate(ESCALATION_OFFSETS):
        is_last_offset = (i == len(ESCALATION_OFFSETS) - 1)
        try:
            ltp = get_option_ltp(smart_api, trading_symbol, symbol_token)
            if side == "SELL":
                price = round(max(0.05, ltp - offset), 2)
            else:
                price = round(ltp + offset, 2)

            resp = _place_limit_order(
                smart_api, trading_symbol, symbol_token, transaction_type, quantity, price,
                f"{label} (offset=₹{offset})"
            )
            order_id        = resp.get("orderid", "")
            unique_order_id = resp.get("uniqueorderid", "")
            last_order_id = order_id or last_order_id
            last_live_order_id = order_id or last_live_order_id

            if not order_id:
                log.warning(f"{label}: place_order returned no orderid. Response: {resp}")
                continue

            partial_fill_seen  = False
            order_confirmed_dead = False  # rejected/cancelled — nothing to cancel, no fill
            status = None
            for _ in range(FILL_WAIT_RETRIES):
                try:
                    detail = _get_order_detail(smart_api, unique_order_id or order_id)
                    if detail is None:
                        time.sleep(FILL_WAIT_INTERVAL)
                        continue
                    status = detail.get("orderstatus")
                    if status in FILLED_STATUSES:
                        fill_price = float(detail.get("averageprice") or 0)
                        if fill_price:
                            log.info(
                                f"{label}: filled at offset=₹{offset}  "
                                f"avg_fill_price=₹{fill_price:.2f}  status={status}"
                            )
                            return order_id, fill_price
                    if status in DEAD_STATUSES:
                        log.warning(f"{label}: order {order_id} ended status={status} before fill.")
                        order_confirmed_dead = True
                        last_live_order_id = ""  # confirmed not live
                        break
                    filled_qty = _get_filled_quantity(detail)
                    if filled_qty > 0:
                        partial_fill_seen = True
                        log.warning(
                            f"{label}: order {order_id} shows a PARTIAL fill "
                            f"({filled_qty}/{quantity}) at offset=₹{offset}, status={status}. "
                            "Halting escalation — will NOT cancel/re-place at a new offset, to "
                            "avoid ending up with a second position on top of this partial one. "
                            "VERIFY ACTUAL POSITION ON ANGELONE."
                        )
                        break
                except Exception as e:
                    log.warning(f"Error polling order {order_id}: {e}")
                time.sleep(FILL_WAIT_INTERVAL)

            if partial_fill_seen:
                return order_id, None

            if order_confirmed_dead:
                if is_last_offset:
                    if use_market_fallback and EXIT_MARKET_FALLBACK_ENABLED:
                        log.warning(
                            f"{label}: order {order_id} was {status} at the widest "
                            f"offset (₹{offset}) — nothing live to cancel. Going "
                            "straight to a MARKET order (4th retry)."
                        )
                        return _market_fallback(
                            smart_api, trading_symbol, symbol_token, transaction_type,
                            quantity, label, ""
                        )
                    log.warning(
                        f"{label}: order {order_id} was {status} at the widest offset "
                        f"(₹{offset}) — every attempt failed. No order is live and no "
                        "position was created for this signal."
                    )
                    return "", None  # definitively no position — NOT "unconfirmed"
                log.info(
                    f"{label}: order {order_id} was {status} at offset=₹{offset} — "
                    "nothing to cancel, moving straight to the next offset."
                )
                continue

            if is_last_offset:
                if use_market_fallback and EXIT_MARKET_FALLBACK_ENABLED:
                    log.warning(
                        f"{label}: order {order_id} not filled even at widest offset "
                        f"(₹{offset}). Attempting 4th retry — cancel and place a MARKET order."
                    )
                    return _market_fallback(
                        smart_api, trading_symbol, symbol_token, transaction_type,
                        quantity, label, order_id
                    )

                log.warning(
                    f"{label}: order {order_id} not filled even at widest offset "
                    f"(₹{offset}). LEAVING ORDER RESTING — fill unconfirmed. "
                    "Manual check on AngelOne recommended."
                )
                return order_id, None

            try:
                _call(
                    "orders", smart_api.cancelOrder,
                    order_id=order_id, variety=VARIETY_NORMAL,
                )
                log.info(f"{label}: order {order_id} not filled at offset=₹{offset} — cancelled, retrying wider.")
                last_live_order_id = ""  # confirmed cancelled, no longer live
            except Exception as e:
                log.warning(
                    f"{label}: failed to cancel unfilled order {order_id} (offset=₹{offset}): {e}. "
                    "Order state is now uncertain — halting escalation instead of placing another "
                    "order on top of it. Treating as PENDING/UNCONFIRMED. "
                    "VERIFY ACTUAL POSITION ON ANGELONE."
                )
                return order_id, None

            # Re-check right after the cancel ack, in case the exchange
            # filled the order in the brief window before/at cancellation.
            try:
                post_cancel = _get_order_detail(smart_api, unique_order_id or order_id)
                if post_cancel and post_cancel.get("orderstatus") in FILLED_STATUSES:
                    fill_price = float(post_cancel.get("averageprice") or 0)
                    if fill_price:
                        log.warning(
                            f"{label}: order {order_id} actually FILLED right at cancel time "
                            f"(avg_fill_price=₹{fill_price:.2f}) — using this fill instead of "
                            "escalating to a new order."
                        )
                        return order_id, fill_price
                post_cancel_qty = _get_filled_quantity(post_cancel) if post_cancel else 0.0
                if post_cancel_qty > 0:
                    log.warning(
                        f"{label}: order {order_id} shows a PARTIAL fill ({post_cancel_qty}/{quantity}) "
                        "right at cancel time. Halting escalation instead of placing a new order. "
                        "VERIFY ACTUAL POSITION ON ANGELONE."
                    )
                    return order_id, None
            except Exception as e:
                log.warning(f"{label}: post-cancel check failed for order {order_id}: {e}")

        except Exception as exc:
            if last_live_order_id:
                log.error(
                    f"{label}: unexpected error while order {last_live_order_id} may still be "
                    f"live (offset attempt {i + 1}/{len(ESCALATION_OFFSETS)}): {exc}. Returning "
                    "this order as PENDING/UNCONFIRMED instead of retrying with a new order, to "
                    "avoid duplicating a possibly-live position. VERIFY ACTUAL POSITION ON ANGELONE.",
                    exc_info=True,
                )
                return last_live_order_id, None
            log.error(
                f"{label}: unexpected error with no order currently live "
                f"(offset attempt {i + 1}/{len(ESCALATION_OFFSETS)}): {exc}. "
                "Nothing to lose track of, so re-raising is safe.",
                exc_info=True,
            )
            raise

    return last_live_order_id, None


def sell_option(
    smart_api: SmartConnect, trading_symbol: str, symbol_token: str, quantity: int, label: str
) -> tuple[str, float | None]:
    """SELL to open, escalating the LIMIT price offset (₹2 → ₹5 → ₹10) until filled."""
    return _execute_with_escalation(
        smart_api, trading_symbol, symbol_token, TRANSACTION_TYPE_SELL, quantity, "SELL", f"SELL {label}"
    )


def buy_to_cover_option(
    smart_api: SmartConnect, trading_symbol: str, symbol_token: str, quantity: int, label: str
) -> tuple[str, float | None]:
    """
    BUY to cover a short, escalating the LIMIT price offset (₹2 → ₹5 → ₹10)
    until filled. If all three LIMIT offsets fail, a 4th retry cancels the
    resting order and fires one MARKET order (see EXIT_MARKET_FALLBACK_ENABLED).
    """
    return _execute_with_escalation(
        smart_api, trading_symbol, symbol_token, TRANSACTION_TYPE_BUY, quantity, "BUY",
        f"BUY (cover) {label}", use_market_fallback=True,
    )


def switch_strike_on_profit(
    smart_api: SmartConnect,
    pos: dict,
    option_type: str,   # "PE" or "CE"
    leg_label: str,     # "PUT SELL" or "CALL SELL"
    instruments_df: pd.DataFrame,
    expiry: pd.Timestamp,
    fut_symbol: str,
    fut_token: str,
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
        or if the re-entry definitively failed.
    """
    symbol       = pos["symbol"]
    symbol_token = pos["symbol_token"]
    entry_price  = pos.get("entry_price", 0.0)
    quantity     = QUANTITY * pos["lot_size"]

    log.info(
        f"[PROFIT SWITCH] {leg_label}  symbol={symbol}  "
        f"entry=₹{entry_price:.2f}  profit >= {PROFIT_SWITCH_THRESHOLD} pts  "
        "→ Exiting current strike and re-entering new ATM strike."
    )

    # ── Step 1: Buy-to-cover current strike ─────────────────────────────────
    cover_order_id, cover_fill = buy_to_cover_option(smart_api, symbol, symbol_token, quantity, leg_label)
    exit_time = datetime.now(IST)

    if cover_fill is None and cover_order_id:
        # Cover fill is unconfirmed — keep leg open with exit_pending.
        # Do NOT attempt a re-entry on top of an unresolved short.
        pnl_est = None
        try:
            exit_ltp = get_option_ltp(smart_api, symbol, symbol_token)
            pnl_est  = (entry_price - exit_ltp) * quantity if entry_price else None
        except Exception:
            pass
        log.warning(
            f"[PROFIT SWITCH] {leg_label}: cover order {cover_order_id} fill UNCONFIRMED — "
            "leg kept OPEN (exit_pending=True). Re-entry skipped until cover resolves. "
            "VERIFY ACTUAL POSITION ON ANGELONE."
        )
        log_trade(
            leg=leg_label, action="EXIT", symbol=symbol, quantity=quantity,
            price=None, fill_confirmed=False, order_id=cover_order_id,
            reason=f"Profit switch (>= {PROFIT_SWITCH_THRESHOLD} pts) — cover UNCONFIRMED",
            pnl=pnl_est, when=exit_time,
        )
        pos["exit_pending"]  = True
        pos["exit_order_id"] = cover_order_id
        pos["exit_reason"]   = f"profit switch >= {PROFIT_SWITCH_THRESHOLD} pts"
        pos["exit_quantity"] = quantity
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
    pnl = (entry_price - cover_price) * quantity if entry_price and cover_price else None
    log.info(
        f"[PROFIT SWITCH] {leg_label}: cover confirmed  symbol={symbol}  "
        f"entry=₹{entry_price:.2f}  cover=₹{cover_price:.2f}  "
        f"P&L=₹{pnl:.2f}  time={exit_time:%Y-%m-%d %H:%M:%S}"
    )
    log_trade(
        leg=leg_label, action="EXIT", symbol=symbol, quantity=quantity,
        price=cover_price, fill_confirmed=True, order_id=cover_order_id,
        reason=f"Profit switch (>= {PROFIT_SWITCH_THRESHOLD} pts)",
        pnl=pnl, when=exit_time,
    )

    # ── Step 2: Re-enter on new ATM strike ──────────────────────────────────
    try:
        bn_ltp  = get_banknifty_ltp(smart_api, fut_symbol, fut_token)
        new_atm = get_atm_strike(bn_ltp)
        log.info(
            f"[PROFIT SWITCH] {leg_label}: BankNifty LTP={bn_ltp:.2f}  "
            f"new ATM strike={new_atm}  →  Re-selling {option_type} (min premium ₹{MIN_OPTION_PRICE})"
        )
        new_sym, new_token, new_lot, new_ltp_check = find_option_with_min_price(
            smart_api, instruments_df, expiry, new_atm, option_type
        )
    except Exception as e:
        log.error(
            f"[PROFIT SWITCH] {leg_label}: failed to resolve new ATM strike after cover: {e}. "
            "Re-entry skipped — leg is now FLAT."
        )
        return None

    new_qty = QUANTITY * new_lot
    new_order_id, new_fill = sell_option(smart_api, new_sym, new_token, new_qty, leg_label)
    re_entry_time = datetime.now(IST)

    if not new_order_id:
        log.error(
            f"[PROFIT SWITCH] {leg_label}: re-entry sell order FAILED for all offsets — "
            f"symbol={new_sym}  qty={new_qty}. Leg is FLAT after the cover."
        )
        log_trade(
            leg=leg_label, action="ENTRY", symbol=new_sym, quantity=new_qty,
            price=new_ltp_check, fill_confirmed=False, order_id="",
            reason="Profit switch re-entry — FAILED", when=re_entry_time,
        )
        return None

    new_pos = {
        "symbol":       new_sym,
        "symbol_token": new_token,
        "lot_size":     new_lot,
        "entry_price":  new_fill,    # None if unconfirmed
        "entry_time":   re_entry_time,
        "order_id":     new_order_id,
    }
    if new_fill is not None:
        log.info(
            f"[PROFIT SWITCH] {leg_label}: re-entry confirmed  symbol={new_sym}  "
            f"strike={new_atm}  fill=₹{new_fill:.2f}  qty={new_qty}  "
            f"time={re_entry_time:%Y-%m-%d %H:%M:%S}  order_id={new_order_id}"
        )
    else:
        log.warning(
            f"[PROFIT SWITCH] {leg_label}: re-entry order placed but fill UNCONFIRMED  "
            f"symbol={new_sym}  qty={new_qty}  time={re_entry_time:%Y-%m-%d %H:%M:%S}  "
            f"order_id={new_order_id}. VERIFY ACTUAL POSITION ON ANGELONE."
        )
    log_trade(
        leg=leg_label, action="ENTRY", symbol=new_sym, quantity=new_qty,
        price=new_fill if new_fill is not None else new_ltp_check,
        fill_confirmed=new_fill is not None, order_id=new_order_id,
        reason=f"Profit switch re-entry (>= {PROFIT_SWITCH_THRESHOLD} pts on prev strike)",
        when=re_entry_time,
    )
    return new_pos


# ─── ORDER FILL PRICE POLLER ─────────────────────────────────────────────────
def _await_fill_price(smart_api: SmartConnect, order_id: str, retries: int = 6) -> float | None:
    """
    Poll order detail until the order is filled.
    Returns average_fill_price, or None on timeout (caller falls back to LTP).
    """
    for _ in range(retries):
        try:
            detail = _get_order_detail(smart_api, order_id)
            if detail and detail.get("orderstatus") in FILLED_STATUSES:
                price = float(detail.get("averageprice") or 0)
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
    smart_api: SmartConnect,
    pos: dict,
    quantity: int,
    leg_label: str,
    reason: str = "force square-off",
) -> dict:
    """
    Buy-to-cover a short option position.
    pos dict: {"symbol": str, "symbol_token": str, "entry_price": float, "order_id": str}

    Returns a trade record dict suitable for the trade journal.
    """
    symbol       = pos["symbol"]
    symbol_token = pos["symbol_token"]
    entry_price  = pos.get("entry_price", 0.0)
    entry_time   = pos.get("entry_time")
    entry_str    = f"₹{entry_price:.2f}" if entry_price else "unknown"
    log.warning(f"[{reason}] Covering short {leg_label}  entry={entry_str}  symbol={symbol}")

    order_id, fill_price = buy_to_cover_option(smart_api, symbol, symbol_token, quantity, leg_label)
    exit_time            = datetime.now(IST)
    exit_fill_confirmed  = fill_price is not None
    exit_price           = fill_price if fill_price is not None else get_option_ltp(smart_api, symbol, symbol_token)

    hold_str = ""
    if entry_time is not None:
        hold_str = f"  held={str(exit_time - entry_time).split('.')[0]}"

    if not exit_fill_confirmed:
        log.warning(
            f"    {leg_label} cover order {order_id} fill UNCONFIRMED — "
            f"using LTP ₹{exit_price:.2f} for logging only. "
            "VERIFY ACTUAL POSITION ON ANGELONE."
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
    smart_api: SmartConnect,
    pos: dict,
    quantity: int,
    leg_label: str,
    reason: str = "force square-off",
) -> dict | None:
    """
    Wraps square_off_position() and decides whether the leg can actually be
    marked flat, instead of the caller blindly clearing its position state.

    Returns:
      - None      → exit fill CONFIRMED. Leg is genuinely flat.
      - a pos dict → exit fill UNCONFIRMED. Leg is kept OPEN with
                     exit_pending=True so it is never silently "forgotten"
                     while a cover order may still be live. The main loop's
                     pending-exit resolver checks exit_order_id on each
                     cycle and finalizes the leg once the fill is confirmed.
    """
    trade = square_off_position(smart_api, pos, quantity, leg_label, reason=reason)
    if trade["exit_fill_confirmed"]:
        return None

    pos["exit_pending"]  = True
    pos["exit_order_id"] = trade["exit_order_id"]
    pos["exit_reason"]   = reason
    pos["exit_quantity"] = quantity
    log.warning(
        f"{leg_label}: cover order unconfirmed — leg kept OPEN (exit_pending=True) "
        f"instead of being cleared, so it isn't lost or duplicated. Will keep checking "
        f"order {trade['exit_order_id']} until it resolves. VERIFY ACTUAL POSITION ON ANGELONE."
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
    # authentication fails, wait gracefully instead of crashing and being
    # immediately restarted by a watchdog or service.
    while True:
        now_ist = datetime.now(IST)
        now_hm  = now_ist.strftime("%H:%M")
        status  = market_status(now_hm)

        try:
            smart_api = authenticate()
            break
        except Exception as auth_err:
            sleep_s = 300 if status == "CLOSED" else 60
            log.warning(
                f"Authentication failed ({auth_err}). "
                f"Sleeping {sleep_s} s before retrying "
                f"(market status: {status})…"
            )
            time.sleep(sleep_s)

    last_auth_time = datetime.now(IST)

    instruments_df = get_cached_instruments_df()
    expiry         = get_monthly_expiry_date(instruments_df)
    fut_symbol, fut_token = get_active_banknifty_fut_symbol(instruments_df)

    # State for each leg
    # None → flat; dict → {"symbol", "symbol_token", "lot_size", "entry_price", "order_id"}
    put_pos:  dict | None = None
    call_pos: dict | None = None
    prev_direction: int | None = None
    # Tracks whether the "enter immediately, no flip required" first-trade
    # logic has fired yet. Deliberately separate from `prev_direction is None`
    # -- see the matching comment in the Groww-broker version for the full
    # rationale (prevents the immediate-entry signal from being silently
    # discarded if the script starts before ENTRY_START_TIME).
    first_trade_done: bool = False

    log.info(
        f"Strategy started.  Underlying: {UNDERLYING}  "
        f"Monthly expiry: {expiry.date()}  "
        f"Supertrend({ST_LENGTH}, {ST_FACTOR})  "
        f"Exit threshold: ±{EXIT_POINTS_THRESHOLD} pts  "
        f"Entry window: {ENTRY_START_TIME}–{ENTRY_END_TIME}  "
        f"Force-close: {SQUARE_OFF_TIME}"
    )

    def _leg_status_label(api_client: SmartConnect, leg_pos: dict | None, leg_name: str) -> str:
        """Build a per-minute status string for one leg, including the
        LIVE LTP whenever a position is open.

        Defined once outside the main loop so Python doesn't recreate the
        function object on every 60-second iteration (~900 times/day).
        api_client is passed explicitly so this works correctly after a
        mid-session re-authentication (where smart_api may be reassigned).
        Never lets an LTP-fetch hiccup break the per-minute summary log —
        falls back to entry-only text on any error."""
        if not leg_pos or leg_pos.get("entry_price") is None:
            return f"{leg_name}: FLAT"
        entry = leg_pos["entry_price"]
        try:
            live_ltp = get_option_ltp(api_client, leg_pos["symbol"], leg_pos["symbol_token"])
            diff     = live_ltp - entry   # positive = loss for the seller
            return (
                f"{leg_name} SELL entry=₹{entry:.2f}  LTP=₹{live_ltp:.2f}  "
                f"diff={diff:+.2f}"
            )
        except Exception as e:
            return f"{leg_name} SELL entry=₹{entry:.2f}  LTP=unavailable ({e})"

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
                if now_wd < 5 and now_hm >= STRATEGY_WAKE_TIME and now_hm < MARKET_OPEN:
                    log.info(
                        f"[{now_ts}] Pre-market ({now_hm}). "
                        f"Waiting for market open at {MARKET_OPEN} IST — sleeping 30 s."
                    )
                    time.sleep(30)
                    continue

                # Weekend or genuine overnight closure: sleep until STRATEGY_WAKE_TIME
                # on the next trading day (Monday if today is Fri/Sat/Sun).
                wh, wm     = (int(p) for p in STRATEGY_WAKE_TIME.split(":"))
                wake_today = now_ist_c.replace(hour=wh, minute=wm, second=0, microsecond=0)
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
                reset_candle_buffer()
                continue

            # ── Force square-off at SQUARE_OFF_TIME ───────────────────────────
            if status == "SQUAREOFF":
                had_positions = put_pos is not None or call_pos is not None
                if put_pos is not None:
                    try:
                        put_pos = attempt_square_off(
                            smart_api, put_pos, QUANTITY * put_pos["lot_size"],
                            "PUT SELL", reason="SQUARE-OFF TIME"
                        )
                    except Exception as e:
                        log.error(
                            f"Failed to square off PUT SELL position "
                            f"(symbol={put_pos['symbol']}): {e}. "
                            "MANUAL INTERVENTION REQUIRED — check open positions on AngelOne."
                        )
                if call_pos is not None:
                    try:
                        call_pos = attempt_square_off(
                            smart_api, call_pos, QUANTITY * call_pos["lot_size"],
                            "CALL SELL", reason="SQUARE-OFF TIME"
                        )
                    except Exception as e:
                        log.error(
                            f"Failed to square off CALL SELL position "
                            f"(symbol={call_pos['symbol']}): {e}. "
                            "MANUAL INTERVENTION REQUIRED — check open positions on AngelOne."
                        )
                if put_pos is not None or call_pos is not None:
                    log.warning(
                        "One or more legs could not be confirmed flat at force-close "
                        "(unconfirmed cover order or square-off failure). "
                        "MANUAL INTERVENTION REQUIRED — check open positions on AngelOne."
                    )
                if not had_positions:
                    log.info(f"[{now_ts}] SQUARE-OFF TIME — no open positions.")
                log.info("Session ended. Exiting strategy.")
                break

            # ── Fetch 1-min candles for BANKNIFTY futures ─────────────────────
            df = fetch_1min_candles(smart_api, fut_symbol, fut_token)

            # Drop the last row if it's a still-forming (not yet closed) candle.
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
                    f"Cross-check this timestamp/close against your charting "
                    f"platform's last closed 1-min candle on the SAME symbol "
                    f"(BankNifty FUTURES, not the spot index) to confirm alignment."
                )

            # ── Supertrend direction ──────────────────────────────────────────
            curr_direction = compute_supertrend_direction(df, ST_LENGTH, ST_FACTOR)
            last_close     = float(df.iloc[-1]["close"])
            trend_label    = "BULLISH ▲" if curr_direction == 1 else "BEARISH ▼"

            put_label  = _leg_status_label(smart_api, put_pos, "PUT")
            call_label = _leg_status_label(smart_api, call_pos, "CALL")
            log.info(
                f"[{now_ts}]  ST={trend_label}  BankNifty Close={last_close:.2f}  "
                f"|  {put_label}  |  {call_label}"
            )

            entry_allowed = ENTRY_START_TIME <= now_hm <= ENTRY_END_TIME

            # First trade of the day: enter immediately on the current Supertrend
            # direction once the entry window is actually open — no flip required.
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
                    prev_direction = curr_direction

            # ── Recover missing entry prices ──────────────────────────────────
            for leg_pos, leg_name in [(put_pos, "PUT"), (call_pos, "CALL")]:
                if leg_pos is not None and leg_pos.get("entry_price") is None:
                    try:
                        confirmed = _await_fill_price(smart_api, leg_pos["order_id"])
                        if confirmed is not None:
                            leg_pos["entry_price"] = confirmed
                            log.info(
                                f"[TRADE] {leg_name} ENTRY_RECOVERED  "
                                f"symbol={leg_pos['symbol']}  fill=₹{confirmed:.2f}  "
                                f"order_id={leg_pos['order_id']}  reason=late fill confirmed"
                            )
                        else:
                            ltp = get_option_ltp(smart_api, leg_pos["symbol"], leg_pos["symbol_token"])
                            leg_pos["entry_price"] = ltp
                            log.warning(
                                f"{leg_name} entry order {leg_pos['order_id']} still unfilled — "
                                f"using LTP ₹{ltp:.2f} as a provisional entry price. "
                                "VERIFY ACTUAL POSITION ON ANGELONE."
                            )
                    except Exception as e:
                        log.warning(f"Could not recover {leg_name} entry price: {e}")

            # ── Resolve pending exits ──────────────────────────────────────────
            for leg_pos, leg_name, leg_label in (
                (put_pos, "PUT", "PUT SELL"), (call_pos, "CALL", "CALL SELL")
            ):
                if leg_pos is None or not leg_pos.get("exit_pending"):
                    continue
                try:
                    confirmed = _await_fill_price(smart_api, leg_pos["exit_order_id"])
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
                    bn_ltp    = get_banknifty_ltp(smart_api, fut_symbol, fut_token)
                    atm       = get_atm_strike(bn_ltp)
                    log.info(
                        f"SIGNAL  ST → BULLISH  BankNifty LTP={bn_ltp:.2f}  "
                        f"ATM strike={atm}  →  Selecting PUT (min premium ₹{MIN_OPTION_PRICE})"
                    )
                    instruments_df = get_cached_instruments_df()
                    expiry         = get_monthly_expiry_date(instruments_df)
                    put_sym, put_token, put_lot, put_ltp_check = find_option_with_min_price(
                        smart_api, instruments_df, expiry, atm, "PE"
                    )
                    trade_qty  = QUANTITY * put_lot
                    order_id, fill_price = sell_option(smart_api, put_sym, put_token, trade_qty, "PUT")
                    entry_time = datetime.now(IST)
                    if not order_id:
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
                        put_pos = {
                            "symbol":       put_sym,
                            "symbol_token": put_token,
                            "lot_size":     put_lot,
                            "entry_price":  fill_price,   # None if fill unconfirmed
                            "entry_time":   entry_time,
                            "order_id":     order_id,
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
                                "— fill price unknown. VERIFY ACTUAL POSITION ON ANGELONE."
                            )
                        log_trade(
                            leg="PUT SELL", action="ENTRY", symbol=put_sym, quantity=trade_qty,
                            price=fill_price if fill_price is not None else put_ltp_check,
                            fill_confirmed=fill_price is not None, order_id=order_id,
                            reason="ST flip BULLISH", when=entry_time,
                        )

            # ── Profit-based strike switch: PUT leg ───────────────────────────
            if put_pos is not None and not put_pos.get("exit_pending") and put_pos.get("entry_price") is not None:
                _put_ltp_switch = get_option_ltp(smart_api, put_pos["symbol"], put_pos["symbol_token"])
                _put_profit_pts = put_pos["entry_price"] - _put_ltp_switch  # positive = profit for seller
                if _put_profit_pts >= PROFIT_SWITCH_THRESHOLD:
                    log.info(
                        f"[PROFIT SWITCH] PUT SELL: profit={_put_profit_pts:.2f} pts >= "
                        f"{PROFIT_SWITCH_THRESHOLD} pts threshold  "
                        f"(entry=₹{put_pos['entry_price']:.2f}  LTP=₹{_put_ltp_switch:.2f})  "
                        "→ Switching to new nearest ATM PUT strike."
                    )
                    instruments_df = get_cached_instruments_df()
                    expiry         = get_monthly_expiry_date(instruments_df)
                    put_pos = switch_strike_on_profit(
                        smart_api, put_pos, "PE", "PUT SELL",
                        instruments_df, expiry, fut_symbol, fut_token
                    )

            # Exit: ST turned BEARISH AND P/L threshold reached
            # NOTE: deliberately `if`, not `elif` off the profit-switch block above —
            # see the matching Groww-broker comment: chaining these as if/elif
            # would silently swallow this exit check whenever a position is open
            # but under the profit-switch threshold.
            if curr_direction == -1 and put_pos is not None:
                if put_pos.get("exit_pending"):
                    log.info(
                        "    PUT SELL: exit already pending confirmation — "
                        "skipping a new cover attempt this cycle."
                    )
                elif put_pos.get("entry_price") is None:
                    log.warning("PUT SELL: entry price unknown, skipping exit check.")
                else:
                    put_ltp  = get_option_ltp(smart_api, put_pos["symbol"], put_pos["symbol_token"])
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
                            smart_api, put_pos, trade_qty, "PUT SELL",
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
                    bn_ltp    = get_banknifty_ltp(smart_api, fut_symbol, fut_token)
                    atm       = get_atm_strike(bn_ltp)
                    log.info(
                        f"SIGNAL  ST → BEARISH  BankNifty LTP={bn_ltp:.2f}  "
                        f"ATM strike={atm}  →  Selecting CALL (min premium ₹{MIN_OPTION_PRICE})"
                    )
                    instruments_df = get_cached_instruments_df()
                    expiry         = get_monthly_expiry_date(instruments_df)
                    call_sym, call_token, call_lot, call_ltp_check = find_option_with_min_price(
                        smart_api, instruments_df, expiry, atm, "CE"
                    )
                    trade_qty  = QUANTITY * call_lot
                    order_id, fill_price = sell_option(smart_api, call_sym, call_token, trade_qty, "CALL")
                    entry_time = datetime.now(IST)
                    if not order_id:
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
                        call_pos = {
                            "symbol":       call_sym,
                            "symbol_token": call_token,
                            "lot_size":     call_lot,
                            "entry_price":  fill_price,   # None if fill unconfirmed
                            "entry_time":   entry_time,
                            "order_id":     order_id,
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
                                "— fill price unknown. VERIFY ACTUAL POSITION ON ANGELONE."
                            )
                        log_trade(
                            leg="CALL SELL", action="ENTRY", symbol=call_sym, quantity=trade_qty,
                            price=fill_price if fill_price is not None else call_ltp_check,
                            fill_confirmed=fill_price is not None, order_id=order_id,
                            reason="ST flip BEARISH", when=entry_time,
                        )

            # ── Profit-based strike switch: CALL leg ──────────────────────────
            if call_pos is not None and not call_pos.get("exit_pending") and call_pos.get("entry_price") is not None:
                _call_ltp_switch = get_option_ltp(smart_api, call_pos["symbol"], call_pos["symbol_token"])
                _call_profit_pts = call_pos["entry_price"] - _call_ltp_switch  # positive = profit for seller
                if _call_profit_pts >= PROFIT_SWITCH_THRESHOLD:
                    log.info(
                        f"[PROFIT SWITCH] CALL SELL: profit={_call_profit_pts:.2f} pts >= "
                        f"{PROFIT_SWITCH_THRESHOLD} pts threshold  "
                        f"(entry=₹{call_pos['entry_price']:.2f}  LTP=₹{_call_ltp_switch:.2f})  "
                        "→ Switching to new nearest ATM CALL strike."
                    )
                    instruments_df = get_cached_instruments_df()
                    expiry         = get_monthly_expiry_date(instruments_df)
                    call_pos = switch_strike_on_profit(
                        smart_api, call_pos, "CE", "CALL SELL",
                        instruments_df, expiry, fut_symbol, fut_token
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
                    call_ltp = get_option_ltp(smart_api, call_pos["symbol"], call_pos["symbol_token"])
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
                            smart_api, call_pos, trade_qty, "CALL SELL",
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
                        smart_api, put_pos, QUANTITY * put_pos["lot_size"],
                        "PUT SELL", reason="KeyboardInterrupt"
                    )
                except Exception as e:
                    log.error(
                        f"Failed to square off PUT SELL position "
                        f"(symbol={put_pos['symbol']}): {e}. "
                        "MANUAL INTERVENTION REQUIRED — check open positions on AngelOne."
                    )
            if call_pos is not None:
                try:
                    call_pos = attempt_square_off(
                        smart_api, call_pos, QUANTITY * call_pos["lot_size"],
                        "CALL SELL", reason="KeyboardInterrupt"
                    )
                except Exception as e:
                    log.error(
                        f"Failed to square off CALL SELL position "
                        f"(symbol={call_pos['symbol']}): {e}. "
                        "MANUAL INTERVENTION REQUIRED — check open positions on AngelOne."
                    )
            log.info("Strategy stopped by user.")
            break

        except Exception as exc:
            if _is_rate_limit_error(exc):
                log.warning(
                    f"AngelOne rate limit persisted past internal retries: {exc}. "
                    "Sleeping 30 s before resuming…"
                )
                time.sleep(30)
            elif _is_auth_error(exc) or "forbidden" in str(exc).lower():
                # jwtToken has expired or been invalidated — force a fresh
                # re-authentication immediately. Sleeping-and-retrying never
                # recovers from this because every subsequent API call will
                # also fail with the same auth error until a new token is
                # obtained.
                now_ist = datetime.now(IST)
                age     = now_ist - last_auth_time
                log.warning(
                    f"AngelOne API authentication error (token age={age}): {exc}. "
                    "Forcing token refresh and re-authenticating…"
                )
                try:
                    if os.path.exists(TOKEN_CACHE_FILE):
                        os.remove(TOKEN_CACHE_FILE)
                        log.info(f"Deleted stale cached token: {TOKEN_CACHE_FILE}")
                except OSError as cache_err:
                    log.warning(f"Could not delete token cache file: {cache_err}")
                try:
                    smart_api      = authenticate(force_refresh=True)
                    last_auth_time = datetime.now(IST)
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
                log.warning(
                    f"Network/timeout error to AngelOne API persisted past "
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
