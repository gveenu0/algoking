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

Entry window  : 09:23 – 15:05 IST  (no new entries outside this window)
Force close   : 15:08 IST  (both legs closed regardless of P&L)

Notes:
  • ATM strike is determined from the BANKNIFTY underlying LTP, rounded to nearest 100.
  • Monthly expiry = the furthest-dated contract expiring in the current calendar month.
    If none remain this month, uses the next month's monthly expiry.
  • Both legs (CE and PE) run independently; having a CE trade does NOT block PE trade.
  • Each transaction is logged with the execution price.
"""

import time
import csv
import os
import logging
import warnings
import pyotp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from growwapi import GrowwAPI


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

# Trade journal — every fill (entry and exit) is appended as a CSV row.
# File is created with a header if it doesn't exist yet. Safe to run the
# script daily; rows simply accumulate (one file = full trade history).
TRADE_LOG_PATH = "trade_journal.csv"
TRADE_LOG_FIELDS = [
    "timestamp_ist", "date", "leg", "action", "symbol", "quantity",
    "price", "fill_confirmed", "order_id", "entry_price", "exit_price",
    "pnl", "reason",
]

IST = timezone(timedelta(hours=5, minutes=30))


# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ─── TRADE JOURNAL (CSV) ──────────────────────────────────────────────────────
def log_trade(record: dict) -> None:
    """
    Append one row to the trade journal CSV (TRADE_LOG_PATH).
    `record` should contain keys matching (a subset of) TRADE_LOG_FIELDS;
    any missing keys are written as empty strings.

    Designed to be safe to call from anywhere — on failure it logs a
    warning rather than raising, so a journal write never breaks the
    trading loop.
    """
    try:
        now_ist = datetime.now(IST)
        row = {field: record.get(field, "") for field in TRADE_LOG_FIELDS}
        row["timestamp_ist"] = now_ist.strftime("%Y-%m-%d %H:%M:%S")
        row["date"]          = now_ist.strftime("%Y-%m-%d")

        file_exists = os.path.isfile(TRADE_LOG_PATH)
        with open(TRADE_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        log.warning(f"Failed to write trade journal entry: {e}. Record: {record}")


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
    df = groww.get_all_instruments()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    return df


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


# ─── BANKNIFTY UNDERLYING LTP ─────────────────────────────────────────────────
def get_banknifty_ltp(groww: GrowwAPI, fut_symbol: str) -> float:
    """Fetch the last traded price of the nearest BANKNIFTY futures contract (NSE FNO).
    The cash index symbol is not reliably supported by the Groww LTP API."""
    key = f"NSE_{fut_symbol}"
    resp = groww.get_ltp(
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
    resp = groww.get_ltp(
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
    """
    global _candle_buffer

    end_dt = datetime.now(IST).replace(tzinfo=None)
    full_seed_needed = _candle_buffer is None or len(_candle_buffer) < ST_LENGTH + 5

    if full_seed_needed:
        start_dt = end_dt - timedelta(minutes=LOOKBACK_BARS + 30)
        log.info("Seeding BANKNIFTY candle buffer with full history…")
    else:
        start_dt = end_dt - timedelta(minutes=15)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        response = groww.get_historical_candles(
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
        _candle_buffer = df_new.sort_values("ts").reset_index(drop=True)
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
    Returns +1 (bullish) or -1 (bearish) for the last COMPLETED candle.
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

    return int(direction[-2])


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
    resp = groww.place_order(
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


# Escalating price offsets (₹) tried in order until the order fills.
# Groww's API does not support MARKET orders for this product, so we widen
# the LIMIT price step by step to chase a fill without crossing the full
# spread blindly on the first attempt.
ESCALATION_OFFSETS = (2, 5, 10)
FILL_WAIT_RETRIES  = 6     # polls per offset attempt
FILL_WAIT_INTERVAL = 1.0   # seconds between polls


def _execute_with_escalation(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    side: str,          # "SELL" or "BUY"
    label: str,
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

    If the final (widest) offset also doesn't fill, the order is LEFT RESTING
    (not cancelled) and (order_id, None) is returned — caller must treat the
    position as "order pending, fill unconfirmed" rather than assuming it's flat.
    """
    FILLED_STATUSES = ("EXECUTED", "COMPLETED", "DELIVERY_AWAITED")
    last_order_id = ""

    for i, offset in enumerate(ESCALATION_OFFSETS):
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

        for _ in range(FILL_WAIT_RETRIES):
            try:
                detail = groww.get_order_detail(groww_order_id=order_id, segment=groww.SEGMENT_FNO)
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
            except Exception as e:
                log.warning(f"Error polling order {order_id}: {e}")
            time.sleep(FILL_WAIT_INTERVAL)

        # Not filled at this offset.
        is_last_offset = (i == len(ESCALATION_OFFSETS) - 1)
        if is_last_offset:
            log.warning(
                f"{label}: order {order_id} not filled even at widest offset "
                f"(₹{offset}). LEAVING ORDER RESTING — fill unconfirmed. "
                "Manual check on Groww recommended."
            )
            return order_id, None

        try:
            groww.cancel_order(groww_order_id=order_id, segment=groww.SEGMENT_FNO)
            log.info(f"{label}: order {order_id} not filled at offset=₹{offset} — cancelled, retrying wider.")
        except Exception as e:
            log.warning(
                f"{label}: failed to cancel unfilled order {order_id} (offset=₹{offset}): {e}. "
                "Retrying with a wider offset anyway — risk of duplicate resting orders, check manually."
            )

    return last_order_id, None


def sell_option(groww: GrowwAPI, trading_symbol: str, quantity: int, label: str) -> tuple[str, float | None]:
    """SELL to open, escalating the LIMIT price offset (₹2 → ₹5 → ₹10) until filled."""
    return _execute_with_escalation(
        groww, trading_symbol, groww.TRANSACTION_TYPE_SELL, quantity, "SELL", f"SELL {label}"
    )


def buy_to_cover_option(groww: GrowwAPI, trading_symbol: str, quantity: int, label: str) -> tuple[str, float | None]:
    """BUY to cover a short, escalating the LIMIT price offset (₹2 → ₹5 → ₹10) until filled."""
    return _execute_with_escalation(
        groww, trading_symbol, groww.TRANSACTION_TYPE_BUY, quantity, "BUY", f"BUY (cover) {label}"
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
            detail = groww.get_order_detail(
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
    entry_str   = f"₹{entry_price:.2f}" if entry_price else "unknown"
    log.warning(f"[{reason}] Covering short {leg_label}  entry={entry_str}  symbol={symbol}")

    order_id, fill_price = buy_to_cover_option(groww, symbol, quantity, leg_label)
    exit_fill_confirmed  = fill_price is not None
    exit_price           = fill_price if fill_price is not None else get_option_ltp(groww, symbol)

    if not exit_fill_confirmed:
        log.warning(
            f"    {leg_label} cover order {order_id} fill UNCONFIRMED — "
            f"using LTP ₹{exit_price:.2f} for logging only. "
            "VERIFY ACTUAL POSITION ON GROWW."
        )

    pnl = (entry_price - exit_price) * quantity if entry_price else None

    log_trade({
        "leg":            leg_label,
        "action":         "EXIT",
        "symbol":         symbol,
        "quantity":       quantity,
        "price":          exit_price,
        "fill_confirmed": exit_fill_confirmed,
        "order_id":       order_id,
        "entry_price":    entry_price or "",
        "exit_price":     exit_price,
        "pnl":            f"{pnl:.2f}" if pnl is not None else "",
        "reason":         reason,
    })

    if entry_price:
        log.info(
            f"    {leg_label} short covered.  "
            f"Entry ₹{entry_price:.2f}  Exit ₹{exit_price:.2f}  "
            f"P&L: ₹{pnl:.2f}"
        )
    else:
        log.info(
            f"    {leg_label} short covered.  Exit ₹{exit_price:.2f}  P&L: N/A"
        )

    return {
        "leg":                  leg_label,
        "symbol":               symbol,
        "quantity":             quantity,
        "entry_price":          entry_price,
        "exit_price":           exit_price,
        "exit_order_id":        order_id,
        "exit_fill_confirmed":  exit_fill_confirmed,
        "pnl":                  pnl,
        "reason":               reason,
    }


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

    instruments_df = get_all_instruments_df(groww)
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
                        square_off_position(
                            groww, put_pos, QUANTITY * put_pos["lot_size"],
                            "PUT SELL", reason="SQUARE-OFF TIME"
                        )
                    except Exception as e:
                        log.error(
                            f"Failed to square off PUT SELL position "
                            f"(symbol={put_pos['symbol']}): {e}. "
                            "MANUAL INTERVENTION REQUIRED — check open positions on Groww."
                        )
                    put_pos = None
                if call_pos is not None:
                    try:
                        square_off_position(
                            groww, call_pos, QUANTITY * call_pos["lot_size"],
                            "CALL SELL", reason="SQUARE-OFF TIME"
                        )
                    except Exception as e:
                        log.error(
                            f"Failed to square off CALL SELL position "
                            f"(symbol={call_pos['symbol']}): {e}. "
                            "MANUAL INTERVENTION REQUIRED — check open positions on Groww."
                        )
                    call_pos = None
                if not had_positions:
                    log.info(f"[{now_ts}] SQUARE-OFF TIME — no open positions.")
                log.info("Session ended. Exiting strategy.")
                break

            # ── Fetch 1-min candles for BANKNIFTY index ───────────────────────
            df = fetch_1min_candles(groww, fut_groww_symbol)
            if len(df) < ST_LENGTH + 5:
                log.warning(
                    f"[{now_ts}] Only {len(df)} bars available "
                    f"(need {ST_LENGTH + 5}). Retrying in 60 s…"
                )
                time.sleep(60)
                continue

            # Diagnostic: verify the -2 / -1 assumption below. The code assumes
            # df.iloc[-1] is a still-forming/partial candle and df.iloc[-2] is
            # the last fully-closed candle (this matches behaviour of the
            # deprecated get_historical_candle_data). If the last candle's
            # timestamp is already >1 min old relative to now_ist, the API may
            # be returning only fully-closed candles, in which case df.iloc[-1]
            # (not -2) would be the correct "last completed candle" and the
            # signal below would be running one minute stale. Verify once at
            # startup and adjust compute_supertrend_direction()/last_close
            # (both use index -2) if needed.
            if prev_direction is None:
                last_ts = df.iloc[-1]["ts"]
                log.info(
                    f"[DIAG] Last candle ts={last_ts}  now={now_ist.strftime('%Y-%m-%d %H:%M:%S')}  "
                    f"— if last_ts is >1 min behind 'now', df.iloc[-1] is likely the "
                    f"last completed candle and indices below should be -1, not -2."
                )

            # ── Supertrend direction ──────────────────────────────────────────
            curr_direction = compute_supertrend_direction(df, ST_LENGTH, ST_FACTOR)
            last_close     = float(df.iloc[-2]["close"])
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
                            log.info(f"Recovered {leg_name} entry price (confirmed fill): ₹{confirmed:.2f}")
                            log_trade({
                                "leg": leg_name, "action": "ENTRY_RECOVERED",
                                "symbol": leg_pos["symbol"], "price": confirmed,
                                "fill_confirmed": True, "order_id": leg_pos["order_id"],
                                "entry_price": confirmed, "reason": "late fill confirmed",
                            })
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
                        f"ATM strike={atm}  →  Selling PUT"
                    )
                    # Refresh instruments to get current ATM put symbol
                    instruments_df = get_all_instruments_df(groww)
                    expiry         = get_monthly_expiry_date(instruments_df)
                    put_sym, put_lot = find_option_symbol(
                        instruments_df, expiry, atm, "PE"
                    )
                    trade_qty = QUANTITY * put_lot
                    order_id, fill_price = sell_option(groww, put_sym, trade_qty, "PUT")
                    put_pos   = {
                        "symbol":      put_sym,
                        "lot_size":    put_lot,
                        "entry_price": fill_price,   # None if fill unconfirmed
                        "order_id":    order_id,
                    }
                    if fill_price is not None:
                        log.info(
                            f"PUT SELL opened.  Symbol={put_sym}  "
                            f"Qty={trade_qty}  Entry ₹{fill_price:.2f}"
                        )
                        log_trade({
                            "leg": "PUT", "action": "ENTRY", "symbol": put_sym,
                            "quantity": trade_qty, "price": fill_price,
                            "fill_confirmed": True, "order_id": order_id,
                            "entry_price": fill_price, "reason": "ST flip BULLISH",
                        })
                    else:
                        log.warning(
                            f"PUT SELL: order {order_id} fill UNCONFIRMED. "
                            "Entry price will be recovered on next iteration. "
                            "VERIFY ACTUAL POSITION ON GROWW."
                        )
                        log_trade({
                            "leg": "PUT", "action": "ENTRY", "symbol": put_sym,
                            "quantity": trade_qty, "price": "",
                            "fill_confirmed": False, "order_id": order_id,
                            "reason": "ST flip BULLISH (fill unconfirmed)",
                        })

            # Exit: ST turned BEARISH AND P/L threshold reached
            elif curr_direction == -1 and put_pos is not None:
                if put_pos.get("entry_price") is None:
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
                        square_off_position(
                            groww, put_pos, trade_qty, "PUT SELL",
                            reason="ST flip BEARISH + threshold"
                        )
                        put_pos   = None
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
                        f"ATM strike={atm}  →  Selling CALL"
                    )
                    instruments_df = get_all_instruments_df(groww)
                    expiry         = get_monthly_expiry_date(instruments_df)
                    call_sym, call_lot = find_option_symbol(
                        instruments_df, expiry, atm, "CE"
                    )
                    trade_qty = QUANTITY * call_lot
                    order_id, fill_price = sell_option(groww, call_sym, trade_qty, "CALL")
                    call_pos  = {
                        "symbol":      call_sym,
                        "lot_size":    call_lot,
                        "entry_price": fill_price,   # None if fill unconfirmed
                        "order_id":    order_id,
                    }
                    if fill_price is not None:
                        log.info(
                            f"CALL SELL opened.  Symbol={call_sym}  "
                            f"Qty={trade_qty}  Entry ₹{fill_price:.2f}"
                        )
                        log_trade({
                            "leg": "CALL", "action": "ENTRY", "symbol": call_sym,
                            "quantity": trade_qty, "price": fill_price,
                            "fill_confirmed": True, "order_id": order_id,
                            "entry_price": fill_price, "reason": "ST flip BEARISH",
                        })
                    else:
                        log.warning(
                            f"CALL SELL: order {order_id} fill UNCONFIRMED. "
                            "Entry price will be recovered on next iteration. "
                            "VERIFY ACTUAL POSITION ON GROWW."
                        )
                        log_trade({
                            "leg": "CALL", "action": "ENTRY", "symbol": call_sym,
                            "quantity": trade_qty, "price": "",
                            "fill_confirmed": False, "order_id": order_id,
                            "reason": "ST flip BEARISH (fill unconfirmed)",
                        })

            # Exit: ST turned BULLISH AND P/L threshold reached
            elif curr_direction == 1 and call_pos is not None:
                if call_pos.get("entry_price") is None:
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
                        square_off_position(
                            groww, call_pos, trade_qty, "CALL SELL",
                            reason="ST flip BULLISH + threshold"
                        )
                        call_pos  = None
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
                    square_off_position(
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
                    square_off_position(
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
            if "forbidden" in str(exc).lower():
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
            else:
                log.error(f"Unhandled error: {exc}", exc_info=True)
                log.info("Sleeping 60 s before retrying…")
                time.sleep(60)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_strategy()
