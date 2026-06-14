"""
Gold Mini Intraday Supertrend Strategy — Groww API
===================================================
Instrument : GOLDM (MCX Gold Mini)
Quantity   : 1 lot (100g)
Entry      : BUY when Supertrend (length=20, factor=2) on 1-min chart turns BULLISH
             (1st trade: executes immediately if Supertrend is already bullish at startup)
             No entries allowed in the first 2 minutes after market open
Exit       : SELL when Supertrend turns BEARISH  AND  |P&L| >= ₹220 per unit
Auto S/O   : All open positions are force-closed at SQUARE_OFF_TIME (23:05),
             10 min before broker auto square-off at 23:15
"""

import time
import logging
import pyotp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from growwapi import GrowwAPI


# ─── USER CONFIGURATION ──────────────────────────────────────────────────────
TOTP_API_KEY = "TOTP_API_KEY"   # Replace with your Groww TOTP token
TOTP_SECRET  = "TOTP_SECRET"    # Replace with your Groww TOTP secret

# Supertrend indicator settings
ST_LENGTH = 20       # ATR period
ST_FACTOR = 2.0      # ATR multiplier

# Trade settings
QUANTITY           = 1     # Gold Mini lots per trade (1 lot = 100g)
EXIT_PNL_THRESHOLD = 220   # Minimum |P&L| per unit (₹) required to allow exit on bearish flip

# Instrument
SYMBOL_PREFIX = "GOLDM"

# MCX market hours (IST)  — commodity session 09:00 to 23:30
MARKET_OPEN      = "09:00"
MARKET_CLOSE     = "23:30"
ENTRY_START_TIME = "09:02"  # No new entries until 2 min after market open
SQUARE_OFF_TIME  = "23:05"  # Force-close 25 min before market close (23:30); 10 min before Groww MIS auto square-off (~23:15)

# How many 1-min historical bars to fetch (must be > ST_LENGTH + a few extra)
LOOKBACK_BARS = 120

# IST timezone — used for all market-hours comparisons (safe for Groww Cloud)
IST = timezone(timedelta(hours=5, minutes=30))


# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─── AUTHENTICATION ──────────────────────────────────────────────────────────
def authenticate() -> GrowwAPI:
    """Authenticate using TOTP with retry logic for transient network timeouts."""
    max_retries = 5
    retry_delay = 30  # seconds — aligns with TOTP 30s rotation window
    sanitized_secret = TOTP_SECRET.replace(" ", "").replace("-", "").upper()

    for attempt in range(1, max_retries + 1):
        try:
            totp = pyotp.TOTP(sanitized_secret).now()  # Fresh code on every attempt
            access_token = GrowwAPI.get_access_token(api_key=TOTP_API_KEY, totp=totp)
            log.info(f"Authenticated with Groww API (TOTP, attempt {attempt}).")
            return GrowwAPI(access_token)
        except Exception as e:
            log.error(f"Authentication attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                log.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    raise RuntimeError(f"Authentication failed after {max_retries} attempts — check network connectivity to api.groww.in")


# ─── INSTRUMENT LOOKUP ───────────────────────────────────────────────────────
def get_active_goldmini_symbol(groww: GrowwAPI) -> str:
    """
    Scan the instruments CSV and return the trading symbol of the nearest-expiry
    GOLDM (Gold Mini) contract that is available for trading on MCX today.
    """
    df = groww.get_all_instruments()
    today = pd.Timestamp(datetime.now(IST).date())

    mask = (
        (df["exchange"] == "MCX")
        & df["trading_symbol"].str.startswith(SYMBOL_PREFIX, na=False)
        & (df["instrument_type"].str.upper() == "FUT")
    )
    filtered = df[mask].copy()
    filtered["expiry_date"] = pd.to_datetime(filtered["expiry_date"], errors="coerce")
    active = filtered[filtered["expiry_date"] >= today].sort_values("expiry_date")

    if active.empty:
        raise RuntimeError(
            f"No active {SYMBOL_PREFIX} contract found on MCX. "
            "Check the instruments CSV or market calendar."
        )

    row = active.iloc[0]
    log.info(
        f"Selected instrument: {row['trading_symbol']}  "
        f"(expiry: {row['expiry_date'].date()})"
    )
    return str(row["trading_symbol"])


# ─── MARKET DATA — HISTORICAL CANDLES ────────────────────────────────────────
def fetch_1min_candles(groww: GrowwAPI, trading_symbol: str) -> pd.DataFrame:
    """
    Fetch the most recent LOOKBACK_BARS 1-minute candles for *trading_symbol*
    from MCX COMMODITY segment.

    Returns a DataFrame with columns: ts, open, high, low, close, volume
    sorted ascending by timestamp.
    """
    end_dt   = datetime.now(IST).replace(tzinfo=None)  # naive IST for API string format
    start_dt = end_dt - timedelta(minutes=LOOKBACK_BARS + 30)

    response = groww.get_historical_candle_data(
        trading_symbol=trading_symbol,
        exchange=groww.EXCHANGE_MCX,
        segment=groww.SEGMENT_COMMODITY,
        start_time=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        interval_in_minutes=1,
    )

    candles = response.get("candles", [])
    if not candles:
        raise RuntimeError(
            f"No candle data returned for {trading_symbol}. "
            "Market may be closed or instrument is illiquid."
        )

    df = pd.DataFrame(
        candles,
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    df["ts"] = (
        pd.to_datetime(df["ts"], unit="s")
          .dt.tz_localize("UTC")
          .dt.tz_convert("Asia/Kolkata")
          .dt.tz_localize(None)      # drop tzinfo for clean display
    )
    df = df.sort_values("ts").reset_index(drop=True)
    return df


# ─── MARKET DATA — LAST TRADED PRICE ─────────────────────────────────────────
def get_ltp(groww: GrowwAPI, trading_symbol: str) -> float:
    """
    Fetch the last traded price for *trading_symbol* on MCX COMMODITY.
    The API returns {exchange_trading_symbol: ltp_value}.
    """
    key = f"MCX_{trading_symbol}"
    response = groww.get_ltp(
        segment=groww.SEGMENT_COMMODITY,
        exchange_trading_symbols=key,
    )
    ltp = response.get(key)
    if ltp is None:
        raise RuntimeError(
            f"LTP not found in response for {key}. Response: {response}"
        )
    return float(ltp)


# ─── SUPERTREND INDICATOR ─────────────────────────────────────────────────────
def compute_supertrend_direction(
    df: pd.DataFrame,
    length: int,
    factor: float,
) -> int:
    """
    Compute Supertrend on *df* using Wilder's ATR.

    Parameters
    ----------
    df     : DataFrame with columns open/high/low/close (sorted ascending).
    length : ATR period.
    factor : ATR multiplier.

    Returns
    -------
    int
        Direction of the LAST COMPLETED candle (second-to-last row):
        +1 = bullish, -1 = bearish
    """
    n     = len(df)
    high  = df["high"].to_numpy(dtype=np.float64)
    low   = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    # ── True Range ──────────────────────────────────────────────────────────
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )

    # ── Wilder's ATR: seed with simple average, then smooth ─────────────────
    atr = np.empty(n)
    atr[: length - 1] = np.nan
    atr[length - 1]   = np.mean(tr[:length])
    for i in range(length, n):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length

    # ── Basic bands ─────────────────────────────────────────────────────────
    hl2       = (high + low) / 2.0
    basic_ub  = hl2 + factor * atr    # basic upper band
    basic_lb  = hl2 - factor * atr    # basic lower band

    # ── Final (adjusted) bands & direction ──────────────────────────────────
    final_ub  = np.full(n, np.nan)
    final_lb  = np.full(n, np.nan)
    direction = np.ones(n, dtype=np.int8)     # 1 = bullish, -1 = bearish

    for i in range(length - 1, n):
        if i == length - 1:
            # Initialise at first valid ATR bar
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
            direction[i] = 1
            continue

        # Upper band: tighten only when price was below previous upper band
        if basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        # Lower band: raise only when price was above previous lower band
        if basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        # Previous direction drives the current signal
        if direction[i - 1] == 1:
            # Was bullish: stay bullish unless close drops below lower band
            direction[i] = -1 if close[i] < final_lb[i] else 1
        else:
            # Was bearish: turn bullish if close rises above upper band
            direction[i] = 1 if close[i] > final_ub[i] else -1

    # Use the second-to-last row as the "last confirmed / closed" candle
    # (-1 is still forming / potentially incomplete)
    return int(direction[-2])


# ─── ORDER HELPERS ────────────────────────────────────────────────────────────
def place_buy_order(groww: GrowwAPI, trading_symbol: str) -> dict:
    log.info(f">>> Placing MARKET BUY  {QUANTITY} x {trading_symbol}")
    resp = groww.place_order(
        trading_symbol=trading_symbol,
        quantity=QUANTITY,
        validity=groww.VALIDITY_DAY,
        exchange=groww.EXCHANGE_MCX,
        segment=groww.SEGMENT_COMMODITY,
        product=groww.PRODUCT_MIS,
        order_type=groww.ORDER_TYPE_MARKET,
        transaction_type=groww.TRANSACTION_TYPE_BUY,
    )
    log.info(f"    BUY response  : {resp}")
    return resp


def place_sell_order(groww: GrowwAPI, trading_symbol: str) -> dict:
    log.info(f">>> Placing MARKET SELL {QUANTITY} x {trading_symbol}")
    resp = groww.place_order(
        trading_symbol=trading_symbol,
        quantity=QUANTITY,
        validity=groww.VALIDITY_DAY,
        exchange=groww.EXCHANGE_MCX,
        segment=groww.SEGMENT_COMMODITY,
        product=groww.PRODUCT_MIS,
        order_type=groww.ORDER_TYPE_MARKET,
        transaction_type=groww.TRANSACTION_TYPE_SELL,
    )
    log.info(f"    SELL response : {resp}")
    return resp


# ─── ORDER FILL HELPER ───────────────────────────────────────────────────────
def _await_fill_price(groww: GrowwAPI, order_id: str, retries: int = 5) -> float | None:
    """
    Poll order status until COMPLETE (up to retries × 0.5 s = 2.5 s by default).
    Returns average_fill_price if the order filled, or None so the caller can
    fall back to LTP.
    """
    for _ in range(retries):
        try:
            time.sleep(0.5)
            detail = groww.get_order_status(
                groww_order_id=order_id,
                segment=groww.SEGMENT_COMMODITY,
            )
            if detail.get("order_status") == "COMPLETE":
                price = float(detail.get("average_fill_price") or 0)
                if price:
                    return price
                # COMPLETE but price not yet populated — continue retrying
        except Exception as e:
            log.warning(f"Error polling order {order_id}: {e}")
    log.warning(
        f"Order {order_id} not COMPLETE after {retries} polls — using LTP as fallback."
    )
    return None


# ─── MARKET STATUS ────────────────────────────────────────────────────────────
def market_status(now_hm: str) -> str:
    """
    Returns one of:
      'CLOSED'     — outside market hours, do nothing
      'SQUAREOFF'  — approaching session end, must close open positions
      'OPEN'       — normal trading
    """
    if now_hm >= MARKET_CLOSE:
        return "CLOSED"
    if now_hm >= SQUARE_OFF_TIME:
        return "SQUAREOFF"
    if now_hm >= MARKET_OPEN:
        return "OPEN"
    return "CLOSED"


# ─── STRATEGY LOOP ────────────────────────────────────────────────────────────
def run_strategy() -> None:
    """Main event loop: poll every ~60 seconds and act on Supertrend signals."""
    groww  = authenticate()
    symbol = get_active_goldmini_symbol(groww)

    # State
    position: dict | None = None    # None  → flat;  dict with entry details → long
    prev_direction: int | None = None   # Supertrend direction from last iteration

    log.info(
        f"Strategy initialised for {symbol}. "
        f"Supertrend({ST_LENGTH},{ST_FACTOR}) | Qty={QUANTITY} | "
        f"Exit threshold=₹{EXIT_PNL_THRESHOLD}/unit"
    )
    log.info(
        f"Timing: entries allowed from {ENTRY_START_TIME}  |  "
        f"force close at {SQUARE_OFF_TIME}"
    )
    log.info("Waiting for market to open…")

    while True:
        try:
            now_ist_dt = datetime.now(IST)
            now_ts = now_ist_dt.strftime("%Y-%m-%d %H:%M:%S")
            now_hm = now_ist_dt.strftime("%H:%M")
            status = market_status(now_hm)

            # ── Outside market hours ──────────────────────────────────────────
            if status == "CLOSED":
                log.info(f"[{now_ts}] Market CLOSED. Sleeping 60 s…")
                time.sleep(60)
                continue

            # ── Auto square-off near session end ─────────────────────────────
            if status == "SQUAREOFF":
                if position is not None:
                    log.warning(
                        f"[{now_ts}] SQUARE-OFF TIME reached — closing open position."
                    )
                    sell_resp   = place_sell_order(groww, symbol)
                    sell_id     = sell_resp.get("groww_order_id", "")
                    saved_entry = position["entry_price"]
                    position    = None                          # clear before fill fetch
                    exit_price  = _await_fill_price(groww, sell_id) or get_ltp(groww, symbol)
                    pnl = (exit_price - saved_entry) * QUANTITY
                    log.info(
                        f"    Entry: ₹{saved_entry:.2f}  "
                        f"Exit ₹{exit_price:.2f}  "
                        f"P&L: ₹{pnl:.2f}"
                    )
                else:
                    log.info(f"[{now_ts}] SQUARE-OFF TIME — no open position.")
                log.info("Session ended. Exiting strategy.")
                break

            # ── Fetch candles ─────────────────────────────────────────────────
            df = fetch_1min_candles(groww, symbol)
            if len(df) < ST_LENGTH + 5:
                log.warning(
                    f"[{now_ts}] Only {len(df)} bars available "
                    f"(need {ST_LENGTH + 5}). Retrying in 60 s…"
                )
                time.sleep(60)
                continue

            # ── Compute Supertrend on last confirmed candle ───────────────────
            curr_direction = compute_supertrend_direction(df, ST_LENGTH, ST_FACTOR)
            last_close     = float(df.iloc[-2]["close"])
            trend_label    = "BULLISH ▲" if curr_direction == 1 else "BEARISH ▼"

            pos_label = (
                f"LONG @ ₹{position['entry_price']:.2f}"
                if position else "FLAT"
            )
            log.info(
                f"[{now_ts}]  {trend_label}  |  Close: ₹{last_close:.2f}  |  {pos_label}"
            )

            # ── First iteration: initialise prev_direction ────────────────────
            if prev_direction is None:
                if curr_direction == -1:
                    # Supertrend already bearish on start — observe, no trade
                    prev_direction = curr_direction
                    elapsed = (datetime.now(IST) - now_ist_dt).total_seconds()
                    time.sleep(max(0, 60 - elapsed))
                    continue
                else:
                    # Supertrend already bullish on start — allow immediate
                    # first entry by treating it as if it just flipped bullish
                    prev_direction = -1
                    # fall through to entry logic below

            # ── ENTRY: Supertrend just flipped to bullish ─────────────────────
            if position is None and curr_direction == 1 and prev_direction == -1:
                if now_hm < ENTRY_START_TIME:
                    log.info(
                        f"SIGNAL  Supertrend turned BULLISH but entry blocked "
                        f"(market < {ENTRY_START_TIME} warm-up window). Skipping."
                    )
                else:
                    log.info("SIGNAL  Supertrend turned BULLISH → entering LONG")
                    resp     = place_buy_order(groww, symbol)
                    order_id = resp.get("groww_order_id", "")
                    position = {"entry_price": 0.0, "order_id": order_id}  # set before fill fetch
                    entry_price = _await_fill_price(groww, order_id) or get_ltp(groww, symbol)
                    position["entry_price"] = entry_price
                    log.info(f"Position OPENED at ₹{entry_price:.2f}")

            # ── EXIT: Supertrend flipped bearish AND |P&L| >= threshold ───────
            elif position is not None and curr_direction == -1 and prev_direction == 1:
                ltp          = get_ltp(groww, symbol)
                pnl_per_unit = ltp - position["entry_price"]

                if abs(pnl_per_unit) >= EXIT_PNL_THRESHOLD:
                    log.info(
                        f"SIGNAL  Supertrend turned BEARISH  "
                        f"| P&L = ₹{pnl_per_unit:+.2f}/unit  "
                        f"({abs(pnl_per_unit):.2f} >= {EXIT_PNL_THRESHOLD})  "
                        f"→ SELL"
                    )
                    sell_resp   = place_sell_order(groww, symbol)
                    sell_id     = sell_resp.get("groww_order_id", "")
                    saved_entry = position["entry_price"]
                    position    = None              # clear before fill fetch
                    exit_price  = _await_fill_price(groww, sell_id) or ltp
                    total_pnl   = (exit_price - saved_entry) * QUANTITY
                    log.info(
                        f"Position CLOSED.  "
                        f"Entry ₹{saved_entry:.2f}  "
                        f"Exit ₹{exit_price:.2f}  "
                        f"Total P&L: ₹{total_pnl:.2f}"
                    )
                else:
                    log.info(
                        f"Supertrend turned BEARISH but |P&L| = ₹{abs(pnl_per_unit):.2f} "
                        f"< ₹{EXIT_PNL_THRESHOLD} threshold — HOLDING position."
                    )
                    # Do NOT update prev_direction here — keeps exit re-triggering
                    # each iteration while Supertrend stays bearish.
                    elapsed = (datetime.now(IST) - now_ist_dt).total_seconds()
                    time.sleep(max(0, 60 - elapsed))
                    continue

            prev_direction = curr_direction
            elapsed = (datetime.now(IST) - now_ist_dt).total_seconds()
            time.sleep(max(0, 60 - elapsed))   # align to ~1-minute boundary

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — squaring off any open position…")
            if position is not None:
                place_sell_order(groww, symbol)
            log.info("Strategy stopped.")
            break

        except Exception as exc:
            log.error(f"Unhandled error: {exc}", exc_info=True)
            log.info("Sleeping 60 s before retrying…")
            time.sleep(60)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_strategy()
