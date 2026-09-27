"""
Nifty Futures Intraday Supertrend Long + Short Strategy — Groww API
====================================================================
Instrument : Nearest-expiry NIFTY Futures (NSE FNO)
Quantity   : 1 lot per direction
Supertrend : length=20, factor=1.5  on 1-minute candles

LONG  logic
  Entry  : BUY  when Supertrend flips BULLISH (bearish → bullish)
           (If Supertrend is already bullish at startup, enters immediately)
  Exit   : SELL when Supertrend is BEARISH  AND  |LTP − buy_price| >= 30 points

SHORT logic
  Entry  : SELL when Supertrend flips BEARISH (bullish → bearish)
           (If Supertrend is already bearish at startup, enters immediately)
  Exit   : BUY  when Supertrend is BULLISH  AND  |LTP − sell_price| >= 30 points

Auto S/O   : All open positions force-closed at SQUARE_OFF_TIME (15:15 IST),
             ~10 min before broker auto square-off at ~15:25.
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
ST_LENGTH = 20     # ATR period
ST_FACTOR = 1.5    # ATR multiplier

# Trade settings
QUANTITY             = 1    # NIFTY futures lots per direction
EXIT_POINTS_THRESHOLD = 30  # Exit when |LTP − entry| >= this many index points

# Instrument (nearest-expiry NIFTY FUT on NSE)
SYMBOL_PREFIX    = "NIFTY"
UNDERLYING       = "NIFTY"

# NSE FNO market hours (IST)
MARKET_OPEN      = "09:15"
MARKET_CLOSE     = "15:30"
ENTRY_START_TIME = "09:17"  # No new entries in first 2 min after market open
SQUARE_OFF_TIME  = "15:15"  # Force-close 15 min before market close

# How many 1-min historical bars to fetch  (must be > ST_LENGTH + a few extra)
LOOKBACK_BARS = 120

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))


# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ─── AUTHENTICATION ──────────────────────────────────────────────────────────
def authenticate() -> GrowwAPI:
    """Authenticate via TOTP with retry logic for transient failures."""
    max_retries = 5
    retry_delay = 30  # seconds — aligns with TOTP 30-second window rotation

    sanitized_secret = TOTP_SECRET.replace(" ", "").replace("-", "").upper()

    for attempt in range(1, max_retries + 1):
        try:
            totp = pyotp.TOTP(sanitized_secret).now()
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


# ─── INSTRUMENT LOOKUP ───────────────────────────────────────────────────────
def get_active_nifty_fut_symbol(groww: GrowwAPI) -> tuple[str, int]:
    """
    Scan the instruments CSV and return:
      (trading_symbol, lot_size)
    for the nearest-expiry NIFTY futures contract tradeable on NSE today.
    """
    df = groww.get_all_instruments()
    today = pd.Timestamp(datetime.now(IST).date())

    mask = (
        (df["exchange"] == "NSE")
        & (df["underlying_symbol"] == UNDERLYING)
        & (df["instrument_type"].str.upper() == "FUT")
        & (df["segment"].str.upper() == "FNO")
    )
    filtered = df[mask].copy()
    filtered["expiry_date"] = pd.to_datetime(filtered["expiry_date"], errors="coerce")
    active = filtered[filtered["expiry_date"] >= today].sort_values("expiry_date")

    if active.empty:
        raise RuntimeError(
            f"No active NIFTY FUT contract found on NSE. "
            "Check the instruments CSV or market calendar."
        )

    row = active.iloc[0]
    lot_size = int(row.get("lot_size", 75))
    log.info(
        f"Selected instrument : {row['trading_symbol']}  "
        f"(expiry: {row['expiry_date'].date()}  lot_size: {lot_size})"
    )
    return str(row["trading_symbol"]), lot_size


# ─── MARKET DATA — HISTORICAL CANDLES ────────────────────────────────────────
def fetch_1min_candles(groww: GrowwAPI, trading_symbol: str) -> pd.DataFrame:
    """
    Fetch the most recent LOOKBACK_BARS 1-minute candles for *trading_symbol*
    on NSE FNO.

    Returns a DataFrame with columns: ts, open, high, low, close, volume
    sorted ascending by timestamp.
    """
    end_dt   = datetime.now(IST).replace(tzinfo=None)
    start_dt = end_dt - timedelta(minutes=LOOKBACK_BARS + 30)

    response = groww.get_historical_candle_data(
        trading_symbol=trading_symbol,
        exchange=groww.EXCHANGE_NSE,
        segment=groww.SEGMENT_FNO,
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

    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = (
        pd.to_datetime(df["ts"], unit="s")
          .dt.tz_localize("UTC")
          .dt.tz_convert("Asia/Kolkata")
          .dt.tz_localize(None)
    )
    df = df.sort_values("ts").reset_index(drop=True)
    return df


# ─── MARKET DATA — LAST TRADED PRICE ─────────────────────────────────────────
def get_ltp(groww: GrowwAPI, trading_symbol: str) -> float:
    """
    Fetch the last traded price for *trading_symbol* on NSE FNO.
    """
    key = f"NSE_{trading_symbol}"
    response = groww.get_ltp(
        segment=groww.SEGMENT_FNO,
        exchange_trading_symbols=key,
    )
    ltp = response.get(key)
    if ltp is None:
        raise RuntimeError(
            f"LTP not found in response for {key}. Response: {response}"
        )
    return float(ltp)


# ─── SUPERTREND INDICATOR ─────────────────────────────────────────────────────
def compute_supertrend_direction(df: pd.DataFrame, length: int, factor: float) -> int:
    """
    Compute Supertrend using Wilder's ATR on *df*.

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
    hl2      = (high + low) / 2.0
    basic_ub = hl2 + factor * atr
    basic_lb = hl2 - factor * atr

    # ── Final bands & direction ──────────────────────────────────────────────
    final_ub  = np.full(n, np.nan)
    final_lb  = np.full(n, np.nan)
    direction = np.ones(n, dtype=np.int8)  # 1 = bullish, -1 = bearish

    for i in range(length - 1, n):
        if i == length - 1:
            final_ub[i]  = basic_ub[i]
            final_lb[i]  = basic_lb[i]
            direction[i] = 1
            continue

        # Upper band: tighten only when previous close was below previous upper band
        final_ub[i] = (
            basic_ub[i]
            if basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]
            else final_ub[i - 1]
        )
        # Lower band: raise only when previous close was above previous lower band
        final_lb[i] = (
            basic_lb[i]
            if basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]
            else final_lb[i - 1]
        )

        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < final_lb[i] else 1
        else:
            direction[i] = 1 if close[i] > final_ub[i] else -1

    # Return direction of the last COMPLETED (confirmed) candle
    return int(direction[-2])


# ─── ORDER HELPERS ────────────────────────────────────────────────────────────
def _place_order(
    groww: GrowwAPI,
    trading_symbol: str,
    transaction_type,
    quantity: int,
    label: str,
) -> dict:
    log.info(f">>> Placing MARKET {label}  {quantity} x {trading_symbol}")
    resp = groww.place_order(
        trading_symbol=trading_symbol,
        quantity=quantity,
        validity=groww.VALIDITY_DAY,
        exchange=groww.EXCHANGE_NSE,
        segment=groww.SEGMENT_FNO,
        product=groww.PRODUCT_MIS,
        order_type=groww.ORDER_TYPE_MARKET,
        transaction_type=transaction_type,
    )
    log.info(f"    {label} response : {resp}")
    return resp


def place_buy_order(groww: GrowwAPI, trading_symbol: str, quantity: int) -> dict:
    return _place_order(groww, trading_symbol, groww.TRANSACTION_TYPE_BUY, quantity, "BUY")


def place_sell_order(groww: GrowwAPI, trading_symbol: str, quantity: int) -> dict:
    return _place_order(groww, trading_symbol, groww.TRANSACTION_TYPE_SELL, quantity, "SELL")


# ─── ORDER FILL HELPER ───────────────────────────────────────────────────────
def _await_fill_price(groww: GrowwAPI, order_id: str, retries: int = 6) -> float | None:
    """
    Poll order status until COMPLETE (up to retries × 0.5 s = 3 s by default).
    Returns average_fill_price if filled, or None so caller can fall back to LTP.
    """
    for _ in range(retries):
        try:
            time.sleep(0.5)
            detail = groww.get_order_status(
                groww_order_id=order_id,
                segment=groww.SEGMENT_FNO,
            )
            if detail.get("order_status") in ("COMPLETE", "EXECUTED"):
                price = float(detail.get("average_fill_price") or 0)
                if price:
                    return price
        except Exception as e:
            log.warning(f"Error polling order {order_id}: {e}")
    log.warning(
        f"Order {order_id} not COMPLETE after {retries} polls — using LTP as fallback."
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


# ─── SQUARE-OFF ALL POSITIONS ─────────────────────────────────────────────────
def square_off_all(
    groww: GrowwAPI,
    symbol: str,
    lot_size: int,
    long_pos: dict | None,
    short_pos: dict | None,
    reason: str = "force square-off",
) -> tuple[None, None]:
    """Close any open long and/or short position. Returns (None, None) for cleared state."""
    if long_pos is not None:
        log.warning(f"[{reason}] Closing LONG position @ entry ₹{long_pos['entry_price']:.2f}")
        resp       = place_sell_order(groww, symbol, QUANTITY * lot_size)
        order_id   = resp.get("groww_order_id", "")
        saved_entry = long_pos["entry_price"]
        exit_price = _await_fill_price(groww, order_id) or get_ltp(groww, symbol)
        pnl        = (exit_price - saved_entry) * QUANTITY * lot_size
        log.info(
            f"    LONG closed.  Entry ₹{saved_entry:.2f}  "
            f"Exit ₹{exit_price:.2f}  P&L: ₹{pnl:.2f}"
        )

    if short_pos is not None:
        log.warning(f"[{reason}] Closing SHORT position @ entry ₹{short_pos['entry_price']:.2f}")
        resp        = place_buy_order(groww, symbol, QUANTITY * lot_size)
        order_id    = resp.get("groww_order_id", "")
        saved_entry = short_pos["entry_price"]
        exit_price  = _await_fill_price(groww, order_id) or get_ltp(groww, symbol)
        pnl         = (saved_entry - exit_price) * QUANTITY * lot_size
        log.info(
            f"    SHORT closed.  Entry ₹{saved_entry:.2f}  "
            f"Exit ₹{exit_price:.2f}  P&L: ₹{pnl:.2f}"
        )

    return None, None


# ─── STRATEGY LOOP ────────────────────────────────────────────────────────────
def run_strategy() -> None:
    """
    Main event loop polling every ~60 seconds.

    State machine per direction:
      LONG  — entered on bullish ST flip; exited when ST is bearish AND
              |LTP − entry| >= EXIT_POINTS_THRESHOLD
      SHORT — entered on bearish ST flip; exited when ST is bullish AND
              |LTP − entry| >= EXIT_POINTS_THRESHOLD
    """
    groww    = authenticate()
    symbol, lot_size = get_active_nifty_fut_symbol(groww)

    trade_qty = QUANTITY * lot_size     # actual number of units per order

    # ── State ─────────────────────────────────────────────────────────────────
    long_pos:  dict | None = None   # None → flat;  dict → {"entry_price": float, "order_id": str}
    short_pos: dict | None = None
    prev_direction: int | None = None

    log.info(
        f"Strategy initialised for {symbol}  (lot_size={lot_size}, trade_qty={trade_qty})."
    )
    log.info(
        f"Supertrend({ST_LENGTH}, {ST_FACTOR})  |  "
        f"Exit threshold: ±{EXIT_POINTS_THRESHOLD} pts  |  "
        f"Entries from {ENTRY_START_TIME}  |  Force close at {SQUARE_OFF_TIME}"
    )
    log.info("Waiting for market to open…")

    while True:
        try:
            now_ist_dt  = datetime.now(IST)
            now_ts      = now_ist_dt.strftime("%Y-%m-%d %H:%M:%S")
            now_hm      = now_ist_dt.strftime("%H:%M")
            iter_start  = time.monotonic()
            status      = market_status(now_hm)

            # ── Outside market hours ──────────────────────────────────────────
            if status == "CLOSED":
                log.info(f"[{now_ts}] Market CLOSED. Sleeping 60 s…")
                time.sleep(60)
                continue

            # ── Auto square-off near session end ─────────────────────────────
            if status == "SQUAREOFF":
                if long_pos is not None or short_pos is not None:
                    long_pos, short_pos = square_off_all(
                        groww, symbol, lot_size, long_pos, short_pos,
                        reason="SQUARE-OFF TIME reached"
                    )
                else:
                    log.info(f"[{now_ts}] SQUARE-OFF TIME — no open positions.")
                log.info("Session ended. Exiting strategy.")
                break

            # ── Fetch recent 1-min candles ────────────────────────────────────
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

            long_label  = f"LONG @ ₹{long_pos['entry_price']:.2f}"  if long_pos  else "LONG: FLAT"
            short_label = f"SHORT @ ₹{short_pos['entry_price']:.2f}" if short_pos else "SHORT: FLAT"
            log.info(
                f"[{now_ts}]  ST={trend_label}  Close=₹{last_close:.2f}  "
                f"|  {long_label}  |  {short_label}"
            )

            # ── First iteration: seed prev_direction to trigger entry if needed ─
            if prev_direction is None:
                # Treat startup state as the opposite flip so entry logic fires
                prev_direction = -1 if curr_direction == 1 else 1
                log.info(
                    f"Initial Supertrend state: {trend_label}. "
                    "Seeding prev_direction to allow immediate entry if appropriate."
                )

            # ═══════════════════════════════════════════════════════════════════
            #  LONG-SIDE LOGIC
            # ═══════════════════════════════════════════════════════════════════

            # Entry: ST just flipped BULLISH
            if curr_direction == 1 and prev_direction == -1 and long_pos is None:
                if now_hm < ENTRY_START_TIME:
                    log.info(
                        f"ST flipped BULLISH but entry blocked "
                        f"(before warm-up end {ENTRY_START_TIME})."
                    )
                else:
                    log.info("SIGNAL  ST → BULLISH  →  Opening LONG")
                    resp     = place_buy_order(groww, symbol, trade_qty)
                    order_id = resp.get("groww_order_id", "")
                    long_pos = {"entry_price": 0.0, "order_id": order_id}
                    try:
                        entry = _await_fill_price(groww, order_id) or get_ltp(groww, symbol)
                        long_pos["entry_price"] = entry
                        log.info(f"LONG opened at ₹{entry:.2f}")
                    except Exception as e:
                        log.error(
                            f"Could not fetch LONG entry price ({e}). "
                            "Will retry fetching on next iteration."
                        )

            # Exit check: ST is BEARISH and |LTP − entry| >= threshold
            elif curr_direction == -1 and long_pos is not None:
                # Guard: entry_price==0.0 means fill price was not yet fetched
                # (LTP call failed after order). Retry fetching it first.
                if long_pos["entry_price"] == 0.0:
                    try:
                        order_id = long_pos["order_id"]
                        recovered = _await_fill_price(groww, order_id) or get_ltp(groww, symbol)
                        long_pos["entry_price"] = recovered
                        log.info(f"Recovered LONG entry price: ₹{recovered:.2f}")
                    except Exception as e:
                        log.warning(f"Still cannot fetch LONG entry price: {e}. Skipping exit check.")
                        prev_direction = curr_direction
                        elapsed = time.monotonic() - iter_start
                        time.sleep(max(0.0, 60.0 - elapsed))
                        continue
                ltp           = get_ltp(groww, symbol)
                diff          = ltp - long_pos["entry_price"]
                abs_diff      = abs(diff)
                direction_str = "above" if diff > 0 else "below"

                log.info(
                    f"    LONG exit check: LTP=₹{ltp:.2f}  "
                    f"Entry=₹{long_pos['entry_price']:.2f}  "
                    f"Diff={diff:+.2f} pts ({direction_str} entry)  "
                    f"Threshold={EXIT_POINTS_THRESHOLD} pts"
                )

                if abs_diff >= EXIT_POINTS_THRESHOLD:
                    log.info(
                        f"SIGNAL  ST BEARISH + |diff|={abs_diff:.2f} >= {EXIT_POINTS_THRESHOLD}  "
                        f"→  Closing LONG"
                    )
                    resp        = place_sell_order(groww, symbol, trade_qty)
                    order_id    = resp.get("groww_order_id", "")
                    saved_entry = long_pos["entry_price"]
                    long_pos    = None
                    exit_price  = _await_fill_price(groww, order_id) or ltp
                    total_pnl   = (exit_price - saved_entry) * trade_qty
                    log.info(
                        f"LONG closed.  Entry ₹{saved_entry:.2f}  "
                        f"Exit ₹{exit_price:.2f}  Total P&L: ₹{total_pnl:.2f}"
                    )
                else:
                    log.info(
                        f"    Holding LONG — |diff|={abs_diff:.2f} pts "
                        f"< {EXIT_POINTS_THRESHOLD} pts threshold."
                    )

            # ═══════════════════════════════════════════════════════════════════
            #  SHORT-SIDE LOGIC
            # ═══════════════════════════════════════════════════════════════════

            # Entry: ST just flipped BEARISH
            if curr_direction == -1 and prev_direction == 1 and short_pos is None:
                if now_hm < ENTRY_START_TIME:
                    log.info(
                        f"ST flipped BEARISH but entry blocked "
                        f"(before warm-up end {ENTRY_START_TIME})."
                    )
                else:
                    log.info("SIGNAL  ST → BEARISH  →  Opening SHORT")
                    resp      = place_sell_order(groww, symbol, trade_qty)
                    order_id  = resp.get("groww_order_id", "")
                    short_pos = {"entry_price": 0.0, "order_id": order_id}
                    try:
                        entry = _await_fill_price(groww, order_id) or get_ltp(groww, symbol)
                        short_pos["entry_price"] = entry
                        log.info(f"SHORT opened at ₹{entry:.2f}")
                    except Exception as e:
                        log.error(
                            f"Could not fetch SHORT entry price ({e}). "
                            "Will retry fetching on next iteration."
                        )

            # Exit check: ST is BULLISH and |LTP − entry| >= threshold
            elif curr_direction == 1 and short_pos is not None:
                # Guard: entry_price==0.0 means fill price was not yet fetched.
                if short_pos["entry_price"] == 0.0:
                    try:
                        order_id = short_pos["order_id"]
                        recovered = _await_fill_price(groww, order_id) or get_ltp(groww, symbol)
                        short_pos["entry_price"] = recovered
                        log.info(f"Recovered SHORT entry price: ₹{recovered:.2f}")
                    except Exception as e:
                        log.warning(f"Still cannot fetch SHORT entry price: {e}. Skipping exit check.")
                        prev_direction = curr_direction
                        elapsed = time.monotonic() - iter_start
                        time.sleep(max(0.0, 60.0 - elapsed))
                        continue
                ltp           = get_ltp(groww, symbol)
                diff          = ltp - short_pos["entry_price"]
                abs_diff      = abs(diff)
                direction_str = "above" if diff > 0 else "below"

                log.info(
                    f"    SHORT exit check: LTP=₹{ltp:.2f}  "
                    f"Entry=₹{short_pos['entry_price']:.2f}  "
                    f"Diff={diff:+.2f} pts ({direction_str} entry)  "
                    f"Threshold={EXIT_POINTS_THRESHOLD} pts"
                )

                if abs_diff >= EXIT_POINTS_THRESHOLD:
                    log.info(
                        f"SIGNAL  ST BULLISH + |diff|={abs_diff:.2f} >= {EXIT_POINTS_THRESHOLD}  "
                        f"→  Closing SHORT"
                    )
                    resp        = place_buy_order(groww, symbol, trade_qty)
                    order_id    = resp.get("groww_order_id", "")
                    saved_entry = short_pos["entry_price"]
                    short_pos   = None
                    exit_price  = _await_fill_price(groww, order_id) or ltp
                    total_pnl   = (saved_entry - exit_price) * trade_qty
                    log.info(
                        f"SHORT closed.  Entry ₹{saved_entry:.2f}  "
                        f"Exit ₹{exit_price:.2f}  Total P&L: ₹{total_pnl:.2f}"
                    )
                else:
                    log.info(
                        f"    Holding SHORT — |diff|={abs_diff:.2f} pts "
                        f"< {EXIT_POINTS_THRESHOLD} pts threshold."
                    )

            # ── Advance state and sleep until next minute boundary ────────────
            prev_direction = curr_direction
            elapsed = time.monotonic() - iter_start
            time.sleep(max(0.0, 60.0 - elapsed))

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — squaring off all open positions…")
            square_off_all(groww, symbol, lot_size, long_pos, short_pos, reason="KeyboardInterrupt")
            log.info("Strategy stopped.")
            break

        except Exception as exc:
            log.error(f"Unhandled error: {exc}", exc_info=True)
            log.info("Sleeping 60 s before retrying…")
            time.sleep(60)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_strategy()
