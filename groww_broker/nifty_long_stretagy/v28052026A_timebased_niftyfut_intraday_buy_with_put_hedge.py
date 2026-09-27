"""
Buy NIFTY FUT (1 lot, nearest expiry) + NIFTY ATM Put at 9:16-14:00. Exit all at 15:04.
Optional: Sell naked OTM Call (ATM+500) to collect premium and reduce net cost.

Strategy Legs:
- NIFTY FUT BUY: Profits if market goes up (replaces NIFTYBEES equity leg)
- ATM Put BUY:   Protects if market falls (insurance premium paid daily)
- OTM Call SELL (optional): Collects premium to offset put cost but caps upside at strike

All legs are held together — if any placement fails, entire position is rolled back.
Uses TOTP authentication with Groww API.
"""

# ===== IMPORTS =====
import time
import logging
import os
import pyotp
import pandas as pd
from datetime import datetime, time as dt_time, timedelta
from typing import Optional
from growwapi import GrowwAPI

# ===== LOGGING SETUP =====
_log_dir = os.path.dirname(os.path.abspath(__file__)) or "."
_log_file = os.path.join(_log_dir, f"strategy_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── USER CONFIGURATION ──────────────────────────────────────────────────────
TOTP_API_KEY = "TOTP_API_KEY"   # Replace with your Groww TOTP token
TOTP_SECRET  = "TOTP_SECRET"    # Replace with your Groww TOTP secret

# ===== CONFIGURATION =====
AUTH_CONFIG = {
    "api_key": TOTP_API_KEY,
    "totp_secret": TOTP_SECRET,
}

STRATEGY_CONFIG = {
    # Trading Parameters
    "nifty_underlying": "NIFTY",
    "nifty_fut_lots": 1,           # Number of NIFTY futures lots to buy (1 lot)
    # Lot size is resolved at runtime from the instrument master (typically 65 as per NSE)
    "nifty_option_quantity": 65,   # NIFTY lot size = 65 (1 lot) as per NSE latest update. Use multiples of 65 only.

    # Exchanges / Segments
    "option_exchange": "NSE",

    # Timing
    "entry_start_time": "09:16:00",  # Earliest time to enter (just after market open)
    "entry_cutoff_time": "14:00:00", # No entry after this time (leaves 1h+ for exit)
    "exit_time": "15:04:00",

    # Monitoring
    "check_interval_seconds": 30,
    "order_fill_timeout_seconds": 60,  # Seconds to wait for entry order to confirm as filled

    # Short call (offsets put theta decay — moderately bullish)
    "short_call_strike_offset": 500,   # Sell call this many points above ATM
    "nifty_call_quantity": 65,         # Call lot size (multiple of 65)

    # NIFTY F&O lot size is 65 (as per NSE latest update) — quantity must be a multiple of lot size.
    # Setting nifty_option_quantity to 1 causes order rejection.
    # Reference: instrument CSV column lot_size for NIFTY options/futures = 65
}


# ===== HELPER FUNCTIONS =====

def is_market_open(current_time: datetime) -> bool:
    """Check if NSE market is open (9:15 AM - 3:30 PM, weekdays)"""
    if current_time.weekday() >= 5:
        return False
    return dt_time(9, 15) <= current_time.time() <= dt_time(15, 30)


def authenticate_with_totp() -> GrowwAPI:
    """Authenticate using TOTP with retry logic for transient network timeouts."""
    max_retries = 5
    retry_delay = 30  # seconds — aligns with TOTP 30s rotation window

    raw_secret = AUTH_CONFIG["totp_secret"]
    sanitized_secret = raw_secret.replace(" ", "").replace("-", "").upper()

    for attempt in range(1, max_retries + 1):
        try:
            totp = pyotp.TOTP(sanitized_secret).now()  # Fresh code on every attempt
            access_token = GrowwAPI.get_access_token(
                api_key=AUTH_CONFIG["api_key"],
                totp=totp
            )
            logger.info(f"Authentication successful using TOTP (attempt {attempt})")
            return GrowwAPI(access_token)
        except Exception as e:
            logger.error(f"Authentication attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    raise RuntimeError(f"Authentication failed after {max_retries} attempts — check network connectivity to api.groww.in")


def get_nearest_weekly_expiry(groww: GrowwAPI, underlying: str = "NIFTY",
                               instruments_df=None) -> Optional[str]:
    """Get nearest weekly expiry date for NIFTY options.

    Accepts a pre-loaded instruments_df to avoid re-downloading the instrument
    master on every call. If not provided, fetches it (standalone use).
    Reference: https://groww.in/trade-api/docs/python-sdk/instruments
    """
    try:
        if instruments_df is None:
            instruments_df = groww.get_all_instruments()

        # Filter for the underlying's Put options in FNO segment
        nifty_options = instruments_df[
            (instruments_df['underlying_symbol'] == underlying) &
            (instruments_df['segment'] == 'FNO') &
            (instruments_df['instrument_type'] == 'PE')
        ]

        if nifty_options.empty:
            logger.warning(f"No FNO instruments found for {underlying}")
            return None

        today = datetime.now().date()
        raw_expiries = nifty_options['expiry_date'].dropna().unique()

        # Keep only FUTURE expiries — exclude today even if today is expiry day.
        # Buying 0-DTE options on expiry day provides almost no protection
        # and risks the premium expiring worthless within hours.
        future_expiries = []
        for exp in raw_expiries:
            exp_str = str(exp)[:10]  # Ensure YYYY-MM-DD format
            if datetime.strptime(exp_str, '%Y-%m-%d').date() > today:
                future_expiries.append(exp_str)

        if future_expiries:
            nearest_expiry = min(future_expiries, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
            logger.info(f"Nearest weekly expiry: {nearest_expiry}")
            return nearest_expiry

        return None

    except Exception as e:
        logger.error(f"Error fetching expiries: {e}")
        return None


def get_nearest_fut_instrument(instruments_df: pd.DataFrame, underlying: str = "NIFTY") -> Optional[dict]:
    """Get the nearest-expiry NIFTY futures instrument from a pre-loaded instruments DataFrame.

    Returns a dict with trading_symbol, exchange, segment, lot_size; or None on failure.
    """
    try:
        today = datetime.now().date()
        mask = (
            (instruments_df['underlying_symbol'] == underlying) &
            (instruments_df['instrument_type'].str.upper() == 'FUT') &
            (instruments_df['segment'].str.upper() == 'FNO') &
            (instruments_df['exchange'] == 'NSE')
        )
        fut_df = instruments_df[mask].copy()
        fut_df['expiry_date'] = pd.to_datetime(fut_df['expiry_date'], errors='coerce')
        # Exclude today's expiry — MIS orders on expiring futures contracts are often
        # blocked by the broker from ~3:20 PM onwards, and intraday entry at 9:16 on
        # expiry day carries rollover risk if exit is delayed.
        active = fut_df[fut_df['expiry_date'].dt.date > today].sort_values('expiry_date')

        if active.empty:
            logger.error(f"No active {underlying} FUT contract found on NSE")
            return None

        row = active.iloc[0]
        lot_size = int(row.get('lot_size', 65))
        logger.info(
            f"NIFTY FUT instrument: {row['trading_symbol']}  "
            f"(expiry: {row['expiry_date'].date()}  lot_size: {lot_size})"
        )
        return {
            'trading_symbol': str(row['trading_symbol']),
            'exchange': str(row['exchange']),
            'segment': str(row['segment']),
            'lot_size': lot_size,
        }
    except Exception as e:
        logger.error(f"Error fetching NIFTY FUT instrument: {e}")
        return None


def get_atm_strike(groww: GrowwAPI, underlying: str = "NIFTY",
                   fut_instrument: Optional[dict] = None) -> Optional[int]:
    """Get ATM (At-The-Money) strike price for NIFTY.

    Prefers the NIFTY futures LTP (more accurate for hedging) over the index spot price.
    Falls back to the index if fut_instrument is not provided.
    """
    try:
        if fut_instrument:
            exchange_symbol = f"{fut_instrument['exchange']}_{fut_instrument['trading_symbol']}"
            ltp_data = groww.get_ltp(
                segment=fut_instrument["segment"],
                exchange_trading_symbols=(exchange_symbol,)
            )
            current_price = ltp_data.get(exchange_symbol)
            label = f"NIFTY FUT ({fut_instrument['trading_symbol']})"
        else:
            index_instrument = groww.get_instrument_by_groww_symbol(groww_symbol=f"NSE-{underlying}")
            exchange_symbol = f"{index_instrument['exchange']}_{index_instrument['trading_symbol']}"
            ltp_data = groww.get_ltp(
                segment=index_instrument["segment"],
                exchange_trading_symbols=(exchange_symbol,)
            )
            current_price = ltp_data.get(exchange_symbol)
            label = f"NIFTY Index"

        if current_price is None:
            return None

        # Round to nearest 50 (NIFTY strike interval)
        atm_strike = round(current_price / 50) * 50
        logger.info(f"{label} current price: ₹{current_price:.2f} | ATM strike: {atm_strike}")
        return int(atm_strike)

    except Exception as e:
        logger.error(f"Error calculating ATM strike: {e}")
        return None


# ===== STRATEGY CLASS =====

class IntradayStrategy:
    """Intraday strategy: NIFTY FUT (1 lot) BUY + NIFTY ATM Put + optional naked short call."""

    def __init__(self, groww: GrowwAPI, config: dict):
        self.groww = groww
        self.config = config

        # State tracking
        self.positions = {}
        self.entry_executed = False
        self.exit_executed = False
        self._instruments_df = None  # Cached instrument master — loaded once, reused on retry
        self.entry_retry_count = 0   # Track number of entry attempts
        self.max_retries = 3         # Maximum retry attempts
        self.margin_shortfall_detected = False  # Flag to stop retries on margin shortfall

        # NIFTY FUT instrument and quantity resolved at runtime
        self.nifty_fut_instrument = None
        self.nifty_fut_quantity = None  # lots × lot_size

        # Option instruments will be fetched at entry time
        self.option_instrument = None
        self.short_call_instrument = None  # Short OTM call (sold to collect premium)

    def _load_instruments_df(self):
        """Load and cache instrument master. Fetched once at first call, reused on retry."""
        if self._instruments_df is None:
            logger.info("Loading instrument master from Groww...")
            self._instruments_df = self.groww.get_all_instruments()
            logger.info(f"Instrument master loaded: {len(self._instruments_df)} instruments")
        return self._instruments_df

    def _resolve_fut_instrument(self) -> bool:
        """Resolve and cache the nearest-expiry NIFTY FUT instrument + quantity."""
        if self.nifty_fut_instrument is not None:
            return True
        df = self._load_instruments_df()
        fut = get_nearest_fut_instrument(df, self.config["nifty_underlying"])
        if fut is None:
            return False
        self.nifty_fut_instrument = fut
        self.nifty_fut_quantity = self.config["nifty_fut_lots"] * fut["lot_size"]
        logger.info(
            f"NIFTY FUT resolved: {fut['trading_symbol']}  "
            f"({self.config['nifty_fut_lots']} lot × {fut['lot_size']} = {self.nifty_fut_quantity} qty)"
        )
        return True

    def get_option_instrument(self) -> bool:
        """Fetch NIFTY weekly ATM Put option instrument (and optional short call)."""
        try:
            logger.info("Fetching NIFTY option details...")

            # Get nearest weekly expiry — pass cached df to avoid re-downloading
            expiry_date = get_nearest_weekly_expiry(
                self.groww, self.config["nifty_underlying"],
                instruments_df=self._load_instruments_df()
            )
            if not expiry_date:
                logger.warning("Failed to get expiry date")
                return False

            # Get ATM strike using futures LTP (preferred) or index spot
            atm_strike = get_atm_strike(
                self.groww, self.config["nifty_underlying"],
                fut_instrument=self.nifty_fut_instrument
            )
            if not atm_strike:
                logger.warning("Failed to calculate ATM strike")
                return False

            # Fetch ATM Put instrument from instruments DataFrame
            df = self._load_instruments_df()
            nifty_puts = df[
                (df['underlying_symbol'] == self.config['nifty_underlying']) &
                (df['segment'] == 'FNO') &
                (df['instrument_type'] == 'PE') &
                (df['expiry_date'].apply(lambda x: str(x)[:10]) == expiry_date)
            ].copy()
            if nifty_puts.empty:
                logger.error(f"No PE instruments found for {self.config['nifty_underlying']} expiry {expiry_date}")
                logger.error(f"Available expiries in master: {sorted(df[df['underlying_symbol'] == self.config['nifty_underlying']]['expiry_date'].dropna().unique().tolist())[:10]}")
                return False
            nifty_puts['strike_num'] = pd.to_numeric(nifty_puts['strike_price'], errors='coerce')
            exact_rows = nifty_puts[nifty_puts['strike_num'] == atm_strike]
            if not exact_rows.empty:
                chosen = exact_rows.iloc[0]
                logger.info(f"ATM Put instrument loaded: {chosen['trading_symbol']} (strike={atm_strike})")
            else:
                chosen = nifty_puts.iloc[(nifty_puts['strike_num'] - atm_strike).abs().argsort()[:1]].iloc[0]
                logger.warning(f"ATM strike {atm_strike} PE not found — using closest: {chosen['strike_price']} | {chosen['trading_symbol']}")
            self.option_instrument = {
                'trading_symbol': chosen['trading_symbol'],
                'exchange': chosen['exchange'],
                'segment': chosen['segment'],
            }

            # --- Short call instrument (if configured) ---
            self.short_call_instrument = None
            if self.config.get("short_call_strike_offset"):
                short_strike = atm_strike + self.config["short_call_strike_offset"]
                nifty_calls = df[
                    (df['underlying_symbol'] == self.config['nifty_underlying']) &
                    (df['segment'] == 'FNO') &
                    (df['instrument_type'] == 'CE')
                ]
                calls_this_expiry = nifty_calls[
                    nifty_calls['expiry_date'].apply(lambda x: str(x)[:10]) == expiry_date
                ]
                sc_rows = calls_this_expiry[
                    calls_this_expiry['strike_price'].apply(
                        lambda x: int(float(x)) if pd.notna(x) else -1) == short_strike
                ]
                if sc_rows.empty:
                    logger.warning(f"Short call strike {short_strike} CE not found — skipping short call")
                else:
                    sc = sc_rows.iloc[0]
                    self.short_call_instrument = {
                        'trading_symbol': sc['trading_symbol'],
                        'exchange': sc['exchange'],
                        'segment': sc['segment'],
                    }
                    logger.info(f"Short call instrument loaded: {sc['trading_symbol']}")

            return True

        except Exception as e:
            logger.error(f"Error fetching option instruments: {e}")
            return False

    def get_current_price(self, instrument: dict) -> Optional[float]:
        """Get current market price for an instrument."""
        try:
            exchange_symbol = f"{instrument['exchange']}_{instrument['trading_symbol']}"
            ltp_data = self.groww.get_ltp(
                segment=instrument["segment"],
                exchange_trading_symbols=(exchange_symbol,)
            )
            return ltp_data.get(exchange_symbol)
        except Exception as e:
            logger.warning(f"Error fetching price for {instrument['trading_symbol']}: {e}")
            return None

    def _cancel_or_exit(self, instrument: dict, order_id: str, quantity: int, label: str,
                        confirmed_filled: bool = False, close_transaction_type=None,
                        limit_price: Optional[float] = None) -> None:
        """Cancel an open order or close a filled position.
        close_transaction_type defaults to TRANSACTION_TYPE_SELL (for long positions).
        Pass TRANSACTION_TYPE_BUY to cover/close a short position (e.g. sold call).
        confirmed_filled=True skips the cancel attempt and goes directly to the closing order.
        Pass limit_price to use a LIMIT order for the closing trade (required for NIFTY FUT).
        """
        if close_transaction_type is None:
            close_transaction_type = self.groww.TRANSACTION_TYPE_SELL
        close_word = "BUY BACK" if close_transaction_type == self.groww.TRANSACTION_TYPE_BUY else "SELL"

        if not confirmed_filled:
            try:
                self.groww.cancel_order(
                    segment=instrument["segment"],
                    groww_order_id=order_id
                )
                logger.info(f"[{label}] Order {order_id} cancelled successfully")
                return  # Cancelled — no open position to close
            except Exception as cancel_err:
                logger.warning(f"[{label}] Cancel failed ({cancel_err}) — placing market {close_word} to close position")
        else:
            logger.info(f"[{label}] Order confirmed filled — placing market {close_word} to close position")

        try:
            _close_kwargs = dict(
                trading_symbol=instrument["trading_symbol"],
                quantity=quantity,
                validity=self.groww.VALIDITY_DAY,
                exchange=instrument["exchange"],
                segment=instrument["segment"],
                product=self.groww.PRODUCT_MIS,
                order_type=self.groww.ORDER_TYPE_LIMIT if limit_price is not None else self.groww.ORDER_TYPE_MARKET,
                transaction_type=close_transaction_type,
                order_reference_id=f"CX{label[0]}{label[-1]}-{int(time.time() * 1000)}"
            )
            if limit_price is not None:
                _close_kwargs['price'] = limit_price
            self.groww.place_order(**_close_kwargs)
            _order_desc = f"LIMIT ₹{limit_price:.2f}" if limit_price is not None else "Market"
            logger.info(f"[{label}] {_order_desc} {close_word} placed to close position")
        except Exception as sell_err:
            logger.error(f"[{label}] CRITICAL: Failed to close position: {sell_err}")

    def _rollback_placed_orders(self, placed_orders: dict) -> None:
        """Cancel or close all orders in placed_orders.
        placed_orders: {label: (order_id, instrument, quantity, close_transaction_type)}
        Market orders fill in <1s; confirmed_filled=True skips the cancel attempt and goes
        straight to the closing trade.
        NIFTY FUT uses a LIMIT order (LTP±10) since broker blocks market orders for futures.
        """
        for label, (order_id, instrument, quantity, close_txn) in placed_orders.items():
            limit_price = None
            # NIFTY FUT is placed as a LIMIT order and may not be filled yet — attempt cancel first
            is_confirmed_filled = label != 'NIFTY_FUT'
            if label == 'NIFTY_FUT':
                ltp = self.get_current_price(instrument)
                if ltp is None:
                    logger.error(
                        f"[NIFTY_FUT] CRITICAL: LTP unavailable during rollback — "
                        f"cannot build LIMIT close order. Close this position MANUALLY via the broker app."
                    )
                    continue  # Cannot safely close without a price; skip to avoid a rejected MARKET order
                # Closing a LONG (BUY) with SELL at LTP-10; closing a SHORT (SELL) with BUY at LTP+10
                offset = -10 if close_txn == self.groww.TRANSACTION_TYPE_SELL else 10
                limit_price = round(ltp + offset, 2)
            self._cancel_or_exit(instrument, order_id, quantity, label,
                                 confirmed_filled=is_confirmed_filled, close_transaction_type=close_txn,
                                 limit_price=limit_price)

    def _place_with_retry(self, label: str, instrument: dict, quantity: int,
                          txn_type, ref_prefix: str, close_txn_type,
                          placed_orders: dict, seq: list, leg_max_retries: int = 3,
                          order_type=None, limit_price: Optional[float] = None) -> bool:
        """Place a single leg order with per-leg retry logic.
        Retries up to leg_max_retries times before giving up.
        Returns True if the order was placed successfully, False if all retries failed.
        Sets margin_shortfall_detected immediately on a margin error (no point retrying).
        Pass order_type=ORDER_TYPE_LIMIT and limit_price for limit orders (e.g. NIFTY FUT).
        """
        _order_type = order_type if order_type is not None else self.groww.ORDER_TYPE_MARKET
        for attempt in range(1, leg_max_retries + 1):
            try:
                seq[0] += 1
                _order_kwargs = dict(
                    trading_symbol=instrument["trading_symbol"],
                    quantity=quantity,
                    validity=self.groww.VALIDITY_DAY,
                    exchange=instrument["exchange"],
                    segment=instrument["segment"],
                    product=self.groww.PRODUCT_MIS,
                    order_type=_order_type,
                    transaction_type=txn_type,
                    order_reference_id=f"{ref_prefix}-{int(time.time() * 1000) + seq[0]}"
                )
                if _order_type == self.groww.ORDER_TYPE_LIMIT and limit_price is not None:
                    _order_kwargs['price'] = limit_price
                resp = self.groww.place_order(**_order_kwargs)
                placed_orders[label] = (resp.get('groww_order_id'), instrument, quantity, close_txn_type)
                logger.info(f"[{label}] Order placed (attempt {attempt}/{leg_max_retries}) - ID: {resp.get('groww_order_id')}")
                return True
            except Exception as e:
                logger.error(f"[{label}] Order failed (attempt {attempt}/{leg_max_retries}): {e}")
                if self._is_margin_shortfall_error(str(e)):
                    logger.error("[MARGIN SHORTFALL DETECTED] Stopping all retry attempts")
                    self.margin_shortfall_detected = True
                    return False
                if attempt < leg_max_retries:
                    logger.info(f"[{label}] Retrying leg in 2s...")
                    time.sleep(2)
        logger.error(f"[{label}] All {leg_max_retries} leg retries exhausted")
        return False

    def _wait_for_fill(self, order_id: str, segment: str, label: str, cancel_on_timeout: bool = True):
        """Poll order status until EXECUTED or timeout. Returns fill price (float) if confirmed filled, else None.
        cancel_on_timeout=True  (default): cancels the unfilled order on timeout — use for ENTRY orders.
        cancel_on_timeout=False: skips cancel on timeout — use for EXIT orders (cancelling a SELL
        order leaves the MIS position open, which is worse than waiting for broker auto-squareoff).
        """
        FILLED_STATUSES = {'EXECUTED', 'COMPLETED', 'DELIVERY_AWAITED'}
        TERMINAL_FAIL_STATUSES = {'REJECTED', 'FAILED', 'CANCELLED'}
        timeout = self.config.get("order_fill_timeout_seconds", 30)
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                status_resp = self.groww.get_order_status(
                    groww_order_id=order_id,
                    segment=segment
                )
                status = status_resp.get('order_status', '')
                logger.info(f"[{label}] Order {order_id} status: {status}")
                if status in FILLED_STATUSES:
                    fill_price = status_resp.get('average_traded_price') or status_resp.get('avg_price')
                    try:
                        fill_price = float(fill_price) if fill_price is not None else None
                    except (TypeError, ValueError):
                        fill_price = None
                    if fill_price is None:
                        fill_price = float('nan')
                        logger.warning(f"[{label}] Fill confirmed but avg price missing — price recorded as NaN (verify in broker app)")
                    logger.info(f"[{label}] Fill confirmed — avg price: ₹{fill_price:.2f}")
                    return fill_price
                if status in TERMINAL_FAIL_STATUSES:
                    rejection_reason = (
                        status_resp.get('rejection_reason')
                        or status_resp.get('reason')
                        or status_resp.get('message')
                        or status_resp.get('remark')    # Groww uses 'remark' (singular)
                        or status_resp.get('remarks')
                        or status_resp.get('error_message')
                        or "no reason provided"
                    )
                    logger.warning(
                        f"[{label}] Order {order_id} terminal failure: {status} | "
                        f"Reason: {rejection_reason} | Full response: {status_resp}"
                    )
                    if self._is_non_retryable_rejection(rejection_reason):
                        if 'lpp' in rejection_reason.lower() or 'circuit' in rejection_reason.lower():
                            logger.error(
                                f"[{label}] LPP BREACH — Groww API RMS has a stale LPP threshold. "
                                f"Market orders via app work but API orders are blocked. "
                                f"ACTION REQUIRED: Contact Groww API support to refresh LPP for your account. "
                                f"Rejection detail: {rejection_reason}"
                            )
                        else:
                            logger.error(
                                f"[{label}] Non-retryable rejection (margin/funds) — "
                                f"stopping all further retries. Detail: {rejection_reason}"
                            )
                        self.margin_shortfall_detected = True
                    return None
            except Exception as e:
                logger.warning(f"[{label}] Status poll failed: {e}")
            time.sleep(2)

        logger.warning(f"[{label}] Order {order_id} not filled within {timeout}s")
        if cancel_on_timeout:
            try:
                self.groww.cancel_order(segment=segment, groww_order_id=order_id)
                logger.info(f"[{label}] Cancel request accepted — checking final status to confirm")
            except Exception as e:
                logger.warning(f"[{label}] Cancel of unfilled order failed: {e}")
        else:
            logger.warning(f"[{label}] Skipping cancel (exit order) — position remains open, broker auto-squareoff will close it")

        # Always do a final status check after any cancel attempt.
        # Groww may accept a cancel on an already-filled order without raising an exception,
        # so we must verify the actual status to avoid a phantom rollback.
        try:
            final_resp = self.groww.get_order_status(
                groww_order_id=order_id, segment=segment
            )
            final_status = final_resp.get('order_status', '')
            logger.info(f"[{label}] Final status after cancel: {final_status}")
            if final_status in FILLED_STATUSES:
                fill_price = final_resp.get('average_traded_price') or final_resp.get('avg_price')
                try:
                    fill_price = float(fill_price) if fill_price is not None else None
                except (TypeError, ValueError):
                    fill_price = None
                if fill_price is None:
                    fill_price = float('nan')
                    logger.warning(f"[{label}] Order was filled before cancel but avg price missing — price recorded as NaN (verify in broker app)")
                logger.info(f"[{label}] Order was filled before cancel — avg price: ₹{fill_price:.2f}")
                return fill_price
        except Exception as e2:
            logger.warning(f"[{label}] Final status check failed: {e2}")
        return None

    def _is_margin_shortfall_error(self, error_message: str) -> bool:
        """Check if error message indicates margin shortfall."""
        margin_keywords = [
            'margin',
            'insufficient funds',
            'insufficient balance',
            'insufficient fund',
            'shortage',
            'shortfall',
            'not enough',
            'available balance'
        ]
        error_lower = str(error_message).lower()
        return any(keyword in error_lower for keyword in margin_keywords)

    def _is_non_retryable_rejection(self, rejection_reason: str) -> bool:
        """Check if a rejection reason indicates a non-retryable error.
        LPP (Last Price Protection) / circuit limit breaches will not resolve on retry
        because the market price itself is outside the RMS-allowed range.
        """
        keywords = [
            'lpp', 'lpp breach', 'circuit limit', 'circuit',
            'margin', 'insufficient funds', 'insufficient balance',
            'insufficient fund', 'shortfall', 'not enough', 'available balance',
        ]
        return any(kw in rejection_reason.lower() for kw in keywords)

    def execute_entry_orders(self) -> None:
        """Execute entry orders once, on first opportunity within the entry window.
        Legs: NIFTY FUT BUY + ATM Put BUY + (Short Call SELL if configured).
        Each leg is retried independently up to 3 times before giving up.
        If a leg exhausts all retries, all already-placed legs are rolled back
        and the strategy stops for the day (no further entry attempts).
        """
        self.entry_retry_count += 1
        logger.info(f"=== EXECUTING ENTRY ORDERS (Attempt {self.entry_retry_count}/{self.max_retries}) ===")

        # Resolve NIFTY FUT instrument (cached after first call)
        if not self._resolve_fut_instrument():
            logger.warning("Failed to resolve NIFTY FUT instrument — skipping entry")
            return

        if not self.get_option_instrument():
            logger.warning("Failed to fetch option instruments — skipping entry")
            return

        use_short_call = self.short_call_instrument is not None
        call_qty = self.config.get("nifty_call_quantity", 65)

        # placed_orders: label -> (order_id, instrument, quantity, close_transaction_type)
        placed_orders = {}
        _seq = [0]  # mutable counter — ensures unique order_reference_id even within same millisecond

        # --- NIFTY FUT BUY (LIMIT order at LTP+10; broker blocks market orders for futures) ---
        _fut_ltp = self.get_current_price(self.nifty_fut_instrument)
        if _fut_ltp is None:
            logger.error("[NIFTY_FUT] LTP unavailable — cannot place LIMIT entry order; aborting entry")
            return
        _fut_entry_limit = round(_fut_ltp + 10, 2)
        logger.info(f"[NIFTY_FUT] Placing LIMIT BUY at ₹{_fut_entry_limit:.2f} (LTP=₹{_fut_ltp:.2f}+10): "
                    f"{self.config['nifty_fut_lots']} lot × {self.nifty_fut_instrument['lot_size']} = Qty={self.nifty_fut_quantity}")
        if not self._place_with_retry('NIFTY_FUT', self.nifty_fut_instrument, self.nifty_fut_quantity,
                                      self.groww.TRANSACTION_TYPE_BUY, 'NFUT', self.groww.TRANSACTION_TYPE_SELL,
                                      placed_orders, _seq,
                                      order_type=self.groww.ORDER_TYPE_LIMIT, limit_price=_fut_entry_limit):
            logger.error("[NIFTY_FUT] All leg retries exhausted — aborting this entry attempt")
            return

        time.sleep(3)

        # --- ATM PUT BUY ---
        logger.info(f"[NIFTY_PUT] Placing MARKET BUY: Qty={self.config['nifty_option_quantity']}")
        if not self._place_with_retry('NIFTY_PUT', self.option_instrument, self.config["nifty_option_quantity"],
                                      self.groww.TRANSACTION_TYPE_BUY, 'NPUT', self.groww.TRANSACTION_TYPE_SELL,
                                      placed_orders, _seq):
            logger.error("[NIFTY_PUT] All leg retries exhausted — rolling back NIFTY_FUT")
            self._rollback_placed_orders(placed_orders)
            return

        if use_short_call:
            time.sleep(3)

            # --- SHORT CALL SELL ---
            logger.info(f"[SHORT_CALL] Placing MARKET SELL: Qty={call_qty}")
            if not self._place_with_retry('SHORT_CALL', self.short_call_instrument, call_qty,
                                          self.groww.TRANSACTION_TYPE_SELL, 'NCSS', self.groww.TRANSACTION_TYPE_BUY,
                                          placed_orders, _seq):
                logger.error("[SHORT_CALL] All leg retries exhausted — rolling back all legs")
                self._rollback_placed_orders(placed_orders)
                return

        # --- Verify fills for all legs ---
        fill_prices = {}
        for label, (order_id, instrument, qty, _) in placed_orders.items():
            fill_prices[label] = self._wait_for_fill(order_id, instrument["segment"], label)

        # For any leg that failed fill, retry ONLY that leg (confirmed legs are NOT re-placed)
        FILL_FAIL_RETRY_MAX = 3
        _retry_seq = [100]  # offset to avoid order_reference_id collision with initial placements
        _txn_type_map = {
            'NIFTY_FUT': self.groww.TRANSACTION_TYPE_BUY,
            'NIFTY_PUT': self.groww.TRANSACTION_TYPE_BUY,
            'SHORT_CALL': self.groww.TRANSACTION_TYPE_SELL,
        }
        _ref_prefix_map = {'NIFTY_FUT': 'NFRT', 'NIFTY_PUT': 'NPRT', 'SHORT_CALL': 'NCRT'}

        for label in list(placed_orders.keys()):
            if fill_prices.get(label) is not None:
                continue  # this leg already confirmed filled — leave it alone
            instrument = placed_orders[label][1]
            qty        = placed_orders[label][2]
            close_txn  = placed_orders[label][3]
            txn_type   = _txn_type_map.get(label)
            ref_prefix = _ref_prefix_map.get(label, 'RTRY')

            leg_filled = False
            for attempt in range(1, FILL_FAIL_RETRY_MAX + 1):
                if self.margin_shortfall_detected:
                    logger.error(f"[{label}] Non-retryable rejection detected — skipping fill retries")
                    break
                logger.warning(f"[{label}] Fill failed — retrying order placement "
                               f"(attempt {attempt}/{FILL_FAIL_RETRY_MAX})")
                time.sleep(2)
                retry_placed = {}
                if self._place_with_retry(label, instrument, qty, txn_type,
                                          ref_prefix, close_txn,
                                          retry_placed, _retry_seq, leg_max_retries=1):
                    new_order_id = retry_placed[label][0]
                    placed_orders[label] = retry_placed[label]  # update before fill check so rollback targets correct order
                    new_fill = self._wait_for_fill(new_order_id, instrument["segment"], label)
                    if new_fill is not None:
                        fill_prices[label] = new_fill
                        leg_filled = True
                        logger.info(f"[{label}] Fill retry {attempt} succeeded — "
                                    f"avg price: ₹{new_fill:.2f}")
                        break
                    else:
                        logger.warning(f"[{label}] Fill retry {attempt} — order placed but not filled")
                else:
                    logger.warning(f"[{label}] Fill retry {attempt} — order placement failed")

            if not leg_filled:
                logger.error(f"[{label}] All {FILL_FAIL_RETRY_MAX} fill retries exhausted — aborting entry")
                # Prevent the outer run() retry loop from re-placing all legs fresh
                self.entry_retry_count = self.max_retries
                break  # no point continuing with remaining legs

        if any(fill_prices.get(label) is None for label in placed_orders):
            logger.error("[ENTRY ABORTED] One or more legs did not fill after retries — rolling back confirmed fills")
            for label, (order_id, instrument, qty, close_txn) in placed_orders.items():
                if fill_prices.get(label) is not None:
                    self._cancel_or_exit(instrument, order_id, qty, label,
                                         confirmed_filled=True, close_transaction_type=close_txn)
            return

        # All filled — record positions
        self.positions['NIFTY_FUT'] = {
            'instrument': self.nifty_fut_instrument, 'quantity': self.nifty_fut_quantity,
            'entry_price': fill_prices['NIFTY_FUT'], 'order_id': placed_orders['NIFTY_FUT'][0],
            'side': 'LONG'
        }
        self.positions['NIFTY_PUT'] = {
            'instrument': self.option_instrument, 'quantity': self.config['nifty_option_quantity'],
            'entry_price': fill_prices['NIFTY_PUT'], 'order_id': placed_orders['NIFTY_PUT'][0],
            'side': 'LONG'
        }
        if use_short_call:
            self.positions['SHORT_CALL'] = {
                'instrument': self.short_call_instrument, 'quantity': call_qty,
                'entry_price': fill_prices['SHORT_CALL'], 'order_id': placed_orders['SHORT_CALL'][0],
                'side': 'SHORT'
            }

        self.entry_executed = True
        logger.info("=" * 60)
        logger.info(f"TRADE ENTRY  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  NIFTY_FUT  BUY   {self.nifty_fut_quantity:>6} qty @ ₹{fill_prices['NIFTY_FUT']:.2f}  |  Notional: ₹{fill_prices['NIFTY_FUT'] * self.nifty_fut_quantity:>12,.2f}")
        logger.info(f"  NIFTY_PUT  BUY   {self.config['nifty_option_quantity']:>6} qty @ ₹{fill_prices['NIFTY_PUT']:.2f}  |  Premium : ₹{fill_prices['NIFTY_PUT'] * self.config['nifty_option_quantity']:>12,.2f}")
        if use_short_call:
            logger.info(f"  SHORT_CALL SELL  {call_qty:>6} qty @ ₹{fill_prices['SHORT_CALL']:.2f}  |  Credit  : ₹{fill_prices['SHORT_CALL'] * call_qty:>12,.2f}")
        logger.info("=" * 60)

    def execute_exit_orders(self) -> None:
        """Exit all open positions with market orders at the configured exit time."""
        logger.info("=== EXECUTING EXIT ORDERS ===")

        if not self.positions:
            logger.info("No positions to exit")
            self.exit_executed = True
            return

        # Closing transaction type for each label:
        # LONG positions are closed with SELL; SHORT_CALL (sold) is closed with BUY (cover)
        exit_config = {
            'NIFTY_FUT':  ('NFTS', self.groww.TRANSACTION_TYPE_SELL),
            'NIFTY_PUT':  ('NPTS', self.groww.TRANSACTION_TYPE_SELL),
            'SHORT_CALL': ('NCSB', self.groww.TRANSACTION_TYPE_BUY),   # buy back the sold call
        }

        exit_failed = False
        exit_order_ids = {}  # label -> order_id

        for label, position in self.positions.items():
            if label not in exit_config:
                continue
            ref_prefix, txn_type = exit_config[label]
            side_word = "BUY BACK" if txn_type == self.groww.TRANSACTION_TYPE_BUY else "SELL"
            try:
                ltp = self.get_current_price(position['instrument'])
                # NIFTY FUT requires LIMIT order — broker blocks market orders for futures
                if label == 'NIFTY_FUT':
                    if ltp is None:
                        logger.error(f"[{label}] LTP unavailable — cannot place LIMIT exit order; skipping")
                        exit_failed = True
                        continue
                    exit_limit_price = round(ltp - 10, 2)
                    logger.info(f"[{label}] Placing LIMIT {side_word} at ₹{exit_limit_price:.2f} "
                                f"(LTP=₹{ltp:.2f}-10): Qty={position['quantity']}")
                    _exit_kwargs = dict(
                        trading_symbol=position['instrument']["trading_symbol"],
                        quantity=position['quantity'],
                        validity=self.groww.VALIDITY_DAY,
                        exchange=position['instrument']["exchange"],
                        segment=position['instrument']["segment"],
                        product=self.groww.PRODUCT_MIS,
                        order_type=self.groww.ORDER_TYPE_LIMIT,
                        transaction_type=txn_type,
                        price=exit_limit_price,
                        order_reference_id=f"{ref_prefix}-{int(time.time() * 1000)}"
                    )
                else:
                    logger.info(f"[{label}] Placing MARKET {side_word}: Qty={position['quantity']}" +
                                (f", LTP=₹{ltp:.2f}" if ltp else " (LTP unavailable)"))
                    _exit_kwargs = dict(
                        trading_symbol=position['instrument']["trading_symbol"],
                        quantity=position['quantity'],
                        validity=self.groww.VALIDITY_DAY,
                        exchange=position['instrument']["exchange"],
                        segment=position['instrument']["segment"],
                        product=self.groww.PRODUCT_MIS,
                        order_type=self.groww.ORDER_TYPE_MARKET,
                        transaction_type=txn_type,
                        order_reference_id=f"{ref_prefix}-{int(time.time() * 1000)}"
                    )
                resp = self.groww.place_order(**_exit_kwargs)
                exit_order_ids[label] = resp.get('groww_order_id')
                logger.info(f"[{label}] Exit order placed - ID: {exit_order_ids[label]}")
                time.sleep(3)
            except Exception as e:
                logger.error(f"[{label}] Exit order failed: {e}")
                exit_failed = True

        # Wait for actual exit fills (never cancel exit SELL/BUY-back orders on timeout)
        exit_fills = {}
        for label, order_id in exit_order_ids.items():
            exit_fills[label] = self._wait_for_fill(
                order_id, self.positions[label]['instrument']['segment'],
                f"{label} EXIT", cancel_on_timeout=False
            )

        # Log exit summary
        logger.info("=" * 60)
        logger.info(f"TRADE EXIT   |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        total_pnl = 0.0
        all_fills_available = True
        for label, pos in self.positions.items():
            fill = exit_fills.get(label)
            if label not in exit_order_ids:
                logger.warning(f"  {label:<12} exit order was NOT placed — position may still be open")
                all_fills_available = False
            elif fill is not None:
                side = pos.get('side', 'LONG')
                pnl = (pos['entry_price'] - fill if side == 'SHORT' else fill - pos['entry_price']) * pos['quantity']
                pnl_pct = pnl / (pos['entry_price'] * pos['quantity']) * 100 if pos['entry_price'] else 0.0
                side_word = "BUY BACK" if side == 'SHORT' else "SELL    "
                logger.info(f"  {label:<12} {side_word}  {pos['quantity']:>6} qty @ ₹{fill:.2f}  |  Entry: ₹{pos['entry_price']:.2f}  |  P&L: ₹{pnl:>+10,.2f} ({pnl_pct:+.2f}%)")
                total_pnl += pnl
            else:
                logger.warning(f"  {label:<12} exit fill price unavailable — P&L not calculated")
                all_fills_available = False
        if all_fills_available and self.positions:
            logger.info(f"  NET P&L{'.' * 43} ₹{total_pnl:>+10,.2f}")
        logger.info("=" * 60)

        self.exit_executed = True
        if exit_failed:
            logger.warning("[CRITICAL] One or more exit orders failed — verify positions manually. "
                           "Broker MIS auto-squareoff will close any open positions at 3:20 PM.")
        logger.info("Exit orders execution completed")

    def run(self) -> None:
        """Single-day execution loop.

        Designed for Groww Cloud: runs once for the trading day and exits cleanly.
        Groww Cloud's scheduler handles restarting the script the next morning.
        """
        logger.info("Intraday strategy started - waiting for entry time")

        entry_start_time = datetime.strptime(self.config["entry_start_time"], "%H:%M:%S").time()
        entry_cutoff_time = datetime.strptime(self.config["entry_cutoff_time"], "%H:%M:%S").time()
        exit_time = datetime.strptime(self.config["exit_time"], "%H:%M:%S").time()

        while True:
            try:
                current_time = datetime.now()

                # Check market hours
                if not is_market_open(current_time):
                    logger.info(f"Market closed at {current_time.strftime('%H:%M:%S')} - waiting...")
                    time.sleep(60)
                    continue

                current_time_only = current_time.time()

                # Execute entry — fires once on first tick inside the entry window
                if not self.entry_executed:
                    # Check if we should skip retries
                    if self.margin_shortfall_detected:
                        logger.error("Entry aborted permanently (margin shortfall or leg retry limit reached) — no retries allowed")
                        self.entry_executed = True  # Stop trying
                    elif self.entry_retry_count >= self.max_retries and not self.positions:
                        logger.error(f"Maximum retry attempts ({self.max_retries}) reached — stopping entry attempts")
                        self.entry_executed = True  # Stop trying
                    elif entry_start_time <= current_time_only < entry_cutoff_time:
                        # Only retry if we haven't exceeded limits and entry hasn't succeeded
                        if not self.positions:  # No successful entry yet
                            if self.entry_retry_count < self.max_retries:
                                attempt_word = "Placing" if self.entry_retry_count == 0 else "Retrying"
                                logger.info(f"Entry window open — {attempt_word} orders at {current_time_only.strftime('%H:%M:%S')}")
                                self.execute_entry_orders()
                            else:
                                logger.error(f"Maximum retry attempts ({self.max_retries}) reached — stopping entry attempts")
                                self.entry_executed = True
                        else:
                            # Entry succeeded
                            self.entry_executed = True
                    elif current_time_only >= entry_cutoff_time:
                        logger.warning(f"Entry cutoff {entry_cutoff_time.strftime('%H:%M')} passed with no entry — skipping to exit")
                        self.entry_executed = True

                # Execute exit at 3:04 PM
                if self.entry_executed and not self.exit_executed and current_time_only >= exit_time:
                    # Always attempt to exit — even if window is missed, positions must be closed
                    # before broker MIS auto-squareoff at 3:20 PM
                    exit_window_end = (datetime.combine(datetime.today(), exit_time) + timedelta(minutes=1)).time()
                    if current_time_only > exit_window_end:
                        logger.warning("[WARN] Exit time window missed — forcing exit now to avoid auto-squareoff penalty")
                    self.execute_exit_orders()

                # Exit cleanly once the day's work is done.
                # Groww Cloud scheduler will restart the script next morning.
                if self.exit_executed:
                    logger.info("=== Strategy completed for the day — exiting ===")
                    return

                # Heartbeat log — confirms script is alive and shows current state
                state_msg = ""
                if not self.entry_executed:
                    if current_time_only < entry_start_time:
                        state_msg = f"Waiting for entry window (opens at {entry_start_time.strftime('%H:%M')})"
                    elif current_time_only < entry_cutoff_time:
                        state_msg = f"In entry window, awaiting retry"
                    else:
                        state_msg = "Entry cutoff passed"
                elif self.positions:
                    pos_count = len(self.positions)
                    state_msg = f"Monitoring {pos_count} position(s), exit at {exit_time.strftime('%H:%M')}"
                else:
                    state_msg = f"No positions, exit at {exit_time.strftime('%H:%M')}"

                logger.info(f"[HEARTBEAT] {current_time.strftime('%H:%M:%S')} | {state_msg}")

                # Sleep before next check
                time.sleep(self.config["check_interval_seconds"])

            except Exception as e:
                logger.error(f"ERROR: {e}")
                time.sleep(60)


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    logger.info("=== Intraday NIFTY FUT (1 lot) + NIFTY ATM Put + Optional Short Call Strategy ===")

    # Authenticate using TOTP
    groww = authenticate_with_totp()

    # Initialize and run strategy
    strategy = IntradayStrategy(groww, STRATEGY_CONFIG)
    strategy.run()
