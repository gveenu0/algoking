"""
Buy NIFTYBEES + NIFTY ATM Put at 9:16-14:00. Exit all at 15:04.
Optional: Sell naked OTM Call (ATM+500) to collect premium and reduce net cost.

Strategy Legs:
- NIFTYBEES BUY: Profits if market goes up
- ATM Put BUY: Protects if market falls (insurance premium paid daily)
- OTM Call SELL (optional): Collects premium to offset put cost but caps upside at strike

All legs are held together — if any placement fails, entire position is rolled back.
Uses TOTP authentication with Groww API.
"""

# ===== IMPORTS =====
import time
import logging
import os
import pyotp
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
    "niftybees_symbol": "NIFTYBEES",
    "niftybees_quantity": 6500,
    "nifty_underlying": "NIFTY",
    "nifty_option_quantity": 65,  # NIFTY lot size = 65 (1 lot) as per NSE latest update. Use multiples of 65 only.

    # Exchanges
    "equity_exchange": "NSE",
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
    # Reference: instrument CSV column lot_size for NIFTY options = 65
}


# ===== HELPER FUNCTIONS =====

def is_market_open(current_time: datetime) -> bool:
    """Check if NSE market is open (9:15 AM - 3:30 PM, weekdays)"""
    if current_time.weekday() >= 5:
        return False
    return dt_time(9, 15) <= current_time.time() <= dt_time(15, 30)


def authenticate_with_totp() -> GrowwAPI:
    """Authenticate using TOTP"""
    try:
        raw_secret = AUTH_CONFIG["totp_secret"]
        sanitized_secret = raw_secret.replace(" ", "").replace("-", "").upper()
        totp = pyotp.TOTP(sanitized_secret).now()
        access_token = GrowwAPI.get_access_token(
            api_key=AUTH_CONFIG["api_key"],
            totp=totp
        )
        logger.info("Authentication successful using TOTP")
        return GrowwAPI(access_token)
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise


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

        # Keep only future/current expiries
        future_expiries = []
        for exp in raw_expiries:
            exp_str = str(exp)[:10]  # Ensure YYYY-MM-DD format
            if datetime.strptime(exp_str, '%Y-%m-%d').date() >= today:
                future_expiries.append(exp_str)

        if future_expiries:
            nearest_expiry = min(future_expiries, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
            logger.info(f"Nearest weekly expiry: {nearest_expiry}")
            return nearest_expiry

        return None

    except Exception as e:
        logger.error(f"Error fetching expiries: {e}")
        return None


def get_atm_strike(groww: GrowwAPI, underlying: str = "NIFTY") -> Optional[int]:
    """Get ATM (At-The-Money) strike price for NIFTY"""
    try:
        # Get NIFTY index instrument
        index_instrument = groww.get_instrument_by_groww_symbol(groww_symbol=f"NSE-{underlying}")

        # Get current NIFTY price
        # Pass as tuple so get_ltp returns the keyed format {"NSE_NIFTY": price}
        exchange_symbol = f"{index_instrument['exchange']}_{index_instrument['trading_symbol']}"
        ltp_data = groww.get_ltp(
            segment=index_instrument["segment"],
            exchange_trading_symbols=(exchange_symbol,)
        )

        current_price = ltp_data.get(exchange_symbol)

        if current_price is None:
            return None

        # Round to nearest 50 (NIFTY strike interval)
        atm_strike = round(current_price / 50) * 50
        logger.info(f"NIFTY current price: ₹{current_price:.2f} | ATM strike: {atm_strike}")
        return int(atm_strike)

    except Exception as e:
        logger.error(f"Error calculating ATM strike: {e}")
        return None


_MONTH_ABBR = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

def format_expiry_for_symbol(expiry_date: str) -> str:
    """Convert YYYY-MM-DD to DMmmYY format for Groww symbol (no leading zero on day).
    Uses a hardcoded month map to avoid locale-dependent strftime('%b') output.
    Reference: instrument CSV shows NSE-NIFTY-27Mar25-29050-PE (no leading zero on day)
    """
    dt = datetime.strptime(expiry_date, '%Y-%m-%d')
    return f"{dt.day}{_MONTH_ABBR[dt.month]}{dt.strftime('%y')}"


# ===== STRATEGY CLASS =====

class IntradayStrategy:
    """Intraday strategy for NIFTYBEES + NIFTY ATM Put with optional naked short call"""

    def __init__(self, groww: GrowwAPI, config: dict):
        self.groww = groww
        self.config = config

        # State tracking
        self.positions = {}
        self.entry_executed = False
        self.exit_executed = False
        self._instruments_df = None  # Cached instrument master — loaded once, reused on retry
        self.entry_retry_count = 0  # Track number of entry attempts
        self.max_retries = 3  # Maximum retry attempts
        self.margin_shortfall_detected = False  # Flag to stop retries on margin shortfall

        # Fetch NIFTYBEES instrument
        logger.info("Fetching NIFTYBEES instrument...")
        try:
            self.niftybees_instrument = groww.get_instrument_by_exchange_and_trading_symbol(
                exchange=config["equity_exchange"],
                trading_symbol=config["niftybees_symbol"]
            )
            logger.info(f"NIFTYBEES instrument loaded: {self.niftybees_instrument['trading_symbol']}")
        except Exception as e:
            logger.error(f"Failed to load NIFTYBEES instrument: {e}")
            raise

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

    def get_option_instrument(self) -> bool:
        """Fetch NIFTY weekly ATM Put option instrument"""
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

            # Get ATM strike
            atm_strike = get_atm_strike(self.groww, self.config["nifty_underlying"])
            if not atm_strike:
                logger.warning("Failed to calculate ATM strike")
                return False

            # Format expiry for Groww symbol
            expiry_formatted = format_expiry_for_symbol(expiry_date)

            # Construct Groww symbol for Put option
            option_symbol = f"{self.config['option_exchange']}-{self.config['nifty_underlying']}-{expiry_formatted}-{atm_strike}-PE"
            logger.info(f"Option symbol: {option_symbol}")

            # Fetch ATM Put instrument
            self.option_instrument = self.groww.get_instrument_by_groww_symbol(
                groww_symbol=option_symbol
            )
            logger.info(f"ATM Put instrument loaded: {self.option_instrument['trading_symbol']}")

            # --- Short call instrument (if configured) ---
            self.short_call_instrument = None
            if self.config.get("short_call_strike_offset"):
                short_strike = atm_strike + self.config["short_call_strike_offset"]
                df = self._load_instruments_df()
                nifty_calls = df[
                    (df['underlying_symbol'] == self.config['nifty_underlying']) &
                    (df['segment'] == 'FNO') &
                    (df['instrument_type'] == 'CE')
                ]
                calls_this_expiry = nifty_calls[
                    nifty_calls['expiry_date'].apply(lambda x: str(x)[:10]) == expiry_date
                ]

                # Short call at ATM + offset
                sc_rows = calls_this_expiry[
                    calls_this_expiry['strike_price'].apply(
                        lambda x: int(x) if x == x else -1) == short_strike  # NaN-safe
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
        """Get current market price for an instrument"""
        try:
            # Pass as tuple so get_ltp returns the keyed format {"NSE_SYMBOL": price}
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
                        confirmed_filled: bool = False, close_transaction_type=None) -> None:
        """Cancel an open order or close a filled position.
        close_transaction_type defaults to TRANSACTION_TYPE_SELL (for long positions).
        Pass TRANSACTION_TYPE_BUY to cover/close a short position (e.g. sold call).
        confirmed_filled=True skips the cancel attempt and goes directly to the closing order.
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
            self.groww.place_order(
                trading_symbol=instrument["trading_symbol"],
                quantity=quantity,
                validity=self.groww.VALIDITY_DAY,
                exchange=instrument["exchange"],
                segment=instrument["segment"],
                product=self.groww.PRODUCT_MIS,
                order_type=self.groww.ORDER_TYPE_MARKET,
                transaction_type=close_transaction_type,
                order_reference_id=f"CX{label[0]}{label[-1]}-{int(time.time() * 1000)}"
            )
            logger.info(f"[{label}] Market {close_word} placed to close position")
        except Exception as sell_err:
            logger.error(f"[{label}] CRITICAL: Failed to close position: {sell_err}")

    def _rollback_placed_orders(self, placed_orders: dict) -> None:
        """Cancel or close all orders in placed_orders.
        placed_orders: {label: (order_id, instrument, quantity, close_transaction_type)}
        Market orders fill in <1s; confirmed_filled=True skips the cancel attempt (which Groww
        may silently accept on a filled order) and goes straight to the closing trade.
        """
        for label, (order_id, instrument, quantity, close_txn) in placed_orders.items():
            self._cancel_or_exit(instrument, order_id, quantity, label,
                                 confirmed_filled=True, close_transaction_type=close_txn)

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
                    fill_price = status_resp.get('average_traded_price') or status_resp.get('avg_price') or 0.0
                    try:
                        fill_price = float(fill_price)
                    except (TypeError, ValueError):
                        fill_price = 0.0
                    logger.info(f"[{label}] Fill confirmed — avg price: ₹{fill_price:.2f}")
                    return fill_price
                if status in TERMINAL_FAIL_STATUSES:
                    logger.warning(f"[{label}] Order {order_id} terminal failure: {status}")
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
                fill_price = final_resp.get('average_traded_price') or final_resp.get('avg_price') or 0.0
                try:
                    fill_price = float(fill_price)
                except (TypeError, ValueError):
                    fill_price = 0.0
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
            'fund',
            'shortage',
            'shortfall',
            'not enough',
            'available balance'
        ]
        error_lower = str(error_message).lower()
        return any(keyword in error_lower for keyword in margin_keywords)

    def execute_entry_orders(self) -> None:
        """Execute entry orders once, on first opportunity within the entry window.
        Legs: NIFTYBEES BUY + ATM Put BUY + (Short Call SELL if configured).
        All legs placed back-to-back. All must fill or the entire position is rolled back.
        """
        self.entry_retry_count += 1
        logger.info(f"=== EXECUTING ENTRY ORDERS (Attempt {self.entry_retry_count}/{self.max_retries}) ===")

        if not self.get_option_instrument():
            logger.warning("Failed to fetch option instruments - skipping entry")
            return

        use_short_call = self.short_call_instrument is not None
        call_qty = self.config.get("nifty_call_quantity", 65)

        # placed_orders: label -> (order_id, instrument, quantity, close_transaction_type)
        placed_orders = {}
        _seq = [0]  # mutable counter — ensures unique order_reference_id even within same millisecond

        def place(label, instrument, quantity, txn_type, ref_prefix, close_txn_type):
            _seq[0] += 1
            resp = self.groww.place_order(
                trading_symbol=instrument["trading_symbol"],
                quantity=quantity,
                validity=self.groww.VALIDITY_DAY,
                exchange=instrument["exchange"],
                segment=instrument["segment"],
                product=self.groww.PRODUCT_MIS,
                order_type=self.groww.ORDER_TYPE_MARKET,
                transaction_type=txn_type,
                order_reference_id=f"{ref_prefix}-{int(time.time() * 1000) + _seq[0]}"
            )
            placed_orders[label] = (resp.get('groww_order_id'), instrument, quantity, close_txn_type)
            logger.info(f"[{label}] Order placed - ID: {resp.get('groww_order_id')}")

        # --- NIFTYBEES BUY ---
        try:
            logger.info(f"[NIFTYBEES] Placing MARKET BUY: Qty={self.config['niftybees_quantity']}")
            place('NIFTYBEES', self.niftybees_instrument, self.config["niftybees_quantity"],
                  self.groww.TRANSACTION_TYPE_BUY, 'NBEE', self.groww.TRANSACTION_TYPE_SELL)
        except Exception as e:
            logger.error(f"[NIFTYBEES] Order failed: {e} — aborting entry")
            if self._is_margin_shortfall_error(str(e)):
                logger.error("[MARGIN SHORTFALL DETECTED] Stopping all retry attempts")
                self.margin_shortfall_detected = True
            return

        time.sleep(3)

        # --- ATM PUT BUY ---
        try:
            logger.info(f"[NIFTY_PUT] Placing MARKET BUY: Qty={self.config['nifty_option_quantity']}")
            place('NIFTY_PUT', self.option_instrument, self.config["nifty_option_quantity"],
                  self.groww.TRANSACTION_TYPE_BUY, 'NPUT', self.groww.TRANSACTION_TYPE_SELL)
        except Exception as e:
            logger.error(f"[NIFTY_PUT] Order failed: {e} — rolling back")
            if self._is_margin_shortfall_error(str(e)):
                logger.error("[MARGIN SHORTFALL DETECTED] Stopping all retry attempts")
                self.margin_shortfall_detected = True
            self._rollback_placed_orders(placed_orders)
            return

        if use_short_call:
            time.sleep(3)

            # --- SHORT CALL SELL ---
            try:
                logger.info(f"[SHORT_CALL] Placing MARKET SELL: Qty={call_qty}")
                place('SHORT_CALL', self.short_call_instrument, call_qty,
                      self.groww.TRANSACTION_TYPE_SELL, 'NCSS', self.groww.TRANSACTION_TYPE_BUY)
            except Exception as e:
                logger.error(f"[SHORT_CALL] Order failed: {e} — rolling back")
                if self._is_margin_shortfall_error(str(e)):
                    logger.error("[MARGIN SHORTFALL DETECTED] Stopping all retry attempts")
                    self.margin_shortfall_detected = True
                self._rollback_placed_orders(placed_orders)
                return

        # --- Verify fills for all legs ---
        fill_prices = {}
        for label, (order_id, instrument, qty, _) in placed_orders.items():
            fill_prices[label] = self._wait_for_fill(order_id, instrument["segment"], label)

        if any(p is None for p in fill_prices.values()):
            logger.error("[ENTRY ABORTED] One or more legs did not fill — rolling back confirmed fills")
            for label, (order_id, instrument, qty, close_txn) in placed_orders.items():
                if fill_prices.get(label) is not None:
                    self._cancel_or_exit(instrument, order_id, qty, label,
                                         confirmed_filled=True, close_transaction_type=close_txn)
            return

        # All filled — record positions
        self.positions['NIFTYBEES'] = {
            'instrument': self.niftybees_instrument, 'quantity': self.config['niftybees_quantity'],
            'entry_price': fill_prices['NIFTYBEES'], 'order_id': placed_orders['NIFTYBEES'][0],
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
        logger.info(f"  NIFTYBEES  BUY   {self.config['niftybees_quantity']:>6} qty @ ₹{fill_prices['NIFTYBEES']:.2f}  |  Cost    : ₹{fill_prices['NIFTYBEES'] * self.config['niftybees_quantity']:>12,.2f}")
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
            'NIFTYBEES':  ('NBES', self.groww.TRANSACTION_TYPE_SELL),
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
                logger.info(f"[{label}] Placing MARKET {side_word}: Qty={position['quantity']}" +
                            (f", LTP=\u20b9{ltp:.2f}" if ltp else " (LTP unavailable)"))
                resp = self.groww.place_order(
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
                logger.info(f"  {label:<12} {side_word}  {pos['quantity']:>6} qty @ \u20b9{fill:.2f}  |  Entry: \u20b9{pos['entry_price']:.2f}  |  P&L: \u20b9{pnl:>+10,.2f} ({pnl_pct:+.2f}%)")
                total_pnl += pnl
            else:
                logger.warning(f"  {label:<12} exit fill price unavailable \u2014 P&L not calculated")
                all_fills_available = False
        if all_fills_available and self.positions:
            logger.info(f"  NET P&L{'.' * 43} \u20b9{total_pnl:>+10,.2f}")
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
                        logger.error("Entry aborted due to margin shortfall — no retries allowed")
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
    logger.info("=== Intraday NIFTYBEES + NIFTY ATM Put + Optional Short Call Strategy ===")

    # Authenticate using TOTP
    groww = authenticate_with_totp()

    # Initialize and run strategy
    strategy = IntradayStrategy(groww, STRATEGY_CONFIG)
    strategy.run()