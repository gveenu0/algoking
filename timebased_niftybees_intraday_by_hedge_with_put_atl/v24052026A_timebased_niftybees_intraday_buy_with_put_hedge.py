"""
Intraday strategy: Buy NIFTYBEES + NIFTY weekly ATM Put. Exit both at 3:04 PM.

Entry window : 9:16 AM  2:00 PM (first tick inside the window triggers entry)
Exit time    : 3:04 PM (market order; forced even if window is slightly missed)
Hedge logic  : NIFTYBEES profits if market rises; ATM Put limits downside if it falls
Net cost     : Put premium paid daily as insurance against a fall
Auth         : TOTP-based authentication via pyotp + Groww API
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
    "exit_price_buffer": 0.001,        # Retained for reference — no longer used (orders are market type)
    "order_fill_timeout_seconds": 60,  # Seconds to wait for entry order to confirm as filled

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
    """Intraday strategy for NIFTYBEES + NIFTY ATM Put"""

    def __init__(self, groww: GrowwAPI, config: dict):
        self.groww = groww
        self.config = config

        # State tracking
        self.positions = {}
        self.entry_executed = False
        self.exit_executed = False
        self._instruments_df = None  # Cached instrument master — loaded once, reused on retry

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

        # Option instrument will be fetched at entry time
        self.option_instrument = None

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

            # Fetch option instrument
            self.option_instrument = self.groww.get_instrument_by_groww_symbol(
                groww_symbol=option_symbol
            )
            logger.info(f"Option instrument loaded: {self.option_instrument['trading_symbol']}")
            return True

        except Exception as e:
            logger.error(f"Error fetching option instrument: {e}")
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
                        confirmed_filled: bool = False) -> None:
        """Cancel an open order or close a filled position via market SELL.
        Used to roll back one leg when the other leg of the hedge pair fails.
        Set confirmed_filled=True when the order is already known to be filled —
        this skips the cancel attempt and goes directly to market SELL, preventing
        a silent cancel on a filled order from leaving the position open.
        """
        if not confirmed_filled:
            try:
                self.groww.cancel_order(
                    segment=instrument["segment"],
                    groww_order_id=order_id
                )
                logger.info(f"[{label}] Order {order_id} cancelled successfully")
                return  # Cancelled — no open position to close
            except Exception as cancel_err:
                logger.warning(f"[{label}] Cancel failed ({cancel_err}) — placing market SELL to close filled position")
        else:
            logger.info(f"[{label}] Order confirmed filled — placing market SELL to close position")

        try:
            self.groww.place_order(
                trading_symbol=instrument["trading_symbol"],
                quantity=quantity,
                validity=self.groww.VALIDITY_DAY,
                exchange=instrument["exchange"],
                segment=instrument["segment"],
                product=self.groww.PRODUCT_MIS,
                order_type=self.groww.ORDER_TYPE_MARKET,
                transaction_type=self.groww.TRANSACTION_TYPE_SELL,
                order_reference_id=f"CNCL-{int(time.time() * 1000)}"
            )
            logger.info(f"[{label}] Market SELL placed to close position")
        except Exception as sell_err:
            logger.error(f"[{label}] CRITICAL: Failed to cancel or exit position: {sell_err}")

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

    def execute_entry_orders(self) -> None:
        """Execute entry orders once, on first opportunity within the entry window.
        Both NIFTYBEES and NIFTY PUT are placed back-to-back immediately so both
        hit the market at the same time. Fill confirmation happens after both are placed.
        If either leg fails to place or fill, both are cancelled/exited.
        """
        logger.info("=== EXECUTING ENTRY ORDERS ===")

        # Fetch option instrument
        if not self.get_option_instrument():
            logger.warning("Failed to fetch option instrument - skipping entry")
            return

        # --- Step 1: Place NIFTYBEES order ---
        try:
            logger.info(f"[NIFTYBEES] Placing MARKET BUY order: Qty={self.config['niftybees_quantity']}")
            response = self.groww.place_order(
                trading_symbol=self.niftybees_instrument["trading_symbol"],
                quantity=self.config["niftybees_quantity"],
                validity=self.groww.VALIDITY_DAY,
                exchange=self.niftybees_instrument["exchange"],
                segment=self.niftybees_instrument["segment"],
                product=self.groww.PRODUCT_MIS,
                order_type=self.groww.ORDER_TYPE_MARKET,
                transaction_type=self.groww.TRANSACTION_TYPE_BUY,
                order_reference_id=f"NBEE-{int(time.time() * 1000)}"
            )
            niftybees_order_id = response.get('groww_order_id')
            logger.info(f"[NIFTYBEES] Order placed - ID: {niftybees_order_id}")
        except Exception as e:
            logger.error(f"[NIFTYBEES] Order failed: {e} — aborting entry")
            return

        # --- Step 2: Place NIFTY PUT order immediately (no fill wait in between) ---
        try:
            logger.info(f"[NIFTY PUT] Placing MARKET BUY order: Qty={self.config['nifty_option_quantity']}")
            response = self.groww.place_order(
                trading_symbol=self.option_instrument["trading_symbol"],
                quantity=self.config["nifty_option_quantity"],
                validity=self.groww.VALIDITY_DAY,
                exchange=self.option_instrument["exchange"],
                segment=self.option_instrument["segment"],
                product=self.groww.PRODUCT_MIS,
                order_type=self.groww.ORDER_TYPE_MARKET,
                transaction_type=self.groww.TRANSACTION_TYPE_BUY,
                order_reference_id=f"NPUT-{int(time.time() * 1000)}"
            )
            put_order_id = response.get('groww_order_id')
            logger.info(f"[NIFTY PUT] Order placed - ID: {put_order_id}")
        except Exception as e:
            logger.error(f"[NIFTY PUT] Order failed: {e} — cancelling NIFTYBEES order")
            self._cancel_or_exit(self.niftybees_instrument, niftybees_order_id,
                                 self.config["niftybees_quantity"], "NIFTYBEES")
            logger.error("[ENTRY ABORTED] Hedge pair must be held together")
            return

        # --- Step 3: Verify fills for both legs (after both are placed) ---
        niftybees_fill_price = self._wait_for_fill(
            niftybees_order_id, self.niftybees_instrument["segment"], "NIFTYBEES"
        )
        put_fill_price = self._wait_for_fill(
            put_order_id, self.option_instrument["segment"], "NIFTY PUT"
        )

        if niftybees_fill_price is None or put_fill_price is None:
            logger.error("[ENTRY ABORTED] One or both legs did not fill — rolling back both")
            if niftybees_fill_price is not None:
                # Confirmed filled — skip cancel, go directly to market SELL
                self._cancel_or_exit(self.niftybees_instrument, niftybees_order_id,
                                     self.config["niftybees_quantity"], "NIFTYBEES",
                                     confirmed_filled=True)
            if put_fill_price is not None:
                # Confirmed filled — skip cancel, go directly to market SELL
                self._cancel_or_exit(self.option_instrument, put_order_id,
                                     self.config["nifty_option_quantity"], "NIFTY PUT",
                                     confirmed_filled=True)
            return

        # Both legs placed and confirmed filled — record positions together
        self.positions['NIFTYBEES'] = {
            'instrument': self.niftybees_instrument,
            'quantity': self.config['niftybees_quantity'],
            'entry_price': niftybees_fill_price,
            'order_id': niftybees_order_id
        }
        self.positions['NIFTY_PUT'] = {
            'instrument': self.option_instrument,
            'quantity': self.config['nifty_option_quantity'],
            'entry_price': put_fill_price,
            'order_id': put_order_id
        }
        self.entry_executed = True
        logger.info("=" * 60)
        logger.info(f"TRADE ENTRY  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  NIFTYBEES  BUY   {self.config['niftybees_quantity']:>6} qty @ \u20b9{niftybees_fill_price:.2f}  |  Cost    : \u20b9{niftybees_fill_price * self.config['niftybees_quantity']:>12,.2f}")
        logger.info(f"  NIFTY PUT  BUY   {self.config['nifty_option_quantity']:>6} qty @ \u20b9{put_fill_price:.2f}  |  Premium : \u20b9{put_fill_price * self.config['nifty_option_quantity']:>12,.2f}")
        logger.info("=" * 60)

    def execute_exit_orders(self) -> None:
        """Execute exit orders at 3:04 PM.
        Places MARKET SELL for both NIFTYBEES and NIFTY PUT positions.
        """
        logger.info("=== EXECUTING EXIT ORDERS ===")

        if not self.positions:
            logger.info("No positions to exit")
            self.exit_executed = True
            return

        exit_failed = False
        nbes_order_id = None
        npts_order_id = None

        # Exit NIFTYBEES position
        if 'NIFTYBEES' in self.positions:
            try:
                position = self.positions['NIFTYBEES']
                exit_ltp = self.get_current_price(position['instrument'])
                logger.info(f"[NIFTYBEES] Placing MARKET SELL order: Qty={position['quantity']}" +
                            (f", LTP=₹{exit_ltp:.2f}" if exit_ltp else " (LTP unavailable)"))
                response = self.groww.place_order(
                    trading_symbol=position['instrument']["trading_symbol"],
                    quantity=position['quantity'],
                    validity=self.groww.VALIDITY_DAY,
                    exchange=position['instrument']["exchange"],
                    segment=position['instrument']["segment"],
                    product=self.groww.PRODUCT_MIS,
                    order_type=self.groww.ORDER_TYPE_MARKET,
                    transaction_type=self.groww.TRANSACTION_TYPE_SELL,
                    order_reference_id=f"NBES-{int(time.time() * 1000)}"
                )
                nbes_order_id = response.get('groww_order_id')
                logger.info(f"[NIFTYBEES] Exit order placed - ID: {nbes_order_id}")
            except Exception as e:
                logger.error(f"[NIFTYBEES] Exit order failed: {e}")
                exit_failed = True

        # Exit NIFTY Put position
        if 'NIFTY_PUT' in self.positions:
            try:
                position = self.positions['NIFTY_PUT']
                exit_ltp = self.get_current_price(position['instrument'])
                logger.info(f"[NIFTY PUT] Placing MARKET SELL order: Qty={position['quantity']}" +
                            (f", LTP=₹{exit_ltp:.2f}" if exit_ltp else " (LTP unavailable)"))
                response = self.groww.place_order(
                    trading_symbol=position['instrument']["trading_symbol"],
                    quantity=position['quantity'],
                    validity=self.groww.VALIDITY_DAY,
                    exchange=position['instrument']["exchange"],
                    segment=position['instrument']["segment"],
                    product=self.groww.PRODUCT_MIS,
                    order_type=self.groww.ORDER_TYPE_MARKET,
                    transaction_type=self.groww.TRANSACTION_TYPE_SELL,
                    order_reference_id=f"NPTS-{int(time.time() * 1000)}"
                )
                npts_order_id = response.get('groww_order_id')
                logger.info(f"[NIFTY PUT] Exit order placed - ID: {npts_order_id}")
            except Exception as e:
                logger.error(f"[NIFTY PUT] Exit order failed: {e}")
                exit_failed = True

        # Wait for actual exit fills to get real execution prices
        nbes_exit_fill = None
        npts_exit_fill = None
        if nbes_order_id:
            nbes_exit_fill = self._wait_for_fill(
                nbes_order_id, self.positions['NIFTYBEES']['instrument']['segment'], "NIFTYBEES EXIT",
                cancel_on_timeout=False
            )
        if npts_order_id:
            npts_exit_fill = self._wait_for_fill(
                npts_order_id, self.positions['NIFTY_PUT']['instrument']['segment'], "NIFTY PUT EXIT",
                cancel_on_timeout=False
            )

        # Log exit summary with actual execution prices and real P&L
        logger.info("=" * 60)
        logger.info(f"TRADE EXIT   |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if nbes_exit_fill is not None and 'NIFTYBEES' in self.positions:
            pos = self.positions['NIFTYBEES']
            pnl = (nbes_exit_fill - pos['entry_price']) * pos['quantity']
            pnl_pct = (nbes_exit_fill - pos['entry_price']) / pos['entry_price'] * 100 if pos['entry_price'] else 0.0
            logger.info(f"  NIFTYBEES  SELL  {pos['quantity']:>6} qty @ ₹{nbes_exit_fill:.2f}  |  Entry: ₹{pos['entry_price']:.2f}  |  P&L: ₹{pnl:>+10,.2f} ({pnl_pct:+.2f}%)")
        else:
            logger.warning("  NIFTYBEES  exit fill price unavailable — P&L not calculated")
        if npts_exit_fill is not None and 'NIFTY_PUT' in self.positions:
            pos = self.positions['NIFTY_PUT']
            pnl = (npts_exit_fill - pos['entry_price']) * pos['quantity']
            pnl_pct = (npts_exit_fill - pos['entry_price']) / pos['entry_price'] * 100 if pos['entry_price'] else 0.0
            logger.info(f"  NIFTY PUT  SELL  {pos['quantity']:>6} qty @ ₹{npts_exit_fill:.2f}  |  Entry: ₹{pos['entry_price']:.2f}  |  P&L: ₹{pnl:>+10,.2f} ({pnl_pct:+.2f}%)")
        else:
            logger.warning("  NIFTY PUT  exit fill price unavailable — P&L not calculated")
        if nbes_exit_fill is not None and npts_exit_fill is not None and 'NIFTYBEES' in self.positions and 'NIFTY_PUT' in self.positions:
            total_pnl = (
                (nbes_exit_fill - self.positions['NIFTYBEES']['entry_price']) * self.positions['NIFTYBEES']['quantity'] +
                (npts_exit_fill - self.positions['NIFTY_PUT']['entry_price']) * self.positions['NIFTY_PUT']['quantity']
            )
            logger.info(f"  NET P&L{'.' * 43} ₹{total_pnl:>+10,.2f}")
        logger.info("=" * 60)

        self.exit_executed = True
        if exit_failed:
            logger.warning("[CRITICAL] One or both exit orders failed — verify positions manually. Broker MIS auto-squareoff will close any open positions at 3:20 PM.")
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
                    if entry_start_time <= current_time_only < entry_cutoff_time:
                        attempt_word = "Placing" if not self.positions else "Retrying"
                        logger.info(f"Entry window open — {attempt_word} orders at {current_time_only.strftime('%H:%M:%S')}")
                        self.execute_entry_orders()
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

                # Sleep before next check
                time.sleep(self.config["check_interval_seconds"])

            except Exception as e:
                logger.error(f"ERROR: {e}")
                time.sleep(60)


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    logger.info("=== Intraday NIFTYBEES + NIFTY ATM Put Strategy ===")

    # Authenticate using TOTP
    groww = authenticate_with_totp()

    # Initialize and run strategy
    strategy = IntradayStrategy(groww, STRATEGY_CONFIG)
    strategy.run()