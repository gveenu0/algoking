"""
Buy NIFTYBEES + NIFTY ATM Put at 9:34 AM. Sell both at 3:04 PM.

NIFTYBEES profits if market goes up
Put option protects if market falls
Both legs are always held together — if one fails to place, the other is cancelled
Net cost is the put premium (paid daily as insurance)
Uses TOTP authentication
"""

# ===== IMPORTS =====
import time
import pyotp
from datetime import datetime, time as dt_time, timedelta
from typing import Optional
from growwapi import GrowwAPI

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
    "entry_time": "09:34:00",
    "exit_time": "15:04:00",

    # Monitoring
    "check_interval_seconds": 30,

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
        totp = pyotp.TOTP(AUTH_CONFIG["totp_secret"]).now()
        access_token = GrowwAPI.get_access_token(
            api_key=AUTH_CONFIG["api_key"],
            totp=totp
        )
        print("Authentication successful using TOTP")
        return GrowwAPI(access_token)
    except Exception as e:
        print(f"Authentication failed: {e}")
        raise


def get_nearest_weekly_expiry(groww: GrowwAPI, underlying: str = "NIFTY") -> Optional[str]:
    """Get nearest weekly expiry date for NIFTY options.

    NOTE: groww.get_expiries() does NOT exist in the Groww Python SDK.
    The correct approach per the SDK docs is to use get_all_instruments()
    which returns a DataFrame with an 'expiry_date' column, then filter.
    Reference: https://groww.in/trade-api/docs/python-sdk/instruments
    """
    try:
        instruments_df = groww.get_all_instruments()

        # Filter for the underlying's Put options in FNO segment
        nifty_options = instruments_df[
            (instruments_df['underlying_symbol'] == underlying) &
            (instruments_df['segment'] == 'FNO') &
            (instruments_df['instrument_type'] == 'PE')
        ]

        if nifty_options.empty:
            print(f"No FNO instruments found for {underlying}")
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
            print(f"Nearest weekly expiry: {nearest_expiry}")
            return nearest_expiry

        return None

    except Exception as e:
        print(f"Error fetching expiries: {e}")
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
        print(f"NIFTY current price: ₹{current_price:.2f} | ATM strike: {atm_strike}")
        return int(atm_strike)

    except Exception as e:
        print(f"Error calculating ATM strike: {e}")
        return None


_MONTH_ABBR = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

def format_expiry_for_symbol(expiry_date: str) -> str:
    """Convert YYYY-MM-DD to DDMmmYY format for Groww symbol.
    Uses a hardcoded month map to avoid locale-dependent strftime('%b') output.
    Reference: instrument CSV shows NSE-NIFTY-27Mar25-29050-PE (title-case month)
    """
    dt = datetime.strptime(expiry_date, '%Y-%m-%d')
    return f"{dt.day:02d}{_MONTH_ABBR[dt.month]}{dt.strftime('%y')}"


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

        # Fetch NIFTYBEES instrument
        print("Fetching NIFTYBEES instrument...")
        try:
            self.niftybees_instrument = groww.get_instrument_by_exchange_and_trading_symbol(
                exchange=config["equity_exchange"],
                trading_symbol=config["niftybees_symbol"]
            )
            print(f"NIFTYBEES instrument loaded: {self.niftybees_instrument['trading_symbol']}")
        except Exception as e:
            print(f"Failed to load NIFTYBEES instrument: {e}")
            raise

        # Option instrument will be fetched at entry time
        self.option_instrument = None

    def get_option_instrument(self) -> bool:
        """Fetch NIFTY weekly ATM Put option instrument"""
        try:
            print("\nFetching NIFTY option details...")

            # Get nearest weekly expiry
            expiry_date = get_nearest_weekly_expiry(self.groww, self.config["nifty_underlying"])
            if not expiry_date:
                print("Failed to get expiry date")
                return False

            # Get ATM strike
            atm_strike = get_atm_strike(self.groww, self.config["nifty_underlying"])
            if not atm_strike:
                print("Failed to calculate ATM strike")
                return False

            # Format expiry for Groww symbol
            expiry_formatted = format_expiry_for_symbol(expiry_date)

            # Construct Groww symbol for Put option
            option_symbol = f"{self.config['option_exchange']}-{self.config['nifty_underlying']}-{expiry_formatted}-{atm_strike}-PE"
            print(f"Option symbol: {option_symbol}")

            # Fetch option instrument
            self.option_instrument = self.groww.get_instrument_by_groww_symbol(
                groww_symbol=option_symbol
            )
            print(f"Option instrument loaded: {self.option_instrument['trading_symbol']}")
            return True

        except Exception as e:
            print(f"Error fetching option instrument: {e}")
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
            print(f"Error fetching price for {instrument['trading_symbol']}: {e}")
            return None

    def _cancel_or_exit(self, instrument: dict, order_id: str, quantity: int, label: str) -> None:
        """Cancel an open order. If already filled, place a market SELL to close it.
        Used to roll back one leg when the other leg of the hedge pair fails.
        """
        try:
            self.groww.cancel_order(
                segment=instrument["segment"],
                groww_order_id=order_id
            )
            print(f"[{label}] Order {order_id} cancelled successfully")
        except Exception as cancel_err:
            print(f"[{label}] Cancel failed ({cancel_err}) — placing market SELL to close filled position")
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
                print(f"[{label}] Market SELL placed to close position")
            except Exception as sell_err:
                print(f"[{label}] CRITICAL: Failed to cancel or exit position: {sell_err}")

    def execute_entry_orders(self) -> None:
        """Execute entry orders at 9:34 AM.
        NIFTYBEES and NIFTY PUT must both succeed — they are a hedge pair.
        If NIFTY PUT fails after NIFTYBEES is placed, NIFTYBEES is cancelled/exited.
        """
        print("\n=== EXECUTING ENTRY ORDERS ===")

        # Fetch option instrument
        if not self.get_option_instrument():
            print("Failed to fetch option instrument - skipping entry")
            return

        # --- Step 1: Place NIFTYBEES order ---
        niftybees_price = self.get_current_price(self.niftybees_instrument)
        if niftybees_price is None:
            print("[NIFTYBEES] Failed to get current price — aborting entry")
            return

        try:
            print(f"[NIFTYBEES] Placing BUY order: Qty={self.config['niftybees_quantity']}, Price=₹{niftybees_price:.2f}")
            response = self.groww.place_order(
                trading_symbol=self.niftybees_instrument["trading_symbol"],
                quantity=self.config["niftybees_quantity"],
                price=niftybees_price,
                validity=self.groww.VALIDITY_DAY,
                exchange=self.niftybees_instrument["exchange"],
                segment=self.niftybees_instrument["segment"],
                product=self.groww.PRODUCT_MIS,
                order_type=self.groww.ORDER_TYPE_LIMIT,
                transaction_type=self.groww.TRANSACTION_TYPE_BUY,
                order_reference_id=f"NBEE-{int(time.time() * 1000)}"
            )
            niftybees_order_id = response.get('groww_order_id')
            print(f"[NIFTYBEES] Order placed - ID: {niftybees_order_id}")
        except Exception as e:
            print(f"[NIFTYBEES] Order failed: {e} — aborting entry")
            return

        # --- Step 2: Place NIFTY PUT order ---
        option_price = self.get_current_price(self.option_instrument)
        if option_price is None:
            print("[NIFTY PUT] Failed to get current price — cancelling NIFTYBEES order")
            self._cancel_or_exit(self.niftybees_instrument, niftybees_order_id,
                                 self.config["niftybees_quantity"], "NIFTYBEES")
            print("[ENTRY ABORTED] Hedge pair must be held together")
            return

        try:
            print(f"[NIFTY PUT] Placing BUY order: Qty={self.config['nifty_option_quantity']}, Price=₹{option_price:.2f}")
            response = self.groww.place_order(
                trading_symbol=self.option_instrument["trading_symbol"],
                quantity=self.config["nifty_option_quantity"],
                price=option_price,
                validity=self.groww.VALIDITY_DAY,
                exchange=self.option_instrument["exchange"],
                segment=self.option_instrument["segment"],
                product=self.groww.PRODUCT_MIS,
                order_type=self.groww.ORDER_TYPE_LIMIT,
                transaction_type=self.groww.TRANSACTION_TYPE_BUY,
                order_reference_id=f"NPUT-{int(time.time() * 1000)}"
            )
            put_order_id = response.get('groww_order_id')
            print(f"[NIFTY PUT] Order placed - ID: {put_order_id}")
        except Exception as e:
            print(f"[NIFTY PUT] Order failed: {e} — cancelling NIFTYBEES order")
            self._cancel_or_exit(self.niftybees_instrument, niftybees_order_id,
                                 self.config["niftybees_quantity"], "NIFTYBEES")
            print("[ENTRY ABORTED] Hedge pair must be held together")
            return

        # Both legs placed — record positions together
        self.positions['NIFTYBEES'] = {
            'instrument': self.niftybees_instrument,
            'quantity': self.config['niftybees_quantity'],
            'entry_price': niftybees_price,
            'order_id': niftybees_order_id
        }
        self.positions['NIFTY_PUT'] = {
            'instrument': self.option_instrument,
            'quantity': self.config['nifty_option_quantity'],
            'entry_price': option_price,
            'order_id': put_order_id
        }
        self.entry_executed = True
        print("Entry orders execution completed — hedge pair active")

    def execute_exit_orders(self) -> None:
        """Execute exit orders at 3:04 PM"""
        print("\n=== EXECUTING EXIT ORDERS ===")

        if not self.positions:
            print("No positions to exit")
            self.exit_executed = True
            return

        # Exit NIFTYBEES position
        if 'NIFTYBEES' in self.positions:
            try:
                position = self.positions['NIFTYBEES']
                exit_price = self.get_current_price(position['instrument'])

                if exit_price is None:
                    # Price fetch failed — fall back to market order to ensure position is closed
                    print("[NIFTYBEES] Failed to get exit price — placing market SELL as fallback")
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
                    print(f"[NIFTYBEES] Market exit order placed - ID: {response.get('groww_order_id')}")
                else:
                    print(f"[NIFTYBEES] Placing SELL order: Qty={position['quantity']}, Price=₹{exit_price:.2f}")

                    response = self.groww.place_order(
                        trading_symbol=position['instrument']["trading_symbol"],
                        quantity=position['quantity'],
                        price=exit_price,
                        validity=self.groww.VALIDITY_DAY,
                        exchange=position['instrument']["exchange"],
                        segment=position['instrument']["segment"],
                        product=self.groww.PRODUCT_MIS,
                        order_type=self.groww.ORDER_TYPE_LIMIT,
                        transaction_type=self.groww.TRANSACTION_TYPE_SELL,
                        order_reference_id=f"NBES-{int(time.time() * 1000)}"
                    )

                    order_id = response.get('groww_order_id')
                    print(f"[NIFTYBEES] Exit order placed - ID: {order_id}")

                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    pnl_percent = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                    print(f"[NIFTYBEES] P&L: ₹{pnl:.2f} ({pnl_percent:.2f}%)")

            except Exception as e:
                print(f"[NIFTYBEES] Exit order failed: {e}")

        # Exit NIFTY Put position
        if 'NIFTY_PUT' in self.positions:
            try:
                position = self.positions['NIFTY_PUT']
                exit_price = self.get_current_price(position['instrument'])

                if exit_price is None:
                    # Price fetch failed — fall back to market order to ensure position is closed
                    print("[NIFTY PUT] Failed to get exit price — placing market SELL as fallback")
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
                    print(f"[NIFTY PUT] Market exit order placed - ID: {response.get('groww_order_id')}")
                else:
                    print(f"[NIFTY PUT] Placing SELL order: Qty={position['quantity']}, Price=₹{exit_price:.2f}")

                    response = self.groww.place_order(
                        trading_symbol=position['instrument']["trading_symbol"],
                        quantity=position['quantity'],
                        price=exit_price,
                        validity=self.groww.VALIDITY_DAY,
                        exchange=position['instrument']["exchange"],
                        segment=position['instrument']["segment"],
                        product=self.groww.PRODUCT_MIS,
                        order_type=self.groww.ORDER_TYPE_LIMIT,
                        transaction_type=self.groww.TRANSACTION_TYPE_SELL,
                        order_reference_id=f"NPTS-{int(time.time() * 1000)}"
                    )

                    order_id = response.get('groww_order_id')
                    print(f"[NIFTY PUT] Exit order placed - ID: {order_id}")

                    pnl = (exit_price - position['entry_price']) * position['quantity']
                    pnl_percent = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                    print(f"[NIFTY PUT] P&L: ₹{pnl:.2f} ({pnl_percent:.2f}%)")

            except Exception as e:
                print(f"[NIFTY PUT] Exit order failed: {e}")

        self.exit_executed = True
        print("Exit orders execution completed")

    def run(self) -> None:
        """Single-day execution loop.

        Designed for Groww Cloud: runs once for the trading day and exits cleanly.
        Groww Cloud's scheduler handles restarting the script the next morning.
        """
        print("Intraday strategy started - waiting for entry time")

        entry_time = datetime.strptime(self.config["entry_time"], "%H:%M:%S").time()
        exit_time = datetime.strptime(self.config["exit_time"], "%H:%M:%S").time()

        while True:
            try:
                current_time = datetime.now()

                # Check market hours
                if not is_market_open(current_time):
                    print(f"Market closed at {current_time.strftime('%H:%M:%S')} - waiting...")
                    time.sleep(60)
                    continue

                current_time_only = current_time.time()

                # Execute entry at 9:34 AM
                if not self.entry_executed and current_time_only >= entry_time:
                    # Check if we're within 1 minute of entry time
                    entry_window_end = (datetime.combine(datetime.today(), entry_time) + timedelta(minutes=1)).time()

                    if current_time_only <= entry_window_end:
                        self.execute_entry_orders()
                    else:
                        print(f"Entry time window missed (after {entry_window_end.strftime('%H:%M:%S')})")
                        self.entry_executed = True

                # Execute exit at 3:04 PM
                if self.entry_executed and not self.exit_executed and current_time_only >= exit_time:
                    # Always attempt to exit — even if window is missed, positions must be closed
                    # before broker MIS auto-squareoff at 3:20 PM
                    exit_window_end = (datetime.combine(datetime.today(), exit_time) + timedelta(minutes=1)).time()
                    if current_time_only > exit_window_end:
                        print(f"[WARN] Exit time window missed — forcing exit now to avoid auto-squareoff penalty")
                    self.execute_exit_orders()

                # Exit cleanly once the day's work is done.
                # Groww Cloud scheduler will restart the script next morning.
                if self.exit_executed:
                    print("\n=== Strategy completed for the day — exiting ===")
                    return

                # Sleep before next check
                time.sleep(self.config["check_interval_seconds"])

            except Exception as e:
                print(f"ERROR: {e}")
                time.sleep(60)


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    print("=== Intraday NIFTYBEES + NIFTY ATM Put Strategy ===\n")

    # Authenticate using TOTP
    groww = authenticate_with_totp()

    # Initialize and run strategy
    strategy = IntradayStrategy(groww, STRATEGY_CONFIG)
    strategy.run()