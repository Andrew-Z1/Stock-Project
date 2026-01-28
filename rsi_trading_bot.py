"""
RSI Trading Bot using Alpaca API
Executes daily after market close to analyze price/volume and trade based on RSI signals.

Constraints:
- 3-day cooldown between trades
- Max position: 500 shares
- Trade size: 100 shares per transaction
- RSI < 30 + volume surge = BUY
- RSI > 70 = SELL
"""

import os
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# Alpaca API credentials (set these as environment variables or replace directly)
API_KEY = os.getenv("ALPACA_API_KEY", "your_api_key_here")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "your_secret_key_here")

# Set to True for paper trading, False for live trading
PAPER_TRADING = True

# Trading parameters
SYMBOL = "AAPL"  # Stock to trade
MAX_SHARES = 500
TRADE_SIZE = 100
COOLDOWN_DAYS = 3
RSI_PERIOD = 14
VOLUME_LOOKBACK = 20
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
VOLUME_SURGE_MULTIPLIER = 1.5

# State file to persist last trade date across runs
STATE_FILE = "bot_state.json"


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state():
    """Load bot state from file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_trade_date": None}


def save_state(state):
    """Save bot state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# =============================================================================
# RSI CALCULATION
# =============================================================================

def calculate_rsi(prices, period=14):
    """
    Calculate RSI from a list of closing prices.
    
    Args:
        prices: List of closing prices (oldest to newest)
        period: RSI period (default 14)
    
    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1:
        return None
    
    gains = []
    losses = []
    
    # Calculate gains and losses for the last 'period' price changes
    for i in range(1, period + 1):
        change = prices[-i] - prices[-(i + 1)]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)


# =============================================================================
# VOLUME ANALYSIS
# =============================================================================

def check_volume_surge(volumes, multiplier=1.5):
    """
    Check if current volume exceeds average by the specified multiplier.
    
    Args:
        volumes: List of volumes (oldest to newest)
        multiplier: Volume surge threshold multiplier
    
    Returns:
        Tuple of (is_surge, current_volume, avg_volume)
    """
    if len(volumes) < 2:
        return False, 0, 0
    
    avg_volume = sum(volumes) / len(volumes)
    current_volume = volumes[-1]
    is_surge = current_volume > (avg_volume * multiplier)
    
    return is_surge, current_volume, avg_volume


# =============================================================================
# TRADING FUNCTIONS
# =============================================================================

def get_current_position(trading_client, symbol):
    """Get current shares held for a symbol."""
    try:
        position = trading_client.get_open_position(symbol)
        return int(float(position.qty))
    except Exception:
        return 0


def execute_buy(trading_client, symbol, qty):
    """Execute a market buy order."""
    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
    order = trading_client.submit_order(order_request)
    return order


def execute_sell(trading_client, symbol, qty):
    """Execute a market sell order."""
    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    order = trading_client.submit_order(order_request)
    return order


# =============================================================================
# MAIN BOT LOGIC
# =============================================================================

def run_trading_bot():
    """Main bot execution logic."""
    print("=" * 60)
    print(f"RSI Trading Bot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbol: {SYMBOL} | Paper Trading: {PAPER_TRADING}")
    print("=" * 60)
    
    # Initialize Alpaca clients
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_TRADING)
    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    
    # Load state
    state = load_state()
    last_trade_date = state.get("last_trade_date")
    current_date = datetime.now().date()
    
    # Check cooldown
    if last_trade_date:
        last_trade = datetime.strptime(last_trade_date, "%Y-%m-%d").date()
        days_since_trade = (current_date - last_trade).days
        if days_since_trade < COOLDOWN_DAYS:
            print(f"\n⏸️  Cooldown active - {COOLDOWN_DAYS - days_since_trade} days remaining")
            print("No trade allowed.")
            return
    
    # Fetch historical data (last 20 trading days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=35)  # Extra buffer for weekends/holidays
    
    print(f"\nFetching {VOLUME_LOOKBACK} days of data...")
    
    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    
    bars = data_client.get_stock_bars(request)
    bar_data = bars[SYMBOL]
    
    # Extract prices and volumes
    prices = [bar.close for bar in bar_data]
    volumes = [bar.volume for bar in bar_data]
    
    print(f"Retrieved {len(prices)} data points")
    
    # Check minimum data requirement
    if len(prices) < RSI_PERIOD + 1:
        print(f"\n❌ Not enough data for RSI calculation")
        print(f"Need at least {RSI_PERIOD + 1} data points, got {len(prices)}")
        return
    
    # Calculate RSI
    rsi = calculate_rsi(prices, RSI_PERIOD)
    print(f"\n📊 RSI ({RSI_PERIOD}-day): {rsi}")
    
    # Check volume surge
    volume_surge, current_vol, avg_vol = check_volume_surge(volumes[-VOLUME_LOOKBACK:], VOLUME_SURGE_MULTIPLIER)
    print(f"📈 Volume: {current_vol:,.0f} (Avg: {avg_vol:,.0f}) - Surge: {'Yes' if volume_surge else 'No'}")
    
    # Get current position
    shares_held = get_current_position(trading_client, SYMBOL)
    print(f"💼 Current Position: {shares_held} shares")
    
    # Trading decision
    print("\n" + "-" * 40)
    
    if rsi < RSI_OVERSOLD and volume_surge and shares_held < MAX_SHARES:
        # BUY signal
        buy_qty = min(TRADE_SIZE, MAX_SHARES - shares_held)
        print(f"🟢 BUY SIGNAL: RSI oversold at {rsi} with volume confirmation")
        print(f"Executing buy order for {buy_qty} shares...")
        
        try:
            order = execute_buy(trading_client, SYMBOL, buy_qty)
            print(f"✅ Order submitted: {order.id}")
            state["last_trade_date"] = current_date.strftime("%Y-%m-%d")
            save_state(state)
        except Exception as e:
            print(f"❌ Order failed: {e}")
    
    elif rsi > RSI_OVERBOUGHT and shares_held > 0:
        # SELL signal
        print(f"🔴 SELL SIGNAL: RSI overbought at {rsi}")
        print(f"Executing sell order for {shares_held} shares...")
        
        try:
            order = execute_sell(trading_client, SYMBOL, shares_held)
            print(f"✅ Order submitted: {order.id}")
            state["last_trade_date"] = current_date.strftime("%Y-%m-%d")
            save_state(state)
        except Exception as e:
            print(f"❌ Order failed: {e}")
    
    else:
        # HOLD
        print(f"⚪ HOLD - RSI at {rsi}, no action taken")
        if rsi < RSI_OVERSOLD and not volume_surge:
            print("   (RSI oversold but no volume confirmation)")
        elif rsi < RSI_OVERSOLD and shares_held >= MAX_SHARES:
            print("   (RSI oversold but max position reached)")
        elif rsi > RSI_OVERBOUGHT and shares_held == 0:
            print("   (RSI overbought but no position to sell)")
    
    # Final status
    print("\n" + "=" * 60)
    final_shares = get_current_position(trading_client, SYMBOL)
    print(f"Final Position: {final_shares} shares")
    print(f"Last Trade Date: {state.get('last_trade_date', 'None')}")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_trading_bot()
