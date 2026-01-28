# RSI Trading Bot with Alpaca

A daily trading bot that uses RSI (Relative Strength Index) and volume confirmation to execute trades.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get Alpaca API keys:**
   - Sign up at [Alpaca](https://alpaca.markets/)
   - Go to Paper Trading → API Keys
   - Generate a new key pair

3. **Configure credentials** (choose one method):
   
   **Option A - Environment variables:**
   ```bash
   export ALPACA_API_KEY="your_api_key"
   export ALPACA_SECRET_KEY="your_secret_key"
   ```
   
   **Option B - Edit the script directly:**
   Replace the placeholder values in `rsi_trading_bot.py`:
   ```python
   API_KEY = "your_api_key"
   SECRET_KEY = "your_secret_key"
   ```

4. **Configure the stock symbol:**
   Edit `SYMBOL` in the script (default is "AAPL")

## Running the Bot

```bash
python rsi_trading_bot.py
```

## Trading Logic

| Condition | Action |
|-----------|--------|
| RSI < 30 AND volume surge AND shares < 500 | BUY 100 shares |
| RSI > 70 AND holding shares | SELL all shares |
| Otherwise | HOLD |

## Safety Constraints

- **Cooldown:** 3 days between trades
- **Max Position:** 500 shares
- **Trade Size:** 100 shares per transaction
- **Volume Confirmation:** Current volume must exceed 1.5× the 20-day average

## Scheduling (Optional)

To run daily after market close, use cron (Linux/Mac):
```bash
# Run at 4:30 PM EST every weekday
30 16 * * 1-5 cd /path/to/bot && python rsi_trading_bot.py >> bot.log 2>&1
```

Or Windows Task Scheduler for Windows systems.

## Files

- `rsi_trading_bot.py` - Main bot script
- `bot_state.json` - Auto-generated state file tracking last trade date
- `requirements.txt` - Python dependencies

## Paper vs Live Trading

The bot defaults to paper trading (`PAPER_TRADING = True`). Only set to `False` after thorough testing.
