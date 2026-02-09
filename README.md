# 🤖 HFT Trading Bot

A high-frequency algorithmic trading bot that runs on **Alpaca** paper trading via **GitHub Actions**. It scans multiple stocks every 60 seconds using fast technical indicators, automatically entering and exiting positions to capture short-term moves.

---

## How It Works

The bot uses 5-minute bars and a **multi-indicator scoring system** to decide when to buy and sell. A trade only triggers when enough indicators agree — no single indicator can force a trade.

### Buy Signal Scoring

| Indicator | Condition | Points |
|-----------|-----------|--------|
| RSI(5) | < 25 (oversold) | +2 |
| RSI(5) | < 35 (approaching oversold) | +1 |
| VWAP | Price below VWAP | +2 |
| VWAP | Price within 0.2% of VWAP | +1 |
| EMA | 8-EMA crosses above 21-EMA | +2 |
| EMA | 8-EMA > 21-EMA (uptrend) | +1 |
| MACD | Histogram flips positive | +1 |
| Bollinger | Price below lower band | +2 |
| Volume | Spike > 2x average | +1 |
| Momentum | Price rising last 3 bars | +1 |

**Buy threshold: 3 points**

### Sell Signal Scoring

| Indicator | Condition | Points |
|-----------|-----------|--------|
| RSI(5) | > 75 (overbought) | +2 |
| VWAP | Price above VWAP | +2 |
| EMA | 8-EMA crosses below 21-EMA | +2 |
| Bollinger | Price above upper band | +2 |
| Trailing stop | Price hits trailing stop | +1 |
| Stop-loss | Price hits hard stop (1x ATR) | +3 |
| Take-profit | Price hits target (1.5x ATR) | +2 |

**Sell threshold: 2 points**

---

## Risk Management

| Parameter | Value |
|-----------|-------|
| Risk per trade | 2% of portfolio |
| Max portfolio heat | 20% total risk |
| Stop-loss | 1x ATR below entry |
| Take-profit | 1.5x ATR above entry |
| Trailing stop | Follows at 1x ATR below price |
| Max position per symbol | 30% of portfolio |
| Max open positions | 3 simultaneous |
| Max shares per symbol | 2,000 |

Position sizes are calculated dynamically so that if a stop-loss is hit, you lose exactly 2% of your portfolio — not a random amount.

---

## Watchlist

The bot scans these stocks by default:

- **AAPL** — Apple
- **MSFT** — Microsoft
- **TSLA** — Tesla
- **NVDA** — NVIDIA
- **AMD** — AMD

Edit the `symbols` list in `BotConfig` to change these.

---

## Setup

### 1. Alpaca Account

1. Sign up at [alpaca.markets](https://alpaca.markets) (free)
2. Switch to **Paper Trading** in the dashboard
3. Generate **API keys** (right sidebar → "Generate New Keys")
4. Save both the API Key ID and Secret Key

### 2. GitHub Repository

Create a repo with this structure:

```
your-repo/
├── .github/
│   └── workflows/
│       └── rsi_bot.yml
├── rsi_trading_bot.py
└── README.md
```

### 3. GitHub Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** and add:

| Secret Name | Value |
|-------------|-------|
| `ALPACA_API_KEY` | Your Alpaca API Key ID |
| `ALPACA_SECRET_KEY` | Your Alpaca Secret Key |
| `GH_PAT` | GitHub Personal Access Token (for job chaining) |

### 4. GitHub Personal Access Token (for continuous running)

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **"Generate new token (classic)"**
3. Name it `bot-chain`, check the **`repo`** scope
4. Copy the token and add it as the `GH_PAT` secret above

### 5. Launch

- **Automatic:** The bot starts every weekday at 9:25 AM ET
- **Manual:** Go to **Actions** tab → **"HFT Trading Bot"** → **"Run workflow"**

---

## Running Locally

```bash
# Install dependencies
pip install alpaca-py pytz

# Set your API keys
export ALPACA_API_KEY=your_key_here
export ALPACA_SECRET_KEY=your_secret_here

# Single run
python rsi_trading_bot.py

# Continuous loop (every 60 seconds)
python rsi_trading_bot.py --loop

# Backtest (default 30 days)
python rsi_trading_bot.py --backtest

# Backtest with custom period
python rsi_trading_bot.py --backtest --days 90
```

On Windows, use `set` instead of `export`:

```cmd
set ALPACA_API_KEY=your_key_here
set ALPACA_SECRET_KEY=your_secret_here
python rsi_trading_bot.py
```

---

## GitHub Actions Behavior

| Detail | Value |
|--------|-------|
| Schedule | Auto-starts 9:25 AM ET, Mon–Fri |
| Loop interval | Every 60 seconds |
| Job duration | ~5 hours per run |
| Chaining | Auto-triggers next job until 4 PM ET |
| State persistence | `bot_state.json` cached between runs |
| Logs | Saved as artifacts for 7 days |

### Usage Warning

Continuous mode uses **~350 minutes/day** of GitHub Actions. The free tier includes **2,000 minutes/month**, giving you roughly **5–6 trading days** before hitting the limit. Options:

- **GitHub Pro** ($4/month) — 3,000 minutes
- **Reduce frequency** — edit the loop interval in `BotConfig`
- **Run locally** — use `--loop` on your own machine (free, unlimited)

---

## Monitoring

### GitHub

- **Actions** tab → click running job → expand **"run-bot"** for live logs
- Download log artifacts after each run

### Alpaca

- Go to [app.alpaca.markets](https://app.alpaca.markets)
- Switch to **Paper Trading**
- Check **Portfolio** for balance changes
- Check **Orders** for trade history
- Check **Positions** for current holdings

### Log Output Example

```
12:30:01 | INFO | --- TSLA ---
12:30:01 | INFO | TSLA — 195 bars (5min), latest $248.52
12:30:01 | INFO | TSLA BUY score: 5/3
12:30:01 | INFO |   RSI(5) 22.4 < 25 [+2]
12:30:01 | INFO |   Price $248.52 below VWAP $251.30 (-1.11%) [+2]
12:30:01 | INFO |   Volume spike 892450 vs avg 341200 [+1]
12:30:01 | INFO | TSLA BUY 45 @ $248.52 | Stop $245.18 | Target $253.53
12:30:01 | INFO | ORDER BUY 45 x TSLA — ID abc123
```

---

## Configuration

All settings are in the `BotConfig` dataclass at the top of `rsi_trading_bot.py`. Key parameters to tune:

```python
# Trade more/fewer stocks
symbols: list[str] = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD"]

# Adjust risk
risk_per_trade_pct: float = 0.02      # 2% risk per trade
max_portfolio_heat_pct: float = 0.20   # 20% max exposure

# Adjust signal sensitivity
buy_score_threshold: int = 3           # Lower = more trades
sell_score_threshold: int = 2          # Lower = quicker exits

# Adjust speed
bar_timeframe: str = "5min"            # "1min", "5min", or "15min"
loop_interval_seconds: int = 60        # Seconds between evaluations
```

---

## Files

| File | Purpose |
|------|---------|
| `rsi_trading_bot.py` | Main bot script |
| `.github/workflows/rsi_bot.yml` | GitHub Actions workflow |
| `bot_state.json` | Persisted state (auto-created) |
| `bot.log` | Execution logs (auto-created) |

---

## Disclaimer

This bot is for **paper trading and educational purposes only**. Algorithmic trading involves significant risk. Past backtest performance does not guarantee future results. Do not use this with real money without thorough testing and understanding of the risks involved.
