"""
High-Frequency RSI Trading Bot v3

Designed to run every 1-5 minutes via GitHub Actions or locally.
Uses minute-level bars with fast indicators for scalping.

Strategy:
  - 5-minute bars for signal generation
  - Fast RSI (5-period) + VWAP + EMA crossover (8/21) + momentum
  - Scalps small moves: tight stops, quick profits
  - Multiple trades per day, both long and short

Signal scoring (BUY):
  +2  RSI(5) < 25
  +1  RSI(5) < 35
  +2  Price below VWAP (mean reversion)
  +1  Price within 0.2% of VWAP
  +2  EMA(8) crosses above EMA(21)
  +1  EMA(8) > EMA(21) (trend confirmation)
  +2  Bollinger Band bounce (price < lower band)
  +1  Volume spike (2x average)
  +1  Positive momentum (price rising last 3 bars)
  ---
  Threshold: 3

Signal scoring (SELL):
  +2  RSI(5) > 75
  +2  Price above VWAP
  +2  EMA(8) crosses below EMA(21)
  +2  Price > upper Bollinger Band
  +1  Trailing stop hit
  +3  Stop-loss hit
  +2  Take-profit hit
  ---
  Threshold: 2

Risk management:
  - Risk 2% per trade
  - Stop: 1x ATR (tight for scalping)
  - Target: 1.5x ATR (quick profits)
  - Trailing stop follows at 1x ATR
  - Max 20% portfolio heat
  - Max 3 open positions

Usage:
  python rsi_trading_bot.py                        # Live run
  python rsi_trading_bot.py --backtest             # Backtest 30 days
  python rsi_trading_bot.py --backtest --days 90   # Backtest 90 days
  python rsi_trading_bot.py --loop                 # Run continuously every 60s
"""

import os
import sys
import json
import math
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BotConfig:

    # Alpaca credentials
    api_key: str = os.getenv("ALPACA_API_KEY", "PKOCSTWQW72DB2D5M2PX2I6V5U")
    secret_key: str = os.getenv("ALPACA_SECRET_KEY", "AiyCrsY9XSzhnAzeAnL6L1NbsNeosm8khYGxDtPZMLan")
    paper_trading: bool = True

    # Watchlist — multiple symbols for more opportunities
    symbols: list[str] = field(default_factory=lambda: [
        # Tech
        "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOG", "AMZN", "META",
        "AVGO", "CRM", "ORCL", "ADBE", "INTC", "QCOM", "MU", "ANET",
        # Finance
        "JPM", "GS", "MS", "BAC", "V", "MA", "BLK", "C",
        # Healthcare
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
        # Consumer
        "WMT", "COST", "HD", "NKE", "SBUX", "MCD", "KO", "PEP",
        # Energy & Industrial
        "XOM", "CVX", "CAT", "BA", "GE", "UPS", "DE", "LMT",
        # Other movers
        "NFLX", "DIS", "PYPL", "SQ", "COIN", "RIVN", "PLTR", "SOFI",
    ])

    # ---- Risk Management ----
    risk_per_trade_pct: float = 0.01        # 1% risk per trade (more symbols = spread risk)
    max_portfolio_heat_pct: float = 0.40    # 40% max — room for many positions
    atr_stop_multiplier: float = 1.0        # Tight stop for scalping
    atr_profit_multiplier: float = 1.5      # Quick take-profit
    trailing_stop_enabled: bool = True
    max_shares_per_symbol: int = 2000
    max_position_pct: float = 0.10          # Max 10% per symbol (diversified)
    max_open_positions: int = 999          # No limit

    # ---- Fast Indicator Periods (tuned for minute bars) ----
    rsi_period: int = 5             # Fast RSI
    ema_fast: int = 8               # Fast EMA
    ema_slow: int = 21              # Slow EMA
    macd_fast: int = 5              # Fast MACD
    macd_slow: int = 13             # Slow MACD
    macd_signal: int = 4            # MACD signal
    bb_period: int = 15             # Bollinger Bands
    bb_std_dev: float = 2.0
    atr_period: int = 10            # ATR
    vwap_period: int = 50           # Rolling VWAP approximation
    momentum_bars: int = 3          # Bars to check momentum

    # ---- Signal Thresholds ----
    rsi_oversold: float = 25.0
    rsi_approaching_oversold: float = 35.0
    rsi_overbought: float = 75.0
    volume_surge_multiplier: float = 2.0    # Higher for intraday
    buy_score_threshold: int = 4
    sell_score_threshold: int = 2

    # ---- Timeframe ----
    bar_timeframe: str = "5min"     # 1min, 5min, or 15min
    lookback_bars: int = 200        # Number of bars to fetch
    cooldown_days: int = 0          # No cooldown for HFT

    # ---- Operational ----
    state_file: str = "bot_state.json"
    log_file: str = "bot.log"
    dry_run: bool = False
    loop_interval_seconds: int = 60     # For --loop mode
    backtest_days: int = 30


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("hft_bot")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def get_symbol_state(state: dict, symbol: str) -> dict:
    return state.get("symbols", {}).get(symbol, {})


def update_symbol_state(state: dict, symbol: str, updates: dict) -> None:
    state.setdefault("symbols", {}).setdefault(symbol, {})
    state["symbols"][symbol].update(updates)


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def _safe(val: float) -> float:
    return val if not math.isnan(val) else 0.0


def ema(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return [float("nan")] * len(values)
    k = 2.0 / (period + 1)
    result = [float("nan")] * (period - 1)
    sma = sum(values[:period]) / period
    result.append(sma)
    for i in range(period, len(values)):
        sma = values[i] * k + result[-1] * (1 - k)
        result.append(sma)
    return result


def calculate_rsi(prices: list[float], period: int = 5) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    seed_gains = [max(d, 0) for d in deltas[:period]]
    seed_losses = [max(-d, 0) for d in deltas[:period]]
    avg_gain = sum(seed_gains) / period
    avg_loss = sum(seed_losses) / period
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def calculate_macd(
    prices: list[float], fast: int = 5, slow: int = 13, signal: int = 4
) -> tuple[list[float], list[float], list[float]]:
    ema_f = ema(prices, fast)
    ema_s = ema(prices, slow)
    macd_line = []
    for f, s in zip(ema_f, ema_s):
        if math.isnan(f) or math.isnan(s):
            macd_line.append(float("nan"))
        else:
            macd_line.append(f - s)
    valid_macd = [v for v in macd_line if not math.isnan(v)]
    signal_ema = ema(valid_macd, signal)
    signal_line = [float("nan")] * (len(macd_line) - len(signal_ema)) + signal_ema
    histogram = []
    for m, s in zip(macd_line, signal_line):
        if math.isnan(m) or math.isnan(s):
            histogram.append(float("nan"))
        else:
            histogram.append(m - s)
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: list[float], period: int = 15, std_dev: float = 2.0
) -> tuple[list[float], list[float], list[float]]:
    upper, middle, lower = [], [], []
    for i in range(len(prices)):
        if i < period - 1:
            upper.append(float("nan"))
            middle.append(float("nan"))
            lower.append(float("nan"))
        else:
            window = prices[i - period + 1 : i + 1]
            mean = sum(window) / period
            variance = sum((x - mean) ** 2 for x in window) / period
            std = variance ** 0.5
            middle.append(mean)
            upper.append(mean + std_dev * std)
            lower.append(mean - std_dev * std)
    return upper, middle, lower


def calculate_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 10
) -> list[float]:
    if len(highs) < 2:
        return [float("nan")] * len(highs)
    true_ranges = [highs[0] - lows[0]]
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)
    atr_values = [float("nan")] * (period - 1)
    first_atr = sum(true_ranges[:period]) / period
    atr_values.append(first_atr)
    for i in range(period, len(true_ranges)):
        atr_val = (atr_values[-1] * (period - 1) + true_ranges[i]) / period
        atr_values.append(atr_val)
    return atr_values


def calculate_vwap(
    closes: list[float], volumes: list[float], highs: list[float], lows: list[float]
) -> list[float]:
    """Rolling VWAP — cumulative typical_price * volume / cumulative volume."""
    vwap = []
    cum_tp_vol = 0.0
    cum_vol = 0.0
    for i in range(len(closes)):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_tp_vol += typical_price * volumes[i]
        cum_vol += volumes[i]
        if cum_vol > 0:
            vwap.append(cum_tp_vol / cum_vol)
        else:
            vwap.append(closes[i])
    return vwap


def check_volume_surge(volumes: list[float], multiplier: float = 2.0) -> tuple[bool, float, float]:
    if len(volumes) < 2:
        return False, 0.0, 0.0
    current = volumes[-1]
    prior = volumes[:-1]
    avg_prior = sum(prior) / len(prior)
    return current > avg_prior * multiplier, current, avg_prior


def check_momentum(prices: list[float], bars: int = 3) -> tuple[bool, float]:
    """Check if price has been rising over the last N bars."""
    if len(prices) < bars + 1:
        return False, 0.0
    change = prices[-1] - prices[-(bars + 1)]
    pct = (change / prices[-(bars + 1)]) * 100
    rising = all(prices[-i] >= prices[-(i + 1)] for i in range(1, bars + 1))
    return rising, round(pct, 3)


# =============================================================================
# SIGNAL SCORING
# =============================================================================

@dataclass
class SignalReport:
    symbol: str
    side: str
    score: int
    details: list[str]
    rsi: float
    price: float
    vwap: float
    atr: float
    bb_upper: float
    bb_lower: float
    suggested_qty: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0


def score_buy_signal(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    cfg: BotConfig,
) -> SignalReport:

    price = closes[-1]
    details = []
    score = 0

    # RSI
    rsi = calculate_rsi(closes, cfg.rsi_period) or 50.0
    if rsi < cfg.rsi_oversold:
        score += 2
        details.append(f"RSI({cfg.rsi_period}) {rsi:.1f} < {cfg.rsi_oversold} [+2]")
    elif rsi < cfg.rsi_approaching_oversold:
        score += 1
        details.append(f"RSI({cfg.rsi_period}) {rsi:.1f} < {cfg.rsi_approaching_oversold} [+1]")

    # VWAP
    vwap = calculate_vwap(closes, volumes, highs, lows)
    current_vwap = vwap[-1] if vwap else price
    vwap_dist = ((price - current_vwap) / current_vwap) * 100 if current_vwap > 0 else 0

    if price < current_vwap:
        score += 2
        details.append(f"Price ${price:.2f} below VWAP ${current_vwap:.2f} ({vwap_dist:.2f}%) [+2]")
    elif abs(vwap_dist) < 0.2:
        score += 1
        details.append(f"Price near VWAP ({vwap_dist:.2f}%) [+1]")

    # EMA Crossover
    ema_fast = ema(closes, cfg.ema_fast)
    ema_slow_vals = ema(closes, cfg.ema_slow)
    cur_fast = _safe(ema_fast[-1])
    cur_slow = _safe(ema_slow_vals[-1])
    prev_fast = _safe(ema_fast[-2]) if len(ema_fast) > 1 else 0
    prev_slow = _safe(ema_slow_vals[-2]) if len(ema_slow_vals) > 1 else 0

    if prev_fast <= prev_slow and cur_fast > cur_slow:
        score += 2
        details.append(f"EMA({cfg.ema_fast}) crossed above EMA({cfg.ema_slow}) [+2]")
    elif cur_fast > cur_slow:
        score += 1
        details.append(f"EMA({cfg.ema_fast}) > EMA({cfg.ema_slow}) (uptrend) [+1]")

    # MACD
    macd_line, signal_line, histogram = calculate_macd(
        closes, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    )
    cur_hist = _safe(histogram[-1])
    prev_hist = _safe(histogram[-2]) if len(histogram) > 1 else 0
    if prev_hist <= 0 and cur_hist > 0:
        score += 1
        details.append("MACD histogram flipped positive [+1]")

    # Bollinger Bands
    bb_upper, _, bb_lower = calculate_bollinger_bands(closes, cfg.bb_period, cfg.bb_std_dev)
    current_lower = _safe(bb_lower[-1]) or price
    current_upper = _safe(bb_upper[-1]) or price

    if price < current_lower:
        score += 2
        details.append(f"Price below lower BB ${current_lower:.2f} [+2]")

    # Volume spike
    vol_window = volumes[-30:] if len(volumes) >= 30 else volumes
    is_surge, cur_vol, avg_vol = check_volume_surge(vol_window, cfg.volume_surge_multiplier)
    if is_surge:
        score += 1
        details.append(f"Volume spike {cur_vol:.0f} vs avg {avg_vol:.0f} [+1]")

    # Momentum
    rising, mom_pct = check_momentum(closes, cfg.momentum_bars)
    if rising and mom_pct > 0:
        score += 1
        details.append(f"Positive momentum +{mom_pct:.3f}% over {cfg.momentum_bars} bars [+1]")

    # ATR
    atr_values = calculate_atr(highs, lows, closes, cfg.atr_period)
    current_atr = _safe(atr_values[-1]) or price * 0.005

    stop_loss = price - cfg.atr_stop_multiplier * current_atr
    take_profit = price + cfg.atr_profit_multiplier * current_atr

    return SignalReport(
        symbol="",
        side="BUY" if score >= cfg.buy_score_threshold else "HOLD",
        score=score,
        details=details,
        rsi=rsi,
        price=price,
        vwap=current_vwap,
        atr=current_atr,
        bb_upper=current_upper,
        bb_lower=current_lower,
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
    )


def score_sell_signal(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    cfg: BotConfig,
    entry_price: float = 0.0,
    trailing_stop: float = 0.0,
) -> SignalReport:

    price = closes[-1]
    details = []
    score = 0

    # RSI
    rsi = calculate_rsi(closes, cfg.rsi_period) or 50.0
    if rsi > cfg.rsi_overbought:
        score += 2
        details.append(f"RSI({cfg.rsi_period}) {rsi:.1f} > {cfg.rsi_overbought} [+2]")

    # VWAP
    vwap = calculate_vwap(closes, volumes, highs, lows)
    current_vwap = vwap[-1] if vwap else price
    if price > current_vwap * 1.002:
        score += 2
        details.append(f"Price ${price:.2f} above VWAP ${current_vwap:.2f} [+2]")

    # EMA crossover down
    ema_fast = ema(closes, cfg.ema_fast)
    ema_slow_vals = ema(closes, cfg.ema_slow)
    cur_fast = _safe(ema_fast[-1])
    cur_slow = _safe(ema_slow_vals[-1])
    prev_fast = _safe(ema_fast[-2]) if len(ema_fast) > 1 else 0
    prev_slow = _safe(ema_slow_vals[-2]) if len(ema_slow_vals) > 1 else 0

    if prev_fast >= prev_slow and cur_fast < cur_slow:
        score += 2
        details.append(f"EMA({cfg.ema_fast}) crossed below EMA({cfg.ema_slow}) [+2]")

    # Bollinger Bands
    bb_upper, _, bb_lower = calculate_bollinger_bands(closes, cfg.bb_period, cfg.bb_std_dev)
    current_upper = _safe(bb_upper[-1]) or price
    current_lower = _safe(bb_lower[-1]) or price

    if price > current_upper:
        score += 2
        details.append(f"Price above upper BB ${current_upper:.2f} [+2]")

    # Trailing stop
    if trailing_stop > 0 and price <= trailing_stop:
        score += 1
        details.append(f"Trailing stop hit ${trailing_stop:.2f} [+1]")

    # Hard stop-loss
    atr_values = calculate_atr(highs, lows, closes, cfg.atr_period)
    current_atr = _safe(atr_values[-1]) or price * 0.005

    if entry_price > 0:
        hard_stop = entry_price - cfg.atr_stop_multiplier * current_atr
        if price <= hard_stop:
            score += 3
            details.append(f"Stop-loss: ${price:.2f} <= ${hard_stop:.2f} [+3]")

        # Take-profit check
        take_target = entry_price + cfg.atr_profit_multiplier * current_atr
        if price >= take_target:
            score += 2
            details.append(f"Take-profit: ${price:.2f} >= ${take_target:.2f} [+2]")

    return SignalReport(
        symbol="",
        side="SELL" if score >= cfg.sell_score_threshold else "HOLD",
        score=score,
        details=details,
        rsi=rsi,
        price=price,
        vwap=current_vwap,
        atr=current_atr,
        bb_upper=current_upper,
        bb_lower=current_lower,
    )


# =============================================================================
# POSITION SIZING
# =============================================================================

def calculate_position_size(
    portfolio_value: float, price: float, atr: float, cfg: BotConfig
) -> int:
    if atr <= 0 or price <= 0:
        return 0
    risk_dollars = portfolio_value * cfg.risk_per_trade_pct
    risk_per_share = atr * cfg.atr_stop_multiplier
    shares = int(risk_dollars / risk_per_share)
    max_by_portfolio = int((portfolio_value * cfg.max_position_pct) / price)
    shares = min(shares, max_by_portfolio, cfg.max_shares_per_symbol)
    return max(shares, 0)


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_timeframe(tf_str: str) -> TimeFrame:
    from alpaca.data.timeframe import TimeFrameUnit
    mapping = {
        "1min": TimeFrame.Minute,
        "5min": TimeFrame(5, TimeFrameUnit.Minute),
        "15min": TimeFrame(15, TimeFrameUnit.Minute),
        "1hour": TimeFrame.Hour,
        "1day": TimeFrame.Day,
    }
    return mapping.get(tf_str, TimeFrame(5, TimeFrameUnit.Minute))


def fetch_ohlcv(
    data_client: StockHistoricalDataClient,
    symbol: str,
    cfg: BotConfig,
    logger: logging.Logger,
    lookback_days: Optional[int] = None,
) -> dict:
    # For minute bars we need fewer calendar days
    if lookback_days is None:
        if "min" in cfg.bar_timeframe:
            lookback_days = 5  # ~5 trading days of minute data
        else:
            lookback_days = 140

    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=get_timeframe(cfg.bar_timeframe),
            start=start,
            end=end,
        )
        bars = data_client.get_stock_bars(request)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch data for {symbol}: {exc}") from exc

    try:
        bar_list = bars[symbol]
    except (KeyError, IndexError):
        bar_list = []

    if not bar_list:
        raise RuntimeError(f"No data returned for {symbol}")

    data = {
        "open": [float(b.open) for b in bar_list],
        "high": [float(b.high) for b in bar_list],
        "low": [float(b.low) for b in bar_list],
        "close": [float(b.close) for b in bar_list],
        "volume": [float(b.volume) for b in bar_list],
        "timestamps": [b.timestamp for b in bar_list],
    }

    logger.info(
        "%s — %d bars (%s), latest $%.2f",
        symbol, len(data["close"]), cfg.bar_timeframe, data["close"][-1],
    )
    return data


# =============================================================================
# PORTFOLIO HELPERS
# =============================================================================

def get_position_qty(trading_client: TradingClient, symbol: str) -> int:
    try:
        pos = trading_client.get_open_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0


def get_portfolio_value(trading_client: TradingClient) -> float:
    account = trading_client.get_account()
    return float(account.portfolio_value)


def get_buying_power(trading_client: TradingClient) -> float:
    account = trading_client.get_account()
    return float(account.buying_power)


def get_all_positions(trading_client: TradingClient) -> list:
    try:
        return trading_client.get_all_positions()
    except Exception:
        return []


def count_open_positions(trading_client: TradingClient) -> int:
    return len(get_all_positions(trading_client))


def calculate_portfolio_heat(
    trading_client: TradingClient, state: dict
) -> float:
    portfolio_value = get_portfolio_value(trading_client)
    if portfolio_value <= 0:
        return 1.0
    positions = get_all_positions(trading_client)
    total_risk = 0.0
    for pos in positions:
        sym = pos.symbol
        sym_state = get_symbol_state(state, sym)
        entry = sym_state.get("entry_price", float(pos.avg_entry_price))
        stop = sym_state.get("stop_loss", entry * 0.99)
        qty = int(float(pos.qty))
        risk = max(0, (entry - stop) * qty)
        total_risk += risk
    return total_risk / portfolio_value


# =============================================================================
# ORDER EXECUTION
# =============================================================================

def place_order(
    trading_client: TradingClient,
    symbol: str,
    qty: int,
    side: OrderSide,
    dry_run: bool,
    logger: logging.Logger,
) -> Optional[str]:
    action = "BUY" if side == OrderSide.BUY else "SELL"
    if dry_run:
        logger.info("[DRY RUN] %s %d x %s", action, qty, symbol)
        return "dry-run"
    request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    try:
        order = trading_client.submit_order(request)
        logger.info("ORDER %s %d x %s — ID %s", action, qty, symbol, order.id)
        return str(order.id)
    except Exception as exc:
        logger.error("ORDER FAILED %s %d x %s — %s", action, qty, symbol, exc)
        return None


# =============================================================================
# PER-SYMBOL EVALUATION
# =============================================================================

def evaluate_symbol(
    symbol: str,
    cfg: BotConfig,
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    state: dict,
    logger: logging.Logger,
) -> None:

    logger.info("--- %s ---", symbol)

    try:
        data = fetch_ohlcv(data_client, symbol, cfg, logger)
    except RuntimeError as exc:
        logger.error(str(exc))
        return

    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    volumes = data["volume"]

    min_needed = max(cfg.ema_slow, cfg.bb_period, cfg.vwap_period) + 10
    if len(closes) < min_needed:
        logger.warning("%s — need %d bars, got %d", symbol, min_needed, len(closes))
        return

    shares_held = get_position_qty(trading_client, symbol)
    portfolio_value = get_portfolio_value(trading_client)

    if shares_held > 0:
        # ---- CHECK EXIT ----
        sym_state = get_symbol_state(state, symbol)
        entry_price = sym_state.get("entry_price", 0)
        trailing_stop = sym_state.get("trailing_stop", 0)

        # Update trailing stop
        if cfg.trailing_stop_enabled and entry_price > 0:
            atr_values = calculate_atr(highs, lows, closes, cfg.atr_period)
            current_atr = _safe(atr_values[-1]) or closes[-1] * 0.005
            new_trail = closes[-1] - cfg.atr_stop_multiplier * current_atr
            if new_trail > trailing_stop:
                trailing_stop = round(new_trail, 2)
                update_symbol_state(state, symbol, {"trailing_stop": trailing_stop})

        report = score_sell_signal(closes, highs, lows, volumes, cfg, entry_price, trailing_stop)
        report.symbol = symbol

        logger.info("%s SELL score: %d/%d", symbol, report.score, cfg.sell_score_threshold)
        for d in report.details:
            logger.info("  %s", d)

        if report.side == "SELL":
            if entry_price > 0:
                pnl = (report.price - entry_price) * shares_held
                pnl_pct = ((report.price - entry_price) / entry_price) * 100
                logger.info("%s P&L: $%.2f (%.2f%%)", symbol, pnl, pnl_pct)

            order_id = place_order(trading_client, symbol, shares_held, OrderSide.SELL, cfg.dry_run, logger)
            if order_id:
                update_symbol_state(state, symbol, {
                    "last_trade_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "last_side": "SELL",
                    "entry_price": 0,
                    "trailing_stop": 0,
                    "stop_loss": 0,
                    "take_profit": 0,
                })
        else:
            logger.info("%s HOLD (%d shares, entry $%.2f)", symbol, shares_held, entry_price)

    else:
        # ---- CHECK ENTRY ----
        # Max positions check
        open_count = count_open_positions(trading_client)
        if open_count >= cfg.max_open_positions:
            logger.info("%s — skip: %d/%d positions open", symbol, open_count, cfg.max_open_positions)
            return

        # Portfolio heat check
        heat = calculate_portfolio_heat(trading_client, state)
        if heat >= cfg.max_portfolio_heat_pct:
            logger.info("%s — skip: heat %.1f%% >= %.1f%%", symbol, heat * 100, cfg.max_portfolio_heat_pct * 100)
            return

        report = score_buy_signal(closes, highs, lows, volumes, cfg)
        report.symbol = symbol

        logger.info("%s BUY score: %d/%d", symbol, report.score, cfg.buy_score_threshold)
        for d in report.details:
            logger.info("  %s", d)

        if report.side == "BUY":
            qty = calculate_position_size(portfolio_value, report.price, report.atr, cfg)
            buying_power = get_buying_power(trading_client)
            affordable = int(buying_power // report.price) if report.price > 0 else 0
            qty = min(qty, affordable, cfg.max_shares_per_symbol)

            if qty <= 0:
                logger.warning("%s — BUY signal but qty=0", symbol)
                return

            logger.info(
                "%s BUY %d @ $%.2f | Stop $%.2f | Target $%.2f",
                symbol, qty, report.price, report.stop_loss, report.take_profit,
            )

            order_id = place_order(trading_client, symbol, qty, OrderSide.BUY, cfg.dry_run, logger)
            if order_id:
                update_symbol_state(state, symbol, {
                    "last_trade_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "last_side": "BUY",
                    "last_qty": qty,
                    "entry_price": report.price,
                    "stop_loss": report.stop_loss,
                    "take_profit": report.take_profit,
                    "trailing_stop": report.stop_loss,
                })
        else:
            logger.info("%s — no signal", symbol)


# =============================================================================
# BACKTESTING
# =============================================================================

def run_backtest(cfg: BotConfig, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info("BACKTEST — %d days — %s bars", cfg.backtest_days, cfg.bar_timeframe)
    logger.info("=" * 60)

    data_client = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)

    for symbol in cfg.symbols:
        logger.info("Backtesting %s...", symbol)

        try:
            data = fetch_ohlcv(data_client, symbol, cfg, logger, lookback_days=cfg.backtest_days + 10)
        except RuntimeError as exc:
            logger.error(str(exc))
            continue

        closes = data["close"]
        highs = data["high"]
        lows = data["low"]
        volumes = data["volume"]

        min_warmup = max(cfg.ema_slow, cfg.bb_period, cfg.vwap_period) + 10
        if len(closes) < min_warmup + 20:
            logger.warning("%s — not enough data", symbol)
            continue

        cash = 100000.0
        shares = 0
        entry_price = 0.0
        trailing_stop = 0.0
        wins = 0
        losses = 0
        total_trades = 0
        max_equity = cash
        max_drawdown = 0.0
        total_pnl = 0.0

        for i in range(min_warmup, len(closes)):
            c = closes[:i + 1]
            h = highs[:i + 1]
            l = lows[:i + 1]
            v = volumes[:i + 1]
            price = closes[i]
            equity = cash + shares * price

            max_equity = max(max_equity, equity)
            dd = (max_equity - equity) / max_equity
            max_drawdown = max(max_drawdown, dd)

            if shares > 0:
                # Update trailing stop
                atr_values = calculate_atr(h, l, c, cfg.atr_period)
                current_atr = _safe(atr_values[-1]) or price * 0.005
                new_trail = price - cfg.atr_stop_multiplier * current_atr
                trailing_stop = max(trailing_stop, new_trail)

                report = score_sell_signal(c, h, l, v, cfg, entry_price, trailing_stop)

                if report.side == "SELL":
                    pnl = (price - entry_price) * shares
                    total_pnl += pnl
                    cash += shares * price
                    total_trades += 1
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    shares = 0
                    entry_price = 0
                    trailing_stop = 0
            else:
                report = score_buy_signal(c, h, l, v, cfg)
                if report.side == "BUY" and report.atr > 0:
                    qty = calculate_position_size(cash, price, report.atr, cfg)
                    affordable = int(cash // price)
                    qty = min(qty, affordable)
                    if qty > 0:
                        cash -= qty * price
                        shares = qty
                        entry_price = price
                        trailing_stop = report.stop_loss

        # Close remaining
        if shares > 0:
            pnl = (closes[-1] - entry_price) * shares
            total_pnl += pnl
            cash += shares * closes[-1]
            total_trades += 1
            if pnl > 0:
                wins += 1
            else:
                losses += 1

        ret = ((cash - 100000) / 100000) * 100
        bh_ret = ((closes[-1] - closes[min_warmup]) / closes[min_warmup]) * 100
        wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        logger.info("-" * 50)
        logger.info("%s RESULTS", symbol)
        logger.info("-" * 50)
        logger.info("  Bars analyzed:    %d", len(closes) - min_warmup)
        logger.info("  Final value:      $%.2f", cash)
        logger.info("  Strategy return:  %.2f%%", ret)
        logger.info("  Buy & hold:       %.2f%%", bh_ret)
        logger.info("  Max drawdown:     %.2f%%", max_drawdown * 100)
        logger.info("  Total trades:     %d", total_trades)
        logger.info("  Win rate:         %.0f%% (%dW/%dL)", wr, wins, losses)
        logger.info("  Total P&L:        $%.2f", total_pnl)
        logger.info("-" * 50)


# =============================================================================
# MAIN
# =============================================================================

def run(cfg: Optional[BotConfig] = None) -> None:
    cfg = cfg or BotConfig()
    logger = setup_logging(cfg.log_file)

    if not cfg.api_key or not cfg.secret_key:
        logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        sys.exit(1)

    # Backtest mode
    if "--backtest" in sys.argv:
        if "--days" in sys.argv:
            idx = sys.argv.index("--days")
            if idx + 1 < len(sys.argv):
                cfg.backtest_days = int(sys.argv[idx + 1])
        run_backtest(cfg, logger)
        return

    # Loop mode — run continuously
    loop_mode = "--loop" in sys.argv

    trading_client = TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper_trading)
    data_client = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)

    while True:
        logger.info("=" * 60)
        logger.info(
            "HFT Bot | %s | symbols=%s | %s bars",
            datetime.now().strftime("%H:%M:%S"), cfg.symbols, cfg.bar_timeframe,
        )
        logger.info("=" * 60)

        state = load_state(cfg.state_file)

        for symbol in cfg.symbols:
            try:
                evaluate_symbol(symbol, cfg, trading_client, data_client, state, logger)
            except Exception:
                logger.exception("Error on %s", symbol)

        save_state(state, cfg.state_file)

        if not loop_mode:
            break

        logger.info("Sleeping %ds...", cfg.loop_interval_seconds)
        time.sleep(cfg.loop_interval_seconds)

    logger.info("Done.")


if __name__ == "__main__":
    run()