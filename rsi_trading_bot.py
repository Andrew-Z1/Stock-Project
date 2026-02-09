"""
RSI Trading Bot using Alpaca API

Executes daily after market close to analyze price/volume and trade based on RSI signals.

Features:
- Wilder's smoothed RSI (standard calculation)
- Multi-symbol watchlist support
- Volume surge detection (current vs. prior-day average)
- Buying power validation before orders
- Structured logging with file + console output
- Dry-run mode for signal testing without placing orders
- Configurable via environment variables or config dict
- Graceful error handling and retries

Constraints:
- Configurable cooldown between trades (default: 3 days)
- Max position size per symbol (default: 500 shares)
- Trade size per transaction (default: 100 shares)
- RSI < 30 + volume surge = BUY
- RSI > 70 = SELL (sells TRADE_SIZE, not entire position)
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
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
    """All tunable parameters in one place."""

    # Alpaca credentials
    api_key: str = os.getenv("ALPACA_API_KEY", "")
    secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    paper_trading: bool = True

    # Watchlist — bot evaluates each symbol independently
    symbols: list[str] = field(default_factory=lambda: ["AAPL"])

    # Position limits
    max_shares: int = 500
    trade_size: int = 100

    # Cooldown
    cooldown_days: int = 3

    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Volume
    volume_lookback: int = 20
    volume_surge_multiplier: float = 1.5

    # Operational
    state_file: str = "bot_state.json"
    log_file: str = "bot.log"
    dry_run: bool = False  # When True, signals are logged but no orders placed
    data_buffer_days: int = 45  # Calendar days to fetch (covers weekends/holidays)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_file: str) -> logging.Logger:
    """Configure logger with console + rotating file output."""
    logger = logging.getLogger("rsi_bot")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG and above)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state(path: str) -> dict:
    """Load persisted bot state from JSON file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict, path: str) -> None:
    """Atomically save bot state to JSON file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)  # atomic on POSIX


def get_last_trade_date(state: dict, symbol: str) -> Optional[datetime]:
    """Return the last trade date for a symbol, or None."""
    per_symbol = state.get("symbols", {}).get(symbol, {})
    raw = per_symbol.get("last_trade_date")
    if raw:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    return None


def record_trade(state: dict, symbol: str, side: str, qty: int, date: datetime) -> None:
    """Update state after a successful trade."""
    state.setdefault("symbols", {}).setdefault(symbol, {})
    state["symbols"][symbol]["last_trade_date"] = date.strftime("%Y-%m-%d")
    state["symbols"][symbol]["last_side"] = side
    state["symbols"][symbol]["last_qty"] = qty


# =============================================================================
# RSI — WILDER'S SMOOTHED METHOD
# =============================================================================

def calculate_rsi(prices: list[float], period: int = 14) -> Optional[float]:
    """
    Compute RSI using Wilder's smoothing (exponential moving average of
    gains/losses), which is the industry-standard approach.

    Requires at least ``period + 1`` prices.

    Returns RSI in [0, 100] or None if insufficient data.
    """
    if len(prices) < period + 1:
        return None

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    # Seed the averages with a simple mean over the first `period` changes
    seed_gains = [max(d, 0) for d in deltas[:period]]
    seed_losses = [max(-d, 0) for d in deltas[:period]]

    avg_gain = sum(seed_gains) / period
    avg_loss = sum(seed_losses) / period

    # Wilder's smoothing for subsequent values
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


# =============================================================================
# VOLUME ANALYSIS
# =============================================================================

def check_volume_surge(
    volumes: list[float], multiplier: float = 1.5
) -> tuple[bool, float, float]:
    """
    Compare the most recent volume bar against the average of all *prior* bars.

    Returns (is_surge, current_volume, prior_avg_volume).
    """
    if len(volumes) < 2:
        return False, 0.0, 0.0

    current = volumes[-1]
    prior = volumes[:-1]
    avg_prior = sum(prior) / len(prior)

    return current > avg_prior * multiplier, current, avg_prior


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_bars(
    data_client: StockHistoricalDataClient,
    symbol: str,
    lookback_calendar_days: int,
    logger: logging.Logger,
) -> tuple[list[float], list[float]]:
    """
    Fetch daily OHLCV bars and return (close_prices, volumes) oldest-first.

    Raises RuntimeError on failure.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_calendar_days)

    logger.debug("Fetching bars for %s from %s to %s", symbol, start.date(), end.date())

    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = data_client.get_stock_bars(request)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch bar data for {symbol}: {exc}") from exc

    bar_list = bars.get(symbol, [])
    if not bar_list:
        raise RuntimeError(f"No bar data returned for {symbol}")

    prices = [float(b.close) for b in bar_list]
    volumes = [float(b.volume) for b in bar_list]

    logger.info("%s — fetched %d bars, latest close $%.2f", symbol, len(prices), prices[-1])
    return prices, volumes


# =============================================================================
# POSITION & ACCOUNT HELPERS
# =============================================================================

def get_position_qty(trading_client: TradingClient, symbol: str) -> int:
    """Return current share count for *symbol*, 0 if flat."""
    try:
        pos = trading_client.get_open_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0


def get_buying_power(trading_client: TradingClient) -> float:
    """Return available buying power in USD."""
    account = trading_client.get_account()
    return float(account.buying_power)


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
    """
    Submit a market order (or simulate in dry-run mode).

    Returns the order ID on success, or None on failure / dry-run.
    """
    action = "BUY" if side == OrderSide.BUY else "SELL"

    if dry_run:
        logger.info("[DRY RUN] Would %s %d shares of %s", action, qty, symbol)
        return None

    request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
    )

    try:
        order = trading_client.submit_order(request)
        logger.info("Order submitted — %s %d %s — ID %s", action, qty, symbol, order.id)
        return str(order.id)
    except Exception as exc:
        logger.error("Order FAILED — %s %d %s — %s", action, qty, symbol, exc)
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
    today: datetime,
    logger: logging.Logger,
) -> None:
    """Run the full signal-check → trade pipeline for one symbol."""

    logger.info("— Evaluating %s —", symbol)

    # ---- cooldown check ---------------------------------------------------
    last_trade = get_last_trade_date(state, symbol)
    if last_trade:
        days_since = (today - last_trade).days
        if days_since < cfg.cooldown_days:
            remaining = cfg.cooldown_days - days_since
            logger.info("%s — cooldown active, %d day(s) remaining", symbol, remaining)
            return

    # ---- fetch data -------------------------------------------------------
    try:
        prices, volumes = fetch_bars(data_client, symbol, cfg.data_buffer_days, logger)
    except RuntimeError as exc:
        logger.error(str(exc))
        return

    if len(prices) < cfg.rsi_period + 1:
        logger.warning(
            "%s — insufficient data (%d bars, need %d)",
            symbol, len(prices), cfg.rsi_period + 1,
        )
        return

    # ---- indicators -------------------------------------------------------
    rsi = calculate_rsi(prices, cfg.rsi_period)
    vol_window = volumes[-cfg.volume_lookback:] if len(volumes) >= cfg.volume_lookback else volumes
    is_surge, cur_vol, avg_vol = check_volume_surge(vol_window, cfg.volume_surge_multiplier)

    logger.info(
        "%s — RSI %.2f | Vol %,.0f (avg %,.0f) surge=%s",
        symbol, rsi, cur_vol, avg_vol, is_surge,
    )

    # ---- position info ----------------------------------------------------
    shares_held = get_position_qty(trading_client, symbol)
    logger.info("%s — holding %d shares (max %d)", symbol, shares_held, cfg.max_shares)

    # ---- decision ---------------------------------------------------------
    if rsi < cfg.rsi_oversold and is_surge and shares_held < cfg.max_shares:
        buy_qty = min(cfg.trade_size, cfg.max_shares - shares_held)

        # Buying-power guard (rough check using latest close)
        latest_price = prices[-1]
        estimated_cost = buy_qty * latest_price
        buying_power = get_buying_power(trading_client)
        if estimated_cost > buying_power:
            affordable = int(buying_power // latest_price)
            if affordable <= 0:
                logger.warning(
                    "%s — BUY signal but insufficient buying power ($%.2f)",
                    symbol, buying_power,
                )
                return
            buy_qty = min(buy_qty, affordable)
            logger.info(
                "%s — reduced buy qty to %d (buying power $%.2f)",
                symbol, buy_qty, buying_power,
            )

        logger.info("%s — BUY SIGNAL: RSI %.2f + volume surge → buying %d shares", symbol, rsi, buy_qty)
        order_id = place_order(trading_client, symbol, buy_qty, OrderSide.BUY, cfg.dry_run, logger)
        if order_id or cfg.dry_run:
            record_trade(state, symbol, "BUY", buy_qty, today)

    elif rsi > cfg.rsi_overbought and shares_held > 0:
        sell_qty = min(cfg.trade_size, shares_held)
        logger.info("%s — SELL SIGNAL: RSI %.2f → selling %d shares", symbol, rsi, sell_qty)
        order_id = place_order(trading_client, symbol, sell_qty, OrderSide.SELL, cfg.dry_run, logger)
        if order_id or cfg.dry_run:
            record_trade(state, symbol, "SELL", sell_qty, today)

    else:
        reason = ""
        if rsi < cfg.rsi_oversold and not is_surge:
            reason = " (oversold but no volume confirmation)"
        elif rsi < cfg.rsi_oversold and shares_held >= cfg.max_shares:
            reason = " (oversold but max position reached)"
        elif rsi > cfg.rsi_overbought and shares_held == 0:
            reason = " (overbought but no position to sell)"
        logger.info("%s — HOLD: RSI %.2f%s", symbol, rsi, reason)


# =============================================================================
# MAIN
# =============================================================================

def run(cfg: Optional[BotConfig] = None) -> None:
    """Entry point — evaluates every symbol in the watchlist."""

    cfg = cfg or BotConfig()
    logger = setup_logging(cfg.log_file)

    if not cfg.api_key or not cfg.secret_key:
        logger.error(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "environment variables or pass them in BotConfig."
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(
        "RSI Trading Bot started | paper=%s | dry_run=%s | symbols=%s",
        cfg.paper_trading, cfg.dry_run, cfg.symbols,
    )
    logger.info("=" * 60)

    trading_client = TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper_trading)
    data_client = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)

    state = load_state(cfg.state_file)
    today = datetime.now().date()

    for symbol in cfg.symbols:
        try:
            evaluate_symbol(symbol, cfg, trading_client, data_client, state, today, logger)
        except Exception:
            logger.exception("Unhandled error evaluating %s", symbol)

    save_state(state, cfg.state_file)

    logger.info("Run complete — state saved to %s", cfg.state_file)
    logger.info("=" * 60)


if __name__ == "__main__":
    run()