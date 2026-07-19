"""
Quant X Backtest Engine
========================
Production-grade backtesting with:
- Bar-by-bar simulation (no look-ahead bias)
- Entry at NEXT BAR OPEN (signal fires at candle close, executes next morning)
- Gap-up radar: gap up >2% → watch for pullback to support within 5 bars
- Gap down >3% → skip (extreme)
- R:R re-check: skip if actual R:R < 1.5 at execution price
- Slippage, brokerage, and STT modeling
- Per-strategy breakdown
- Walk-forward validation support

Usage:
    from ml.backtest.engine import BacktestEngine, BacktestConfig
    engine = BacktestEngine()
    results = engine.run(df, strategy, symbol="RELIANCE.NS")
    engine.print_report(results)
"""

import logging
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from ..features.indicators import compute_all_indicators, classify_trend_tier
from ..strategies.base import BaseStrategy, TradeSignal, ExitSignal, Position, Direction
from ..risk_manager import RiskConfig, RiskManager
from ..scanner import TIER1_ONLY

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration.

    For single-stock backtests: max_positions=1, risk_per_trade_pct=2.0
    For portfolio backtests: max_positions=8, risk_per_trade_pct=3.0
    """
    initial_capital: float = 500000.0
    risk_per_trade_pct: float = 3.0  # 3% risk per trade
    slippage_pct: float = 0.05  # 0.05% slippage
    brokerage_pct: float = 0.03  # 0.03% brokerage per side
    stt_pct: float = 0.1  # 0.1% STT on sells
    max_positions: int = 1  # 1 for single-stock backtest (portfolio sim overrides)
    max_position_pct: float = 0.25  # 25% max per position
    max_trades_per_month: int = 12  # Max trades per month (0 = unlimited)
    start_bar: int = 200  # Start after indicators warm up
    min_confidence: float = 65.0  # Minimum signal confidence to accept
    rolling_wr_window: int = 20  # Rolling window for win rate check
    min_rolling_wr: float = 0.25  # Min rolling WR to stay active (relaxed — only pause on extreme cold streak)
    scan_interval: int = 1  # Scan every bar (strategies cache expensive operations internally)


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    symbol: str
    strategy: str
    direction: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    quantity: int
    hold_days: int
    gross_pnl_pct: float
    net_pnl_pct: float  # After costs
    net_pnl_amount: float
    exit_reason: str
    stop_loss: float = 0.0
    target: float = 0.0
    confidence: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    symbol: str
    strategy: str
    config: BacktestConfig
    trades: List[BacktestTrade] = field(default_factory=list)

    # Summary metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_hold_days: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Bar-by-bar backtesting engine.

    Simulates trading one bar at a time:
    1. Scan at bar close for signals → store as pending_signal
    2. Next bar: validate gap. Gap down >3% → skip. Gap up >2% → RADAR
    3. Radar: watch up to 5 bars for pullback to support (signal price, 20/50-SMA)
    4. Enter at bar open + slippage (normal) or bar close (pullback radar entry)
    5. Check exits each bar (SL, target, breakeven, EMA trail)
    6. Record all trades with costs (brokerage + STT)
    """

    def __init__(self, config: Optional[BacktestConfig] = None,
                 risk_manager: Optional[RiskManager] = None):
        self.config = config or BacktestConfig()
        self.risk_manager = risk_manager

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str = "STOCK",
        market_regime: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Run backtest for a single strategy on a single stock.

        Args:
            df: OHLCV DataFrame (minimum 200+ bars)
            strategy: Strategy instance to test
            symbol: Stock symbol
            market_regime: Boolean Series indexed by date (True=bullish, False=bearish).
                          When provided, entries are blocked during bearish regime.

        Returns:
            BacktestResult with all trades and metrics
        """
        # Compute indicators (skip if already computed — e.g. pre-computed by caller)
        if 'ema_200' not in df.columns:
            df = compute_all_indicators(df)
        n = len(df)

        if n < self.config.start_bar + 10:
            logger.warning(f"Not enough data for {symbol}: {n} bars")
            return BacktestResult(symbol=symbol, strategy=strategy.name, config=self.config)

        result = BacktestResult(
            symbol=symbol,
            strategy=strategy.name,
            config=self.config,
        )

        capital = self.config.initial_capital
        equity = capital
        peak_equity = capital
        max_drawdown = 0.0
        position: Optional[Position] = None
        pending_signal: Optional[TradeSignal] = None
        position_confidence: float = 0.0

        # Rolling win rate tracker
        recent_outcomes: List[bool] = []  # True=win, False=loss
        wr_window = self.config.rolling_wr_window
        trading_paused = False
        pause_cooldown: int = 0  # Bars remaining in cooldown before resume

        # Gap-up radar: track signals where next-bar gapped up >2%.
        # Watch for pullback to support within 5 bars — gap up = buyers present.
        radar_signal: Optional[TradeSignal] = None
        radar_bars_left: int = 0
        RADAR_MAX_BARS = 5  # Watch for pullback up to 5 bars after gap-up

        equity_curve = [capital]

        for bar_idx in range(self.config.start_bar, n):
            curr = df.iloc[bar_idx]
            bar_date = str(df.index[bar_idx]) if hasattr(df.index[bar_idx], 'strftime') else str(bar_idx)

            # --- Period resets for risk manager (BEFORE entry/exit) ---
            if self.risk_manager is not None:
                bar_dt = df.index[bar_idx]
                if hasattr(bar_dt, 'date') and bar_idx > self.config.start_bar:
                    prev_dt = df.index[bar_idx - 1]
                    if hasattr(prev_dt, 'date'):
                        if bar_dt.date() != prev_dt.date():
                            self.risk_manager.reset_daily()
                        if bar_dt.isocalendar()[1] != prev_dt.isocalendar()[1]:
                            self.risk_manager.reset_weekly()
                        if bar_dt.month != prev_dt.month:
                            self.risk_manager.reset_monthly()

            # --- GAP-UP RADAR: Check if pullback entry available ---
            if radar_signal is not None and position is None and radar_bars_left > 0:
                low = float(curr['low'])
                close = float(curr['close'])
                sig = radar_signal

                # Pullback targets: signal price (original close), 20-SMA, 50-SMA
                pullback_levels = [sig.entry_price]  # Original signal price
                sma20 = curr.get('sma_20', None)
                sma50 = curr.get('sma_50', None)
                if sma20 is not None and not pd.isna(sma20):
                    pullback_levels.append(float(sma20))
                if sma50 is not None and not pd.isna(sma50):
                    pullback_levels.append(float(sma50))

                # Check: did price pull back to any support level?
                # "Pullback" = low touched or dipped below support, then closed above it
                entered_radar = False
                for lvl in pullback_levels:
                    if lvl <= 0 or lvl <= sig.stop_loss:
                        continue
                    # Low reached within 0.5% of support level AND close held above it
                    if low <= lvl * 1.005 and close >= lvl:
                        entry_price = close  # Enter at close of pullback bar
                        entry_price *= (1 + self.config.slippage_pct / 100)

                        # Validate: entry above stop, R:R >= 1.5
                        if entry_price <= sig.stop_loss:
                            continue
                        actual_risk = entry_price - sig.stop_loss
                        actual_reward = sig.target - entry_price
                        if actual_risk <= 0 or actual_reward <= 0:
                            continue
                        if (actual_reward / actual_risk) < 1.5:
                            continue

                        # Size position
                        if self.risk_manager is not None:
                            if self.risk_manager.is_halted:
                                break
                            quantity = self.risk_manager.calculate_position_size(
                                entry_price, sig.stop_loss)
                        else:
                            risk_amount = capital * (self.config.risk_per_trade_pct / 100)
                            quantity = max(1, int(risk_amount / actual_risk))
                            max_qty = int(capital * self.config.max_position_pct / entry_price)
                            quantity = min(quantity, max_qty)

                        if quantity > 0:
                            position = Position(
                                symbol=symbol,
                                strategy=strategy.name,
                                direction=sig.direction,
                                entry_price=entry_price,
                                entry_date=pd.Timestamp(bar_date) if bar_date else pd.Timestamp.now(),
                                stop_loss=sig.stop_loss,
                                target=sig.target,
                                quantity=quantity,
                                trailing_stop=sig.stop_loss,
                                highest_since_entry=entry_price,
                                lowest_since_entry=entry_price,
                            )
                            position_confidence = sig.confidence
                            if self.risk_manager is not None:
                                self.risk_manager.monthly_trades += 1
                            entered_radar = True
                            radar_signal = None
                            radar_bars_left = 0
                        break

                if not entered_radar:
                    radar_bars_left -= 1
                    if radar_bars_left <= 0:
                        radar_signal = None  # Expired — no pullback within window

            # --- ENTRY: Execute pending signal at this bar's open ---
            # NOTE: No `continue` in this block — equity tracking must always run.
            if pending_signal is not None and position is None:
                entry_price = curr['open']

                if pd.isna(entry_price) or entry_price <= 0:
                    pass  # Invalid open — skip entry
                else:
                    entry_price = float(entry_price)
                    entry_price *= (1 + self.config.slippage_pct / 100)
                    signal_price = pending_signal.entry_price

                    if entry_price > signal_price * 1.02:
                        # Gap up > 2% — don't chase, watch for pullback
                        radar_signal = pending_signal
                        radar_bars_left = RADAR_MAX_BARS
                    elif entry_price < signal_price * 0.97:
                        pass  # Gap down > 3% — extreme, skip
                    elif entry_price <= pending_signal.stop_loss:
                        pass  # Entry at/below stop — trade invalidated
                    else:
                        # Re-check R:R from actual entry price
                        actual_risk = entry_price - pending_signal.stop_loss
                        actual_reward = pending_signal.target - entry_price
                        rr_ok = (actual_risk > 0 and actual_reward > 0
                                 and (actual_reward / actual_risk) >= 1.5)

                        if rr_ok:
                            # Calculate position size
                            quantity = 0
                            halted = self.risk_manager.is_halted if self.risk_manager else False
                            if not halted:
                                if self.risk_manager is not None:
                                    quantity = self.risk_manager.calculate_position_size(
                                        entry_price, pending_signal.stop_loss
                                    )
                                else:
                                    risk_amount = capital * (self.config.risk_per_trade_pct / 100)
                                    quantity = max(1, int(risk_amount / actual_risk))
                                    max_qty = int(capital * self.config.max_position_pct / entry_price)
                                    quantity = min(quantity, max_qty)

                            if quantity > 0:
                                position = Position(
                                    symbol=symbol,
                                    strategy=strategy.name,
                                    direction=pending_signal.direction,
                                    entry_price=entry_price,
                                    entry_date=pd.Timestamp(bar_date) if bar_date else pd.Timestamp.now(),
                                    stop_loss=pending_signal.stop_loss,
                                    target=pending_signal.target,
                                    quantity=quantity,
                                    trailing_stop=pending_signal.stop_loss,
                                    highest_since_entry=entry_price,
                                    lowest_since_entry=entry_price,
                                )
                                position_confidence = pending_signal.confidence
                                if self.risk_manager is not None:
                                    self.risk_manager.monthly_trades += 1

                pending_signal = None

            # --- POSITION MANAGEMENT ---
            if position is not None:
                high = curr['high']
                low = curr['low']
                close = curr['close']
                open_price = curr['open']

                position.hold_days += 1  # Engine owns hold_days increment
                position.highest_since_entry = max(position.highest_since_entry, high)
                position.lowest_since_entry = min(position.lowest_since_entry, low)

                # Intrabar simulation: when BOTH SL and target trigger on same bar,
                # determine which was hit first based on bar direction
                stop_hit = low <= position.stop_loss
                target_hit = high >= position.target

                if stop_hit and target_hit:
                    bar_is_bullish = close >= open_price
                    if bar_is_bullish:
                        # Bullish bar: dipped low first then rallied high → stop hit first
                        exit_first = 'stop'
                    else:
                        # Bearish bar: rallied high first then dropped low → target hit first
                        exit_first = 'target'

                    if exit_first == 'target':
                        exit_signal = ExitSignal(reason="target_hit", exit_price=position.target)
                    else:
                        exit_signal = ExitSignal(reason="stop_loss", exit_price=position.stop_loss)
                else:
                    # Normal exit check via strategy
                    df_slice = df.iloc[:bar_idx + 1]
                    exit_signal = strategy.should_exit(df_slice, position)

                # MAX-HOLD TIME EXIT: smart stale-position management
                # Quick winners (target in 1-2 days) are already handled by
                # target_hit check above — time exit NEVER interferes with those.
                max_hold = getattr(strategy, 'max_hold_bars', 25)
                if exit_signal is None and position.hold_days >= max_hold:
                    risk = position.entry_price - position.stop_loss
                    current_profit = close - position.entry_price
                    r_multiple = current_profit / risk if risk > 0 else 0

                    if close <= position.entry_price:
                        # Losing at max hold → cut loss
                        exit_signal = ExitSignal(reason="time_exit", exit_price=close)
                    elif r_multiple < 1.0:
                        # Small profit but below 1R at max hold → take it, don't
                        # risk it reversing back to a loss
                        exit_signal = ExitSignal(reason="time_exit", exit_price=close)

                # Hard cap: force exit at 1.5x max_hold regardless of P&L
                # Prevents any position from being stuck indefinitely
                if exit_signal is None and position.hold_days >= int(max_hold * 1.5):
                    exit_signal = ExitSignal(reason="time_exit", exit_price=close)

                if exit_signal is not None:
                    exit_price = exit_signal.exit_price

                    # Gap fill realism for long positions:
                    # - Any stop-type exit: if bar opened below stop, fill at open (worse)
                    # - Target exit: if bar opened above target, fill at open (better)
                    if exit_signal.reason != "target_hit":
                        exit_price = min(exit_price, float(open_price))
                    else:
                        exit_price = max(exit_price, float(open_price))

                    # Apply slippage (sell lower)
                    exit_price *= (1 - self.config.slippage_pct / 100)

                    # Calculate P&L
                    gross_pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100
                    gross_pnl_amount = (exit_price - position.entry_price) * position.quantity

                    # Costs
                    exit_cost = exit_price * position.quantity * (
                        self.config.brokerage_pct / 100 + self.config.stt_pct / 100
                    )
                    entry_cost = position.entry_price * position.quantity * (self.config.brokerage_pct / 100)
                    total_cost = entry_cost + exit_cost

                    net_pnl_amount = gross_pnl_amount - total_cost
                    net_pnl_pct = net_pnl_amount / (position.entry_price * position.quantity) * 100

                    trade = BacktestTrade(
                        symbol=symbol,
                        strategy=strategy.name,
                        direction=position.direction.value,
                        entry_date=str(position.entry_date),
                        entry_price=round(position.entry_price, 2),
                        exit_date=bar_date,
                        exit_price=round(exit_price, 2),
                        quantity=position.quantity,
                        hold_days=position.hold_days,
                        gross_pnl_pct=round(gross_pnl_pct, 2),
                        net_pnl_pct=round(net_pnl_pct, 2),
                        net_pnl_amount=round(net_pnl_amount, 2),
                        exit_reason=exit_signal.reason,
                        stop_loss=round(position.stop_loss, 2),
                        target=round(position.target, 2),
                        confidence=round(position_confidence, 1),
                    )

                    result.trades.append(trade)
                    equity += net_pnl_amount
                    if equity <= 0:
                        equity = 0
                        logger.warning(f"Account blown up at bar {bar_idx} — stopping backtest")
                        break
                    capital = equity  # Update capital for next trade sizing
                    position = None

                    # Record result in risk manager (loss limits, consecutive loss tracking)
                    if self.risk_manager is not None:
                        self.risk_manager.record_trade_result(net_pnl_pct)
                        self.risk_manager.config.account_capital = equity

                    # Update rolling win rate tracker
                    recent_outcomes.append(net_pnl_pct > 0)
                    if len(recent_outcomes) > wr_window:
                        recent_outcomes.pop(0)
                    if len(recent_outcomes) >= wr_window:
                        rolling_wr = sum(recent_outcomes) / len(recent_outcomes)
                        if rolling_wr < self.config.min_rolling_wr:
                            trading_paused = True
                            pause_cooldown = wr_window  # Cooldown = window size bars
                        else:
                            trading_paused = False

            # Auto-resume from cold streak pause after cooldown
            if trading_paused and position is None:
                pause_cooldown -= 1
                if pause_cooldown <= 0:
                    trading_paused = False

            # --- SCAN: Look for new signal at bar close ---
            if position is None and pending_signal is None:
                # Skip scanning if trading paused (cold streak) or risk manager halted
                halted = self.risk_manager.is_halted if self.risk_manager else False
                monthly_limit = (self.risk_manager.monthly_trade_limit_reached
                                 if self.risk_manager else False)
                # Scan interval: only run expensive strategy.scan() every N bars
                scan_due = (bar_idx % self.config.scan_interval == 0)
                if trading_paused or halted or monthly_limit or not scan_due:
                    pass
                else:
                    df_slice = df.iloc[:bar_idx + 1]

                    # Trend gate: classify tier and skip if not appropriate
                    tier = classify_trend_tier(df_slice)
                    if tier == 'skip':
                        signal = None
                    elif tier == 'tier2' and strategy.name in TIER1_ONLY:
                        signal = None
                    else:
                        signal = strategy.scan(df_slice, symbol=symbol)

                    if signal is not None:
                        # Confidence filter
                        if signal.confidence < self.config.min_confidence:
                            signal = None
                        # Market regime filter: block entries in bearish regime
                        if signal is not None and market_regime is not None:
                            bar_date = df.index[bar_idx]
                            valid_dates = market_regime.index[market_regime.index <= bar_date]
                            if len(valid_dates) > 0 and not market_regime.loc[valid_dates[-1]]:
                                signal = None  # Bearish regime — skip entry
                        if signal is not None:
                            pending_signal = signal
                            # Fresh signal replaces any stale radar
                            radar_signal = None
                            radar_bars_left = 0

            # Track equity (mark-to-market for open positions)
            mtm_equity = equity
            if position is not None:
                close = float(curr['close'])
                unrealized = (close - position.entry_price) * position.quantity
                mtm_equity = equity + unrealized
            equity_curve.append(mtm_equity)
            peak_equity = max(peak_equity, mtm_equity)
            drawdown = (peak_equity - mtm_equity) / peak_equity * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate final metrics
        result.equity_curve = equity_curve
        self._calculate_metrics(result, max_drawdown)

        return result

    def run_multi_stock(
        self,
        stock_data: Dict[str, pd.DataFrame],
        strategy: BaseStrategy,
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest across multiple stocks.

        Returns:
            Dict mapping symbol -> BacktestResult
        """
        results = {}
        for symbol, df in stock_data.items():
            try:
                result = self.run(df, strategy, symbol)
                results[symbol] = result
                logger.info(
                    f"{symbol}: {result.total_trades} trades, "
                    f"WR={result.win_rate:.1f}%, "
                    f"Return={result.total_return_pct:.1f}%"
                )
            except Exception as e:
                logger.error(f"Backtest failed for {symbol}: {e}")

        return results

    def _calculate_metrics(self, result: BacktestResult, max_drawdown: float):
        """Calculate summary metrics from trades."""
        trades = result.trades
        if not trades:
            return

        pnls = [t.net_pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.total_trades = len(trades)
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = round(len(wins) / len(pnls) * 100, 1)
        result.avg_win_pct = round(np.mean(wins), 2) if wins else 0.0
        result.avg_loss_pct = round(np.mean(losses), 2) if losses else 0.0
        result.avg_pnl_pct = round(np.mean(pnls), 2)
        # Compounded return from actual equity curve (not sum of per-trade %)
        final_equity = result.equity_curve[-1] if result.equity_curve else self.config.initial_capital
        result.total_return_pct = round(
            (final_equity - self.config.initial_capital) / self.config.initial_capital * 100, 2
        )
        result.max_drawdown_pct = round(max_drawdown, 2)
        result.avg_hold_days = round(np.mean([t.hold_days for t in trades]), 1)

        # Profit factor (using actual rupee amounts, not summed percentages)
        wins_amt = [t.quantity * t.entry_price * t.net_pnl_pct / 100
                    for t in trades if t.net_pnl_pct > 0]
        losses_amt = [t.quantity * t.entry_price * t.net_pnl_pct / 100
                      for t in trades if t.net_pnl_pct <= 0]
        total_wins = sum(wins_amt) if wins_amt else 0
        total_losses = abs(sum(losses_amt)) if losses_amt else 0
        result.profit_factor = round(total_wins / total_losses, 2) if total_losses > 0 else None

        # Sharpe ratio (annualized by trades per year, ddof=1 for sample std)
        if len(pnls) > 1:
            pnl_std = np.std(pnls, ddof=1)
            if pnl_std > 0:
                # Estimate trades per year from actual date range
                try:
                    first_date = pd.to_datetime(trades[0].entry_date)
                    last_date = pd.to_datetime(trades[-1].exit_date)
                    years = max(0.5, (last_date - first_date).days / 365.25)
                    trades_per_year = len(trades) / years
                except Exception:
                    trades_per_year = min(250, len(pnls))
                result.sharpe_ratio = round(
                    (np.mean(pnls) / pnl_std) * np.sqrt(trades_per_year), 2
                )

        # Consecutive wins/losses
        max_consec_w = max_consec_l = 0
        curr_w = curr_l = 0
        for p in pnls:
            if p > 0:
                curr_w += 1
                curr_l = 0
            else:
                curr_l += 1
                curr_w = 0
            max_consec_w = max(max_consec_w, curr_w)
            max_consec_l = max(max_consec_l, curr_l)

        result.max_consecutive_wins = max_consec_w
        result.max_consecutive_losses = max_consec_l

    @staticmethod
    def print_report(result: BacktestResult):
        """Print a formatted backtest report."""
        print(f"\n{'='*60}")
        print(f"BACKTEST REPORT: {result.strategy} on {result.symbol}")
        print(f"{'='*60}")
        print(f"Total Trades:          {result.total_trades}")
        print(f"Win Rate:              {result.win_rate}%")
        print(f"Avg Win:               +{result.avg_win_pct}%")
        print(f"Avg Loss:              {result.avg_loss_pct}%")
        print(f"Avg P&L per trade:     {result.avg_pnl_pct}%")
        print(f"Total Return:          {result.total_return_pct}%")
        print(f"Max Drawdown:          -{result.max_drawdown_pct}%")
        print(f"Profit Factor:         {result.profit_factor}")
        print(f"Sharpe Ratio:          {result.sharpe_ratio}")
        print(f"Avg Hold Days:         {result.avg_hold_days}")
        print(f"Max Consec Wins:       {result.max_consecutive_wins}")
        print(f"Max Consec Losses:     {result.max_consecutive_losses}")
        print(f"{'='*60}")

        # Exit reason breakdown
        if result.trades:
            reasons = {}
            for t in result.trades:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            print(f"\nExit Reasons:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  {reason:20s} {count:4d} ({count/len(result.trades)*100:.1f}%)")

        print()
