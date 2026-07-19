"""
Quant X Risk Manager
=====================
Production-grade risk management for real money trading.

Features:
- ATR-based position sizing (1-2% risk per trade)
- Portfolio heat limits (max total risk exposure)
- Daily/weekly/monthly loss limits with auto-halt
- Consecutive loss tracking with size reduction
"""

import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk management configuration.

    All parameters are user-configurable. Pass custom values when creating:
        config = RiskConfig(risk_per_trade_pct=3.0, max_open_positions=5)
    """
    # Account
    account_capital: float = 500000.0  # INR 5 lakhs default

    # Position sizing
    risk_per_trade_pct: float = 3.0  # 3% of account per trade
    max_position_pct: float = 20.0  # Max 20% of account in single position

    # Portfolio limits
    max_open_positions: int = 0  # 0 = unlimited — take every qualified signal
    max_portfolio_heat_pct: float = 25.0  # Max total risk as % of account
    max_trades_per_month: int = 12  # Max trades per month (0 = unlimited)

    # Loss limits
    daily_loss_limit_pct: float = 5.0  # Auto-halt at 5% daily loss
    weekly_loss_limit_pct: float = 8.0  # Reduce size at 8% weekly loss
    monthly_loss_limit_pct: float = 15.0  # Pause at 15% monthly loss

    # Max drawdown protection
    max_drawdown_pct: float = 20.0  # Halt at 20% drawdown from peak equity

    # Consecutive loss handling
    consecutive_loss_reduce_after: int = 3  # Reduce size after 3 consecutive losses
    consecutive_loss_size_mult: float = 0.5  # Reduce to 50%

    # Execution costs (Indian market)
    slippage_pct: float = 0.05  # 0.05% slippage
    brokerage_pct: float = 0.03  # 0.03% brokerage
    stt_pct: float = 0.1  # 0.1% STT on sells


class RiskManager:
    """Manages position sizing and risk limits."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.monthly_pnl: float = 0.0
        self.monthly_trades: int = 0
        self.consecutive_losses: int = 0
        self._halt_daily: bool = False
        self._halt_weekly: bool = False
        self._halt_monthly: bool = False
        self._halt_drawdown: bool = False
        self._peak_equity: float = self.config.account_capital
        self._current_equity: float = self.config.account_capital

    @property
    def is_halted(self) -> bool:
        return self._halt_daily or self._halt_weekly or self._halt_monthly or self._halt_drawdown

    @property
    def halt_reason(self) -> str:
        reasons = []
        if self._halt_daily:
            reasons.append(f"Daily loss limit hit: {self.daily_pnl:.1f}%")
        if self._halt_weekly:
            reasons.append(f"Weekly loss limit hit: {self.weekly_pnl:.1f}%")
        if self._halt_monthly:
            reasons.append(f"Monthly loss limit hit: {self.monthly_pnl:.1f}%")
        if self._halt_drawdown:
            dd_pct = (self._peak_equity - self._current_equity) / self._peak_equity * 100 if self._peak_equity > 0 else 0
            reasons.append(f"Max drawdown hit: {dd_pct:.1f}% from peak")
        return " | ".join(reasons)

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        atr: Optional[float] = None,
    ) -> int:
        """
        Calculate position size based on ATR and account risk.

        Formula:
            Risk Amount = Account Capital × risk_per_trade_pct%
            Stop Distance = |entry_price - stop_loss|
            Position Size = Risk Amount / Stop Distance

        Args:
            entry_price: Planned entry price
            stop_loss: Stop loss price
            atr: ATR value (for reference, stop_loss should already use ATR)

        Returns:
            Number of shares to buy (integer)
        """
        if self.is_halted:
            logger.warning(f"Trading halted: {self.halt_reason}")
            return 0

        # Apply consecutive loss reduction
        risk_pct = self.config.risk_per_trade_pct
        if self.consecutive_losses >= self.config.consecutive_loss_reduce_after:
            risk_pct *= self.config.consecutive_loss_size_mult
            logger.info(
                f"Consecutive losses ({self.consecutive_losses}): "
                f"reducing risk to {risk_pct:.1f}%"
            )

        risk_amount = self.config.account_capital * (risk_pct / 100)
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance <= 0:
            logger.warning("Stop distance is zero, cannot size position")
            return 0

        position_size = int(risk_amount / stop_distance)

        # Cap by max position size
        max_shares = int(
            self.config.account_capital * (self.config.max_position_pct / 100) / entry_price
        )
        position_size = min(position_size, max_shares)

        # Minimum 1 share
        position_size = max(1, position_size)

        logger.debug(
            f"Position size: {position_size} shares | "
            f"Risk: ₹{risk_amount:.0f} | Stop distance: ₹{stop_distance:.2f}"
        )
        return position_size

    def check_portfolio_heat(
        self,
        open_positions: dict,
        new_risk_amount: float = 0,
    ) -> bool:
        """
        Check if adding a new position exceeds portfolio heat limit.

        Portfolio heat = total risk across all positions as % of account.

        Returns:
            True if new position is allowed, False if it would exceed heat limit.
        """
        total_risk = new_risk_amount
        for symbol, position in open_positions.items():
            risk = abs(position.entry_price - position.stop_loss) * position.quantity
            total_risk += risk

        heat_pct = (total_risk / self.config.account_capital) * 100

        if heat_pct > self.config.max_portfolio_heat_pct:
            logger.warning(
                f"Portfolio heat {heat_pct:.1f}% exceeds limit "
                f"{self.config.max_portfolio_heat_pct}%"
            )
            return False

        return True

    def record_trade_result(self, pnl_pct: float):
        """
        Record a trade result and update loss tracking.

        Args:
            pnl_pct: Trade P&L as percentage (positive = profit)
        """
        self.daily_pnl += pnl_pct
        self.weekly_pnl += pnl_pct
        self.monthly_pnl += pnl_pct

        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Check loss limits (each level sets its own flag independently)
        if abs(self.daily_pnl) >= self.config.daily_loss_limit_pct and self.daily_pnl < 0:
            self._halt_daily = True
            logger.critical(f"Daily loss limit hit: {self.daily_pnl:.1f}%")

        if abs(self.weekly_pnl) >= self.config.weekly_loss_limit_pct and self.weekly_pnl < 0:
            self._halt_weekly = True
            logger.critical(f"Weekly loss limit hit: {self.weekly_pnl:.1f}%")

        if abs(self.monthly_pnl) >= self.config.monthly_loss_limit_pct and self.monthly_pnl < 0:
            self._halt_monthly = True
            logger.critical(f"Monthly loss limit hit: {self.monthly_pnl:.1f}%")

    def reset_daily(self):
        """Reset daily P&L counter and clear daily halt (call at start of each trading day)."""
        self.daily_pnl = 0.0
        self._halt_daily = False

    def reset_weekly(self):
        """Reset weekly P&L counter and clear weekly halt (call at start of each trading week)."""
        self.weekly_pnl = 0.0
        self._halt_weekly = False

    def reset_monthly(self):
        """Reset all counters and halt flags (call at start of each month)."""
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.monthly_trades = 0
        self.consecutive_losses = 0
        self._halt_daily = False
        self._halt_weekly = False
        self._halt_monthly = False
        self._halt_drawdown = False

    @property
    def monthly_trade_limit_reached(self) -> bool:
        """Check if monthly trade limit has been reached."""
        limit = self.config.max_trades_per_month
        return limit > 0 and self.monthly_trades >= limit

    def update_equity(self, current_equity: float):
        """
        Update current equity and check max drawdown limit.

        Call this after each trade closes or at end of each trading day.

        Args:
            current_equity: Current account equity in INR.
        """
        self._current_equity = current_equity

        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity > 0 and self.config.max_drawdown_pct > 0:
            drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity * 100
            if drawdown_pct >= self.config.max_drawdown_pct:
                self._halt_drawdown = True
                logger.critical(
                    f"Max drawdown limit hit: {drawdown_pct:.1f}% "
                    f"(peak={self._peak_equity:.0f}, current={current_equity:.0f})"
                )

    def resume_trading(self):
        """Manually resume trading after a halt."""
        self._halt_daily = False
        self._halt_weekly = False
        self._halt_monthly = False
        self._halt_drawdown = False
        logger.info("Trading resumed manually")

    def calculate_execution_cost(self, price: float, quantity: int, is_sell: bool = False) -> float:
        """
        Calculate total execution cost for a trade.

        Returns:
            Total cost in INR (slippage + brokerage + STT for sells)
        """
        trade_value = price * quantity
        slippage = trade_value * (self.config.slippage_pct / 100)
        brokerage = trade_value * (self.config.brokerage_pct / 100)
        stt = trade_value * (self.config.stt_pct / 100) if is_sell else 0

        return round(slippage + brokerage + stt, 2)

    def get_risk_summary(self) -> dict:
        """Get current risk state summary."""
        return {
            'account_capital': self.config.account_capital,
            'daily_pnl': round(self.daily_pnl, 2),
            'weekly_pnl': round(self.weekly_pnl, 2),
            'monthly_pnl': round(self.monthly_pnl, 2),
            'consecutive_losses': self.consecutive_losses,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'risk_per_trade': self.config.risk_per_trade_pct,
        }
