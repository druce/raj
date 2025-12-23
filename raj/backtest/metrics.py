"""Performance metrics calculation."""
import numpy as np
import pandas as pd
from typing import Dict, List

from raj.backtest.signals import Trade


class PerformanceMetrics:
    """Calculate backtesting performance metrics."""

    @staticmethod
    def calculate_returns(equity_curve: pd.Series) -> pd.Series:
        """Calculate daily returns from equity curve.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Series of daily returns
        """
        return equity_curve.pct_change().fillna(0)

    @staticmethod
    def calculate_total_return(equity_curve: pd.Series) -> float:
        """Calculate total return percentage.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Total return as percentage
        """
        if len(equity_curve) == 0:
            return 0.0
        return ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100

    @staticmethod
    def calculate_annualized_return(equity_curve: pd.Series) -> float:
        """Calculate annualized return.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Annualized return as percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        days = (equity_curve.index[-1] - equity_curve.index[0]).days

        if days == 0:
            return 0.0

        years = days / 365.25
        annualized = (total_return ** (1 / years) - 1) * 100

        return annualized

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sharpe ratio.

        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate (default: 0)

        Returns:
            Sharpe ratio (annualized)
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        # Calculate excess returns
        excess_returns = returns - daily_rf

        # Annualize
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)

        return sharpe

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown percentage.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Maximum drawdown as positive percentage
        """
        if len(equity_curve) == 0:
            return 0.0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max

        # Return maximum drawdown as positive percentage
        return abs(drawdown.min()) * 100

    @staticmethod
    def calculate_win_rate(trades: List[Trade]) -> float:
        """Calculate win rate from trade history.

        Args:
            trades: List of completed trades

        Returns:
            Win rate as percentage
        """
        if len(trades) == 0:
            return 0.0

        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        return (winning_trades / len(trades)) * 100

    @staticmethod
    def calculate_profit_factor(trades: List[Trade]) -> float:
        """Calculate profit factor (gross profit / gross loss).

        Args:
            trades: List of completed trades

        Returns:
            Profit factor
        """
        if len(trades) == 0:
            return 0.0

        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def calculate_all_metrics(
        equity_curve: pd.Series,
        trades: List[Trade],
        initial_capital: float
    ) -> Dict:
        """Calculate all performance metrics.

        Args:
            equity_curve: Series of portfolio values over time
            trades: List of completed trades
            initial_capital: Starting capital

        Returns:
            Dictionary with all metrics
        """
        returns = PerformanceMetrics.calculate_returns(equity_curve)

        metrics = {
            'initial_capital': initial_capital,
            'final_value': equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital,
            'total_return': PerformanceMetrics.calculate_total_return(equity_curve),
            'annualized_return': PerformanceMetrics.calculate_annualized_return(equity_curve),
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
            'max_drawdown': PerformanceMetrics.calculate_max_drawdown(equity_curve),
            'num_trades': len(trades),
            'win_rate': PerformanceMetrics.calculate_win_rate(trades),
            'profit_factor': PerformanceMetrics.calculate_profit_factor(trades),
        }

        # Add trade statistics if trades exist
        if len(trades) > 0:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]

            metrics['avg_trade_pnl'] = np.mean([t.pnl for t in trades])
            metrics['avg_trade_pnl_pct'] = np.mean([t.pnl_pct for t in trades])
            metrics['best_trade'] = max([t.pnl for t in trades])
            metrics['worst_trade'] = min([t.pnl for t in trades])

            if winning_trades:
                metrics['avg_win'] = np.mean([t.pnl for t in winning_trades])
                metrics['avg_win_pct'] = np.mean([t.pnl_pct for t in winning_trades])
            else:
                metrics['avg_win'] = 0.0
                metrics['avg_win_pct'] = 0.0

            if losing_trades:
                metrics['avg_loss'] = np.mean([t.pnl for t in losing_trades])
                metrics['avg_loss_pct'] = np.mean([t.pnl_pct for t in losing_trades])
            else:
                metrics['avg_loss'] = 0.0
                metrics['avg_loss_pct'] = 0.0
        else:
            metrics['avg_trade_pnl'] = 0.0
            metrics['avg_trade_pnl_pct'] = 0.0
            metrics['best_trade'] = 0.0
            metrics['worst_trade'] = 0.0
            metrics['avg_win'] = 0.0
            metrics['avg_win_pct'] = 0.0
            metrics['avg_loss'] = 0.0
            metrics['avg_loss_pct'] = 0.0

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict):
        """Print metrics in a formatted way.

        Args:
            metrics: Dictionary of metrics from calculate_all_metrics
        """
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE METRICS")
        print("="*60)

        print(f"\nCapital:")
        print(f"  Initial Capital:     ${metrics['initial_capital']:,.2f}")
        print(f"  Final Value:         ${metrics['final_value']:,.2f}")

        print(f"\nReturns:")
        print(f"  Total Return:        {metrics['total_return']:.2f}%")
        print(f"  Annualized Return:   {metrics['annualized_return']:.2f}%")

        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:.2f}%")

        print(f"\nTrade Statistics:")
        print(f"  Number of Trades:    {metrics['num_trades']}")
        print(f"  Win Rate:            {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")

        if metrics['num_trades'] > 0:
            print(f"  Avg Trade P&L:       ${metrics['avg_trade_pnl']:.2f} ({metrics['avg_trade_pnl_pct']:.2f}%)")
            print(f"  Best Trade:          ${metrics['best_trade']:.2f}")
            print(f"  Worst Trade:         ${metrics['worst_trade']:.2f}")
            print(f"  Avg Win:             ${metrics['avg_win']:.2f} ({metrics['avg_win_pct']:.2f}%)")
            print(f"  Avg Loss:            ${metrics['avg_loss']:.2f} ({metrics['avg_loss_pct']:.2f}%)")

        print("="*60 + "\n")
