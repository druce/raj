"""Backtesting engine."""
import logging
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from raj.backtest.metrics import PerformanceMetrics
from raj.backtest.portfolio import Portfolio
from raj.backtest.signals import Signal, SignalType, Trade
from raj.config import DEFAULT_COMMISSION, DEFAULT_INITIAL_CAPITAL
from raj.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a backtest."""
    strategy_name: str
    equity_curve: pd.Series
    trades: List[Trade]
    metrics: Dict
    signals: Dict[str, pd.Series]  # symbol -> signals


class BacktestEngine:
    """Execute backtests for trading strategies."""

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        commission: float = DEFAULT_COMMISSION
    ):
        """Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission

    def run(
        self,
        strategy: BaseStrategy,
        data: Dict[str, pd.DataFrame],
        show_progress: bool = True
    ) -> BacktestResult:
        """Run backtest on multiple stocks.

        Args:
            strategy: Trading strategy to test
            data: Dictionary mapping symbols to OHLCV DataFrames
            show_progress: Whether to show progress bar

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Running backtest for {strategy.name}")
        logger.info(f"Symbols: {list(data.keys())}")

        # Initialize portfolio
        portfolio = Portfolio(self.initial_capital, self.commission)

        # Generate signals for all symbols
        signals = {}
        for symbol, df in data.items():
            try:
                signals[symbol] = strategy.generate_signals(df)
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
                continue

        # Get common date index (dates present in all stocks)
        date_indices = [df.index for df in data.values()]
        if not date_indices:
            raise ValueError("No data provided")

        common_dates = date_indices[0]
        for idx in date_indices[1:]:
            common_dates = common_dates.intersection(idx)

        common_dates = sorted(common_dates)

        if len(common_dates) == 0:
            raise ValueError("No common dates across all symbols")

        # Track equity over time
        equity_curve = []

        # Iterate through each date
        iterator = tqdm(common_dates, desc="Backtesting") if show_progress else common_dates

        for date in iterator:
            # Get current prices for all symbols
            current_prices = {
                symbol: data[symbol].loc[date, 'close']
                for symbol in data.keys()
                if date in data[symbol].index
            }

            # Update positions with current prices
            portfolio.update_positions(current_prices)

            # Get current portfolio value
            portfolio_value = portfolio.get_total_value()

            # Execute signals for this date
            for symbol in data.keys():
                if date in signals[symbol].index:
                    signal_type = signals[symbol].loc[date]

                    signal = Signal(
                        symbol=symbol,
                        date=date,
                        signal_type=signal_type,
                        quantity=None  # Let portfolio determine quantity
                    )

                    if symbol in current_prices:
                        portfolio.execute_signal(
                            signal,
                            current_prices[symbol],
                            portfolio_value
                        )

            # Record equity at end of day
            equity_curve.append({
                'date': date,
                'equity': portfolio.get_total_value()
            })

        # Close all positions at end
        final_date = common_dates[-1]
        final_prices = {
            symbol: data[symbol].loc[final_date, 'close']
            for symbol in data.keys()
            if final_date in data[symbol].index
        }
        portfolio.close_all_positions(final_prices, final_date)

        # Convert equity curve to Series
        equity_df = pd.DataFrame(equity_curve)
        equity_series = pd.Series(
            equity_df['equity'].values,
            index=equity_df['date']
        )

        # Calculate metrics
        trades = portfolio.get_trade_history()
        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_series,
            trades,
            self.initial_capital
        )

        return BacktestResult(
            strategy_name=strategy.name,
            equity_curve=equity_series,
            trades=trades,
            metrics=metrics,
            signals=signals
        )

    def run_single_stock(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: pd.DataFrame,
        show_progress: bool = True
    ) -> BacktestResult:
        """Run backtest on a single stock.

        Args:
            strategy: Trading strategy to test
            symbol: Stock symbol
            data: OHLCV DataFrame
            show_progress: Whether to show progress bar

        Returns:
            BacktestResult with performance metrics
        """
        return self.run(strategy, {symbol: data}, show_progress)
