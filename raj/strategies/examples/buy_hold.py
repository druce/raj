"""Simple buy-and-hold strategy example."""
import pandas as pd

from raj.backtest.signals import SignalType
from raj.strategies.base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy-and-hold strategy.

    Buys on the first day and holds throughout the entire period.
    This serves as a baseline for comparing other strategies.
    """

    def __init__(self):
        """Initialize buy-and-hold strategy."""
        super().__init__()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate buy-and-hold signals.

        Buys on the first day and holds thereafter.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series of SignalType values
        """
        signals = pd.Series(SignalType.HOLD, index=data.index)

        # Buy on first day
        if len(data) > 0:
            signals.iloc[0] = SignalType.BUY

        return signals
