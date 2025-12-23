"""Base strategy class defining the strategy interface."""
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd

from raj.backtest.signals import SignalType


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, **parameters):
        """Initialize strategy with parameters.

        Args:
            **parameters: Strategy-specific parameters
        """
        self.parameters = parameters
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals for the entire dataset (vectorized).

        This is the main method that strategies must implement. It should
        analyze the OHLCV data and return a Series of signals for each date.

        Args:
            data: DataFrame with OHLCV data, indexed by date
                  Columns: open, high, low, close, volume

        Returns:
            Series of SignalType values indexed by date
        """
        pass

    def get_parameters(self) -> Dict:
        """Get strategy parameters.

        Returns:
            Dictionary of parameter names to values
        """
        return self.parameters.copy()

    def set_parameters(self, **parameters):
        """Update strategy parameters.

        Args:
            **parameters: Parameters to update
        """
        self.parameters.update(parameters)

    def __repr__(self) -> str:
        return f"{self.name}({self.parameters})"
