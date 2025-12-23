"""Signal types and data structures for backtesting."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional


class SignalType(Enum):
    """Type of trading signal."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Signal:
    """Trading signal for a specific stock."""
    symbol: str
    date: datetime
    signal_type: SignalType
    quantity: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Signal({self.symbol}, {self.date.date()}, {self.signal_type.name}, qty={self.quantity})"


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    commission: float
    pnl: float
    pnl_pct: float

    def __repr__(self) -> str:
        return (f"Trade({self.symbol}, {self.entry_date.date()} -> {self.exit_date.date()}, "
                f"PnL: ${self.pnl:.2f} ({self.pnl_pct:.2f}%))")


@dataclass
class Position:
    """Current position in a stock."""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    current_price: float

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss percentage."""
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

    def __repr__(self) -> str:
        return (f"Position({self.symbol}, qty={self.quantity}, "
                f"entry=${self.entry_price:.2f}, current=${self.current_price:.2f}, "
                f"PnL={self.unrealized_pnl_pct:.2f}%)")
