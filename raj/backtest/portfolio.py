"""Portfolio management for backtesting."""
import logging
from datetime import datetime
from typing import Dict, List, Optional

from raj.backtest.signals import Position, Signal, SignalType, Trade
from raj.config import DEFAULT_COMMISSION

logger = logging.getLogger(__name__)


class Portfolio:
    """Manages portfolio state during backtesting."""

    def __init__(self, initial_capital: float, commission: float = DEFAULT_COMMISSION):
        """Initialize portfolio.

        Args:
            initial_capital: Starting cash
            commission: Commission rate per trade (e.g., 0.001 for 0.1%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []

    def execute_signal(
        self,
        signal: Signal,
        price: float,
        portfolio_value: Optional[float] = None
    ):
        """Execute a trading signal.

        Args:
            signal: Trading signal
            price: Current price for execution
            portfolio_value: Total portfolio value (for position sizing)
        """
        if signal.signal_type == SignalType.BUY:
            self._execute_buy(signal, price, portfolio_value)
        elif signal.signal_type == SignalType.SELL:
            self._execute_sell(signal, price)
        # HOLD signals don't require any action

    def _execute_buy(
        self,
        signal: Signal,
        price: float,
        portfolio_value: Optional[float] = None
    ):
        """Execute a buy signal.

        Args:
            signal: Buy signal
            price: Purchase price
            portfolio_value: Total portfolio value for position sizing
        """
        # Skip if already have a position
        if signal.symbol in self.positions:
            logger.debug(f"Already have position in {signal.symbol}, skipping buy")
            return

        # Determine quantity
        if signal.quantity is not None:
            quantity = signal.quantity
        else:
            # Default: use a fixed percentage of portfolio value
            # For simplicity, we'll use available cash
            allocation = portfolio_value * 0.05 if portfolio_value else self.cash * 0.05
            quantity = int(allocation / price)

        if quantity <= 0:
            logger.debug(f"Insufficient cash to buy {signal.symbol}")
            return

        # Calculate cost including commission
        cost = quantity * price
        commission_cost = cost * self.commission
        total_cost = cost + commission_cost

        if total_cost > self.cash:
            # Adjust quantity to fit available cash
            max_cost = self.cash / (1 + self.commission)
            quantity = int(max_cost / price)
            if quantity <= 0:
                logger.debug(f"Insufficient cash to buy even 1 share of {signal.symbol}")
                return
            cost = quantity * price
            commission_cost = cost * self.commission
            total_cost = cost + commission_cost

        # Execute the trade
        self.cash -= total_cost

        # Create position
        self.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            quantity=quantity,
            entry_price=price,
            entry_date=signal.date,
            current_price=price
        )

        logger.debug(f"BUY {quantity} shares of {signal.symbol} at ${price:.2f}, "
                    f"commission: ${commission_cost:.2f}")

    def _execute_sell(self, signal: Signal, price: float):
        """Execute a sell signal.

        Args:
            signal: Sell signal
            price: Sale price
        """
        # Skip if no position
        if signal.symbol not in self.positions:
            logger.debug(f"No position in {signal.symbol}, skipping sell")
            return

        position = self.positions[signal.symbol]

        # Determine quantity to sell
        quantity = signal.quantity if signal.quantity is not None else position.quantity

        if quantity > position.quantity:
            quantity = position.quantity

        # Calculate proceeds
        proceeds = quantity * price
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost

        # Add to cash
        self.cash += net_proceeds

        # Calculate P&L
        cost_basis = quantity * position.entry_price
        pnl = net_proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

        # Record trade
        trade = Trade(
            symbol=signal.symbol,
            entry_date=position.entry_date,
            exit_date=signal.date,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=quantity,
            commission=commission_cost,
            pnl=pnl,
            pnl_pct=pnl_pct
        )
        self.trade_history.append(trade)

        # Update or remove position
        if quantity == position.quantity:
            del self.positions[signal.symbol]
        else:
            position.quantity -= quantity

        logger.debug(f"SELL {quantity} shares of {signal.symbol} at ${price:.2f}, "
                    f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%), commission: ${commission_cost:.2f}")

    def update_positions(self, prices: Dict[str, float]):
        """Update current prices for all positions.

        Args:
            prices: Dictionary mapping symbols to current prices
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position object or None
        """
        return self.positions.get(symbol)

    def get_cash(self) -> float:
        """Get current cash balance.

        Returns:
            Cash balance
        """
        return self.cash

    def get_positions_value(self) -> float:
        """Get total market value of all positions.

        Returns:
            Total positions value
        """
        return sum(pos.market_value for pos in self.positions.values())

    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions).

        Returns:
            Total portfolio value
        """
        return self.cash + self.get_positions_value()

    def get_trade_history(self) -> List[Trade]:
        """Get history of completed trades.

        Returns:
            List of Trade objects
        """
        return self.trade_history.copy()

    def close_all_positions(self, prices: Dict[str, float], date: datetime):
        """Close all open positions at given prices.

        Args:
            prices: Dictionary mapping symbols to prices
            date: Date of closing
        """
        symbols_to_close = list(self.positions.keys())

        for symbol in symbols_to_close:
            if symbol in prices:
                signal = Signal(
                    symbol=symbol,
                    date=date,
                    signal_type=SignalType.SELL,
                    quantity=None  # Sell entire position
                )
                self._execute_sell(signal, prices[symbol])
