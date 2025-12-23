"""Interactive chart generation using Plotly."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict

from raj.backtest.engine import BacktestResult
from raj.backtest.signals import SignalType


class ChartGenerator:
    """Generate interactive Plotly charts for backtest results."""

    @staticmethod
    def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve") -> go.Figure:
        """Plot equity curve over time.

        Args:
            equity_curve: Series of portfolio values
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    @staticmethod
    def plot_drawdown(equity_curve: pd.Series, title: str = "Drawdown") -> go.Figure:
        """Plot drawdown over time.

        Args:
            equity_curve: Series of portfolio values
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown percentage
        drawdown = ((equity_curve - running_max) / running_max) * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    @staticmethod
    def plot_returns_distribution(
        returns: pd.Series,
        title: str = "Returns Distribution"
    ) -> go.Figure:
        """Plot distribution of returns.

        Args:
            returns: Series of daily returns
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns.values * 100,  # Convert to percentage
            nbinsx=50,
            name='Returns',
            marker=dict(
                color='blue',
                line=dict(color='white', width=1)
            )
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template='plotly_white'
        )

        return fig

    @staticmethod
    def plot_signals(
        data: pd.DataFrame,
        signals: pd.Series,
        symbol: str,
        title: str = None
    ) -> go.Figure:
        """Plot price with buy/sell signals.

        Args:
            data: OHLCV DataFrame
            signals: Series of SignalType values
            symbol: Stock symbol
            title: Chart title

        Returns:
            Plotly Figure object
        """
        if title is None:
            title = f"{symbol} - Price and Signals"

        fig = go.Figure()

        # Plot price
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))

        # Plot buy signals
        buy_signals = signals[signals == SignalType.BUY]
        if len(buy_signals) > 0:
            buy_dates = buy_signals.index
            buy_prices = data.loc[buy_dates, 'close']

            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='Buy',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(color='darkgreen', width=1)
                )
            ))

        # Plot sell signals
        sell_signals = signals[signals == SignalType.SELL]
        if len(sell_signals) > 0:
            sell_dates = sell_signals.index
            sell_prices = data.loc[sell_dates, 'close']

            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='Sell',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(color='darkred', width=1)
                )
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_dashboard(result: BacktestResult) -> go.Figure:
        """Create comprehensive dashboard with multiple charts.

        Args:
            result: BacktestResult object

        Returns:
            Plotly Figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Equity Curve',
                'Drawdown',
                'Monthly Returns',
                'Returns Distribution'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}]
            ]
        )

        # 1. Equity Curve
        fig.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # 2. Drawdown
        running_max = result.equity_curve.expanding().max()
        drawdown = ((result.equity_curve - running_max) / running_max) * 100

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )

        # 3. Monthly Returns
        returns = result.equity_curve.pct_change().fillna(0)
        monthly_returns = (1 + returns).resample('M').prod() - 1

        fig.add_trace(
            go.Bar(
                x=monthly_returns.index,
                y=monthly_returns.values * 100,
                name='Monthly Returns',
                marker=dict(
                    color=monthly_returns.values,
                    colorscale='RdYlGn',
                    cmin=-5,
                    cmax=5
                )
            ),
            row=2, col=1
        )

        # 4. Returns Distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values * 100,
                nbinsx=50,
                name='Daily Returns',
                marker=dict(color='blue')
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f"{result.strategy_name} - Performance Dashboard",
            showlegend=False,
            height=800,
            template='plotly_white'
        )

        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)

        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)

        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)

        fig.update_xaxes(title_text="Daily Return (%)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)

        return fig

    @staticmethod
    def show_figure(fig: go.Figure):
        """Display a figure.

        Args:
            fig: Plotly Figure object
        """
        fig.show()

    @staticmethod
    def save_figure(fig: go.Figure, filepath: str):
        """Save a figure to HTML file.

        Args:
            fig: Plotly Figure object
            filepath: Output file path
        """
        fig.write_html(filepath)
