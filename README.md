# Raj Trading System

A modular Python trading system for backtesting strategies on US stock data with comprehensive performance metrics, interactive visualizations, and data caching.

## Features

- **Data Management**: Download and cache daily OHLCV data for top US stocks from yfinance
- **Flexible Strategy Framework**: Easy-to-extend base class for creating custom trading strategies
- **Backtesting Engine**: Simulate trading strategies with realistic commission and position tracking
- **Performance Metrics**: Comprehensive metrics including returns, Sharpe ratio, max drawdown, win rate
- **Interactive Visualizations**: Plotly-based charts and dashboards
- **HTML Reports**: Generate detailed HTML reports with charts and metrics
- **Data Caching**: Efficient Parquet-based caching to minimize API calls
- **Signal Generation**: Generate daily buy/sell/hold signals for live trading

## Installation

### Prerequisites

- Python 3.11+
- uv (package manager)

### Setup

```bash
# Clone the repository
cd raj

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

## Quick Start

### 1. Download Data

Download data for the top 20 US stocks by market cap:

```bash
python examples/download_data.py --start-date 2020-01-01
```

### 2. Run Demo

Run the complete demo workflow:

```bash
python main.py
```

### 3. Run a Backtest

```bash
# Basic backtest
python examples/run_backtest.py

# With interactive charts
python examples/run_backtest.py --show-charts

# Generate HTML report
python examples/run_backtest.py --generate-report
```

### 4. Generate Trading Signals

```bash
# Generate signals for today
python examples/generate_signals.py

# Save signals to JSON
python examples/generate_signals.py --output-json
```

### 5. Explore in Jupyter

```bash
jupyter notebook notebooks/strategy_exploration.ipynb
```

## Project Structure

```
raj/
├── raj/                         # Main package
│   ├── config.py               # Configuration constants
│   ├── data/                   # Data management
│   │   ├── downloader.py       # YFinance data downloader
│   │   ├── storage.py          # Parquet caching
│   │   ├── universe.py         # Stock universe management
│   │   └── provider.py         # Unified data interface
│   ├── strategies/             # Trading strategies
│   │   ├── base.py             # Abstract base class
│   │   └── examples/           # Example strategies
│   ├── backtest/               # Backtesting engine
│   │   ├── engine.py           # Core backtesting logic
│   │   ├── portfolio.py        # Portfolio management
│   │   ├── metrics.py          # Performance metrics
│   │   └── signals.py          # Signal data structures
│   ├── visualization/          # Charts and reports
│   │   ├── charts.py           # Plotly charts
│   │   └── reports.py          # HTML reports
│   └── utils/                  # Utilities
│       └── dates.py            # Date utilities
├── data/                       # Data storage (gitignored)
│   ├── cache/                  # Cached OHLCV data
│   └── universes/              # Stock universe lists
├── examples/                   # Usage examples
│   ├── download_data.py        # Download script
│   ├── run_backtest.py         # Backtest script
│   └── generate_signals.py    # Signal generation
└── notebooks/                  # Jupyter notebooks
    └── strategy_exploration.ipynb
```

## Creating Custom Strategies

Create your own strategy by inheriting from `BaseStrategy`:

```python
from raj.strategies.base import BaseStrategy
from raj.backtest.signals import SignalType
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self, parameter1=10, parameter2=20):
        super().__init__(parameter1=parameter1, parameter2=parameter2)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate BUY/SELL/HOLD signals."""
        signals = pd.Series(SignalType.HOLD, index=data.index)

        # Your strategy logic here
        # signals[condition] = SignalType.BUY
        # signals[condition] = SignalType.SELL

        return signals
```

See `notebooks/strategy_exploration.ipynb` for complete examples.

## Command Reference

### Download Data

```bash
python examples/download_data.py [OPTIONS]

Options:
  --start-date TEXT      Start date (YYYY-MM-DD), default: 2020-01-01
  --end-date TEXT        End date (YYYY-MM-DD), default: today
  --universe TEXT        Universe name, default: top20
  --force-refresh        Force re-download even if cached
  --refresh-universe     Refresh universe definition
```

### Run Backtest

```bash
python examples/run_backtest.py [OPTIONS]

Options:
  --start-date TEXT      Start date (YYYY-MM-DD)
  --end-date TEXT        End date (YYYY-MM-DD)
  --universe TEXT        Universe name, default: top20
  --initial-capital FLOAT    Initial capital, default: 100000
  --commission FLOAT     Commission rate, default: 0.001
  --show-charts          Display interactive charts
  --generate-report      Generate HTML report
```

### Generate Signals

```bash
python examples/generate_signals.py [OPTIONS]

Options:
  --universe TEXT        Universe name, default: top20
  --start-date TEXT      Start date for data
  --output-json          Save signals to JSON file
```

## Performance Metrics

The system calculates the following metrics:

- **Total Return**: Overall percentage return
- **Annualized Return**: Compound annual growth rate
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Trade Statistics**: Average P&L, best/worst trades, etc.

## Data Caching

- Data is cached in Parquet format for fast access
- Cache is stored in `data/cache/`
- Metadata tracked in `data/cache/metadata.json`
- Use `--force-refresh` to bypass cache

## Architecture

### Data Flow

```
DataProvider.get_universe_data()
    ↓
UniverseManager.get_top_n_by_market_cap() → yfinance API
    ↓
DataStorage.load_data() → Cache or Download
    ↓
Strategy.generate_signals()
    ↓
BacktestEngine.run()
    ↓
PerformanceMetrics.calculate_all_metrics()
    ↓
ChartGenerator + ReportGenerator
```

### Key Components

- **DataProvider**: Unified interface for data access with smart caching
- **UniverseManager**: Manages stock universes (top N by market cap)
- **BaseStrategy**: Abstract class defining strategy interface
- **BacktestEngine**: Coordinates backtesting workflow
- **Portfolio**: Tracks positions, executes trades, calculates P&L
- **PerformanceMetrics**: Calculates all performance statistics
- **ChartGenerator**: Creates interactive Plotly visualizations
- **ReportGenerator**: Generates comprehensive HTML reports

## Dependencies

- **yfinance**: Stock data download
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **pyarrow**: Parquet file support
- **ta-lib**: Technical analysis indicators
- **jupyter**: Interactive notebooks

## License

This project is for educational purposes.
