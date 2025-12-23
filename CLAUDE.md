# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Raj is a modular Python trading system for backtesting strategies on US stock data. The system downloads daily OHLCV data for the top 20 US stocks by market cap from yfinance, provides a flexible framework for creating trading strategies, runs backtests with comprehensive performance metrics, and generates interactive visualizations.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # Unix/macOS
```

### Running the Application
```bash
# Run demo workflow
python main.py

# Download data for top 20 stocks
python examples/download_data.py --start-date 2020-01-01

# Run backtest with visualizations
python examples/run_backtest.py --show-charts --generate-report

# Generate trading signals
python examples/generate_signals.py --output-json
```

### Jupyter Notebooks
```bash
# Start Jupyter Notebook
jupyter notebook notebooks/strategy_exploration.ipynb
```

### Package Management
This project uses `uv` (not pip or poetry):
- Add dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

## Architecture

### Module Structure

The codebase is organized into distinct layers:

**Data Layer** (`raj/data/`):
- `downloader.py` - YFinance API integration for downloading stock data
- `storage.py` - Parquet-based caching system with metadata tracking
- `universe.py` - Stock universe management (e.g., top 20 by market cap)
- `provider.py` - **Facade** that unifies downloader, storage, and universe manager

**Strategy Layer** (`raj/strategies/`):
- `base.py` - Abstract `BaseStrategy` class defining the strategy interface
- `examples/buy_hold.py` - Example buy-and-hold strategy for reference
- Custom strategies inherit from `BaseStrategy` and implement `generate_signals()`

**Backtesting Layer** (`raj/backtest/`):
- `signals.py` - Data structures: `SignalType`, `Signal`, `Trade`, `Position`
- `portfolio.py` - Position tracking, trade execution, commission calculation
- `metrics.py` - Performance calculations (returns, Sharpe, drawdown, win rate)
- `engine.py` - **Core coordinator** that runs backtests end-to-end

**Visualization Layer** (`raj/visualization/`):
- `charts.py` - Plotly-based interactive charts (equity curve, drawdown, signals)
- `reports.py` - HTML report generation with embedded charts and metrics

### Key Design Patterns

1. **Facade Pattern**: `DataProvider` provides a simple interface hiding the complexity of downloader, storage, and universe management
2. **Strategy Pattern**: `BaseStrategy` abstract class allows pluggable trading strategies
3. **Repository Pattern**: `DataStorage` abstracts storage mechanism (easy to swap Parquet for other formats)
4. **Dependency Injection**: Strategies receive data as input, don't fetch it themselves

### Data Flow

```
User Request
    ↓
DataProvider.get_universe_data("top20")
    ↓
UniverseManager.get_top_n_by_market_cap(20) → [cached or fetch from yfinance]
    ↓
For each symbol: DataStorage.load_data() → [cache hit or DataDownloader.download_stock()]
    ↓
Strategy.generate_signals(data) → Returns pd.Series of SignalType
    ↓
BacktestEngine.run(strategy, data)
    ├─> Portfolio.execute_signal() for each date
    └─> Track equity curve over time
    ↓
PerformanceMetrics.calculate_all_metrics()
    ↓
ChartGenerator.create_dashboard() + ReportGenerator.generate_backtest_report()
```

### Critical Files and Their Roles

**Data Infrastructure** (Phase 1):
- `raj/config.py` - All configuration constants (paths, defaults)
- `raj/data/provider.py` - **Most important data interface** - use this for all data access
- `raj/data/storage.py` - Parquet caching with smart date range queries
- `raj/data/universe.py` - Manages top 20 stock list with caching

**Strategy Framework** (Phase 2):
- `raj/strategies/base.py` - **All strategies must inherit from this**
- `raj/backtest/signals.py` - Signal types: BUY = 1, SELL = -1, HOLD = 0

**Backtesting Core** (Phase 3):
- `raj/backtest/engine.py` - **Main backtesting coordinator**
- `raj/backtest/portfolio.py` - Tracks positions, executes trades with commission
- `raj/backtest/metrics.py` - Calculates Sharpe, drawdown, win rate, etc.

**User-Facing Scripts**:
- `examples/download_data.py` - CLI for downloading data
- `examples/run_backtest.py` - CLI for running backtests
- `examples/generate_signals.py` - CLI for generating daily signals
- `main.py` - Complete demo workflow

## Key Implementation Details

### Data Caching Strategy
- Data stored as **Parquet files** in `data/cache/` (one file per symbol)
- Metadata tracked in `data/cache/metadata.json` with timestamps and date ranges
- `DataStorage.is_cached()` checks if date range is covered before downloading
- Use `force_refresh=True` to bypass cache

### Strategy Interface
All strategies must implement:
```python
def generate_signals(self, data: pd.DataFrame) -> pd.Series:
    """
    Args:
        data: DataFrame with columns [open, high, low, close, volume]
              indexed by date
    Returns:
        Series of SignalType (BUY/SELL/HOLD) indexed by date
    """
```

### Portfolio Management
- Tracks cash and positions (symbol → Position object)
- Default position sizing: 5% of portfolio value per stock
- Commission applied to both buys and sells
- All positions closed at end of backtest for final P&L calculation

### Performance Metrics
Key metrics calculated:
- **Total Return**: (final_value / initial_capital - 1) * 100
- **Sharpe Ratio**: Annualized, assumes risk-free rate = 0
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### Signal Generation Workflow
1. Load historical data up to today
2. Generate signals for entire dataset
3. Extract signal for most recent date
4. Output to console and/or JSON file

## Common Development Tasks

### Adding a New Strategy
1. Create file in `raj/strategies/` (e.g., `my_strategy.py`)
2. Inherit from `BaseStrategy`
3. Implement `generate_signals(data: pd.DataFrame) -> pd.Series`
4. Use `self.parameters` for strategy parameters
5. Test in `notebooks/strategy_exploration.ipynb`

### Adding a New Performance Metric
1. Add static method to `PerformanceMetrics` class
2. Call from `calculate_all_metrics()` method
3. Update `print_metrics()` if displaying to console
4. Update HTML report template if including in reports

### Modifying Data Source
The system is designed to support multiple data sources:
1. Create new downloader class with same interface as `DataDownloader`
2. Inject into `DataProvider.__init__()`
3. No changes needed in strategy or backtest layers

### Debugging Tips
- Set logging level to DEBUG: `logging.basicConfig(level=logging.DEBUG)`
- Check cache status: `provider.storage.get_cache_info(symbol)`
- Inspect signals: Strategy returns a Series you can examine before backtesting
- View trades: `result.trades` contains all executed trades with P&L

## Data Organization

**Cached Data** (`data/cache/`):
- One `.parquet` file per symbol (e.g., `AAPL.parquet`)
- `metadata.json` tracks download timestamps and date ranges
- Gitignored (don't commit to repository)

**Universe Definitions** (`data/universes/`):
- JSON files with universe name and date (e.g., `top20_2025-12-23.json`)
- Auto-refreshed every 7 days (configurable in `config.py`)
- Gitignored

**Reports** (`reports/`):
- HTML files with embedded Plotly charts
- Created by `ReportGenerator.generate_backtest_report()`
- Gitignored

## Testing Strategy

When testing changes:
1. Start with a small date range (e.g., 1-2 years)
2. Test on a single stock first: `engine.run_single_stock(strategy, symbol, data)`
3. Verify signals look correct before running full backtest
4. Check metrics make sense (Sharpe ratio typically -3 to +3, win rate 30-70%)
5. Use notebooks for interactive debugging

## Dependencies

**Core**:
- `yfinance` - Stock data download (may have rate limits)
- `pandas` - All data is in DataFrames
- `numpy` - Used in metrics calculations
- `pyarrow` - Required for Parquet file support

**Visualization**:
- `plotly` - All charts use Plotly (not matplotlib)

**Development**:
- `jupyter` - For notebook-based exploration
- `ta-lib` - Available for technical indicators (optional)

## Important Constraints

1. **Long-only**: Current implementation only supports long positions (no shorting)
2. **Daily bars**: No intraday data support
3. **Equal weighting**: Default position sizing is equal weight (5% per stock)
4. **No slippage**: Orders execute at close price (can be extended)
5. **Fixed commission**: Percentage-based commission (no per-share or minimum)
6. **YFinance limits**: API may rate-limit; use caching to minimize calls

## Extension Points

The system is designed for easy extension:

**New Strategy**: Inherit `BaseStrategy`, implement `generate_signals()`
**New Data Source**: Implement downloader interface, inject into `DataProvider`
**New Metrics**: Add methods to `PerformanceMetrics`
**New Asset Class**: Data structures are general (OHLCV), may need commission adjustments
**Position Sizing**: Override quantity calculation in `Portfolio._execute_buy()`
**Order Types**: Currently market-on-close, can add limit orders to `Portfolio`
