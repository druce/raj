# Implementation Plan: Trading System

**Document Version:** 1.0
**Created:** 2026-01-06
**Status:** Planning Phase

## Executive Summary

This document provides a detailed, phase-by-phase implementation plan for the backtesting system. The plan follows the architectural principles outlined in ARCHITECTURE.md and breaks the work into 5 discrete phases with clear dependencies, success criteria, and testing strategies.

### Quick Reference

| Phase | Module | Dependencies | Estimated Complexity |
|-------|--------|--------------|---------------------|
| 1 | Data (`Backtester/data/`) | None | High (multi-source integration) |
| 2 | Strategy (`Backtester/strategies/`) | Phase 1 | Medium (vectorization patterns) |
| 3 | Backtest (`Backtester/backtest/`) | Phases 1, 2 | High (simulation accuracy) |
| 4 | Visualization (`Backtester/visualization/`) | Phase 3 | Low (formatting/presentation) |
| 5 | Utils (`Backtester/utils.py`) | All phases | Low (extraction/consolidation) |

### Implementation Principles

1. **Test-First Development**: Write tests before implementation
2. **Fixture-Based Validation**: Store known-good data for regression testing
3. **Vectorization Priority**: Leverage pandas/numpy for performance
4. **Clean Interfaces**: Each module exposes minimal, well-defined APIs
5. **Incremental Integration**: Each phase produces independently testable outputs



### Directory Structure 



```tradingsystem/
├── README.md                           # Project overview and quick start
├── ARCHITECTURE.md                     # Architectural design document
├── IMPLEMENTATION.md                   # Detailed implementation plan
├── DIRECTORY_STRUCTURE.md              # This file
├── CLAUDE.md                           # Claude Code instructions
│
├── pyproject.toml                      # uv package configuration
├── uv.lock                             # uv lock file
├── .gitignore                          # Git ignore rules
├── .python-version                     # Python version specification
│
├── Backtester/                         # Main package
│   ├── __init__.py
│   ├── config.py                       # Configuration constants
│   │
│   ├── data/                           # Phase 1: Data Layer
│   │   ├── __init__.py
│   │   ├── downloader.py               # BaseDownloader, YFinanceDownloader, SchwabDownloader
│   │   ├── storage.py                  # DataStorage (Parquet cache)
│   │   ├── universe.py                 # UniverseManager (top N stocks)
│   │   └── provider.py                 # DataProvider (facade)
│   │
│   ├── strategies/                     # Phase 2: Strategy Layer
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseStrategy (abstract)
│   │   ├── ma_crossover.py             # MovingAverageCrossover strategy
│   │   ├── rsi_strategy.py             # RSIMeanReversion strategy
│   │   └── buy_hold.py                 # BuyAndHold (benchmark)
│   │
│   ├── backtest/                       # Phase 3: Backtest Layer
│   │   ├── __init__.py
│   │   ├── signals.py                  # SignalType, Position, Trade dataclasses
│   │   ├── portfolio.py                # Portfolio (position/cash management)
│   │   ├── engine.py                   # BacktestEngine, BacktestResult
│   │   └── metrics.py                  # PerformanceMetrics
│   │
│   ├── visualization/                  # Phase 4: Visualization Layer
│   │   ├── __init__.py
│   │   ├── charts.py                   # ChartGenerator (Plotly charts)
│   │   └── reports.py                  # ReportGenerator (HTML reports)
│   │
│   └── utils.py                        # Phase 5: Shared utilities
│
├── data/                               # Data storage (gitignored)
│   ├── cache/                          # Cached OHLCV data
│   │   ├── metadata.json               # Cache metadata tracking
│   │   ├── AAPL_yfinance.parquet       # Per-symbol Parquet files
│   │   ├── MSFT_yfinance.parquet
│   │   ├── GOOGL_yfinance.parquet
│   │   └── ...
│   │
│   └── universes/                      # Stock universe definitions
│       ├── top20.json                  # Top 20 stocks by market cap
│       └── top50.json                  # Top 50 stocks
│
├── reports/                            # Generated reports (gitignored)
│   ├── backtest_20260106_143022/       # Timestamped backtest results
│   │   ├── metadata.json               # Strategy params, metrics, run info
│   │   ├── equity_curve.parquet        # Equity curve time series
│   │   ├── trades.parquet              # Trade history
│   │   └── signals/                    # Per-symbol signals
│   │       ├── AAPL.parquet
│   │       ├── MSFT.parquet
│   │       └── ...
│   │
│   ├── backtest_20260106_143022.html   # HTML report with charts
│   └── comparison_ma_vs_buyhold.html   # Strategy comparison report
│
├── examples/                           # Example scripts
│   ├── download_data.py                # Download and cache data
│   ├── run_backtest.py                 # Run backtest with CLI args
│   ├── generate_signals.py             # Generate trading signals for today
│   └── compare_strategies.py           # Compare multiple strategies
│
├── notebooks/                          # Jupyter notebooks
│   ├── strategy_exploration.ipynb      # Interactive strategy development
│   ├── data_analysis.ipynb             # Data quality analysis
│   └── backtest_analysis.ipynb         # Backtest result deep dive
│
├── tests/                              # Test suite
│   ├── __init__.py
│   │
│   ├── fixtures/                       # Test fixtures (known-good data)
│   │   ├── aapl_2020_2024.parquet      # Historical AAPL data (5 years)
│   │   ├── msft_2020_2024.parquet      # Historical MSFT data
│   │   ├── ma_50_200_signals_expected.csv          # Expected MA signals
│   │   ├── rsi_14_30_70_signals_expected.csv       # Expected RSI signals
│   │   ├── ma_crossover_backtest_expected.json     # Expected backtest metrics
│   │   └── sample_ohlcv.csv            # Small sample for unit tests
│   │
│   ├── test_data/                      # Phase 1 tests
│   │   ├── __init__.py
│   │   ├── test_downloader.py          # Test YFinance/Schwab downloaders
│   │   ├── test_storage.py             # Test Parquet caching
│   │   ├── test_universe.py            # Test universe management
│   │   ├── test_provider.py            # Test DataProvider facade
│   │   ├── test_data_quality.py        # Test data validation
│   │   └── test_source_comparison.py   # Compare yfinance vs Schwab
│   │
│   ├── test_strategies/                # Phase 2 tests
│   │   ├── __init__.py
│   │   ├── test_base_strategy.py       # Test abstract base class
│   │   ├── test_ma_crossover.py        # Test MA crossover strategy
│   │   ├── test_rsi_strategy.py        # Test RSI strategy
│   │   └── test_buy_hold.py            # Test buy-and-hold benchmark
│   │
│   ├── test_backtest/                  # Phase 3 tests
│   │   ├── __init__.py
│   │   ├── test_signals.py             # Test SignalType, Position, Trade
│   │   ├── test_portfolio.py           # Test Portfolio logic
│   │   ├── test_engine.py              # Test BacktestEngine
│   │   └── test_metrics.py             # Test metric calculations
│   │
│   ├── test_visualization/             # Phase 4 tests
│   │   ├── __init__.py
│   │   ├── test_charts.py              # Test chart generation
│   │   └── test_reports.py             # Test HTML report generation
│   │
│   ├── test_utils.py                   # Phase 5 tests
│   │
│   ├── integration/                    # Integration tests
│   │   ├── __init__.py
│   │   ├── test_full_pipeline.py       # End-to-end workflow
│   │   └── test_multi_strategy.py      # Multiple strategy backtests
│   │
│   └── performance/                    # Performance benchmarks
│       ├── __init__.py
│       ├── benchmark_vectorization.py  # Signal generation speed
│       └── benchmark_backtest.py       # Backtest execution speed
│
├── scripts/                            # Utility scripts
│   ├── setup_environment.sh            # Environment setup
│   ├── run_all_tests.sh                # Run full test suite
│   ├── generate_fixtures.py            # Generate test fixtures
│   └── clean_cache.py                  # Clean data cache
│
├── docs/                               # Documentation
│   ├── getting_started.md              # Quick start guide
│   ├── api_reference.md                # API documentation
│   ├── strategy_guide.md               # How to write strategies
│   └── examples.md                     # Usage examples
│
└── main.py                             # Entry point / demo workflow
```



---

## Table of Contents

1. [Phase 1: Data Module](#phase-1-data-module)
2. [Phase 2: Strategy Module](#phase-2-strategy-module)
3. [Phase 3: Backtest Module](#phase-3-backtest-module)
4. [Phase 4: Visualization Module](#phase-4-visualization-module)
5. [Phase 5: Utils Module](#phase-5-utils-module)
6. [Implementation Order & Dependencies](#implementation-order--dependencies)
7. [Success Criteria](#success-criteria-by-phase)
8. [Testing Philosophy](#testing-philosophy)

---

## Phase 1: Data Module

**Location:** `Backtester/data/`
**Goal:** Robust data acquisition and caching system supporting yfinance and Schwab
**Dependencies:** None (foundational layer)

### 1.1 Core Components

#### `downloader.py` - Multi-source data downloading

```python
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
import yfinance as yf
import time
import logging

class BaseDownloader(ABC):
    """Abstract interface for data sources"""

    @abstractmethod
    def download_stock(self, symbol: str, start_date: datetime,
                      end_date: datetime) -> pd.DataFrame:
        """
        Download OHLCV data for a single symbol

        Returns:
            DataFrame with columns [open, high, low, close, volume]
            indexed by date (DatetimeIndex)
        """
        pass

    @abstractmethod
    def validate_response(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate downloaded data meets quality standards"""
        pass

class YFinanceDownloader(BaseDownloader):
    """YFinance API integration with rate limiting"""

    def __init__(self, rate_limit_delay: float = 0.5, max_retries: int = 3):
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    def download_stock(self, symbol: str, start_date: datetime,
                      end_date: datetime) -> pd.DataFrame:
        """
        Download from yfinance with exponential backoff retry logic
        """
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)

                # Download
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=True  # Adjust for splits/dividends
                )

                # Normalize column names to lowercase
                data.columns = data.columns.str.lower()

                # Validate
                is_valid, errors = self.validate_response(data)
                if not is_valid:
                    raise ValueError(f"Invalid data: {errors}")

                return data[['open', 'high', 'low', 'close', 'volume']]

            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for {symbol}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def validate_response(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate OHLCV schema and data quality"""
        errors = []

        # Check columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(data.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")

        # Check not empty
        if len(data) == 0:
            errors.append("No data returned")

        # Check index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("Index is not DatetimeIndex")

        # Check for NaN values
        if data[required_cols].isnull().any().any():
            errors.append("Contains NaN values")

        # Check OHLC relationships
        if len(data) > 0:
            high_low_valid = (data['high'] >= data['low']).all()
            if not high_low_valid:
                errors.append("High < Low detected")

            close_in_range = (
                (data['close'] >= data['low']) &
                (data['close'] <= data['high'])
            ).all()
            if not close_in_range:
                errors.append("Close outside [low, high] range")

        return len(errors) == 0, errors

class SchwabDownloader(BaseDownloader):
    """Schwab API integration"""

    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize with Schwab API credentials

        Note: Actual implementation depends on Schwab API specifics
        This is a placeholder showing the interface
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = None
        self.token_expiry = None
        self.logger = logging.getLogger(__name__)

    def _refresh_token(self):
        """Refresh OAuth token if expired"""
        # TODO: Implement Schwab OAuth flow
        pass

    def download_stock(self, symbol: str, start_date: datetime,
                      end_date: datetime) -> pd.DataFrame:
        """
        Download from Schwab API

        Important: Unlike yfinance, Schwab may not auto-adjust for splits.
        We may need to apply adjustments manually using corporate actions data.
        """
        self._refresh_token()

        # TODO: Implement actual Schwab API call
        # Pseudo-code:
        # 1. Make API request to Schwab endpoint
        # 2. Parse response (likely JSON)
        # 3. Convert to DataFrame
        # 4. Normalize column names
        # 5. Apply split/dividend adjustments if needed
        # 6. Validate and return

        raise NotImplementedError("Schwab API integration pending")

    def validate_response(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Same validation as YFinance"""
        # Can reuse YFinanceDownloader.validate_response logic
        # or create shared validation function
        pass
```

**Implementation Notes:**
- YFinance provides adjusted prices automatically (splits/dividends already incorporated)
- Schwab may require manual adjustment - need to fetch corporate actions and apply
- Rate limiting is critical to avoid API bans
- Exponential backoff handles transient network issues

#### `storage.py` - Parquet-based caching with metadata

```python
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List
import logging

class DataStorage:
    """Parquet-based data cache with metadata tracking"""

    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / 'metadata.json'
        self.metadata = self._load_metadata()
        self.logger = logging.getLogger(__name__)

    def _load_metadata(self) -> Dict:
        """Load cache metadata from JSON file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Persist metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def save_data(self, symbol: str, data: pd.DataFrame, source: str = 'yfinance'):
        """
        Save OHLCV data to Parquet with metadata tracking

        Args:
            symbol: Stock ticker
            data: DataFrame with OHLCV columns
            source: Data provider name ('yfinance' or 'schwab')
        """
        # Validate data before saving
        is_valid, errors = self.validate_data_integrity(data)
        if not is_valid:
            raise ValueError(f"Cannot save invalid data for {symbol}: {errors}")

        # Save to Parquet
        file_path = self._get_file_path(symbol, source)
        data.to_parquet(file_path, compression='snappy', index=True)

        # Update metadata
        self.metadata[symbol] = {
            'source': source,
            'start_date': data.index.min().isoformat(),
            'end_date': data.index.max().isoformat(),
            'num_rows': len(data),
            'cached_at': datetime.now().isoformat(),
            'file_path': str(file_path)
        }
        self._save_metadata()

        self.logger.info(
            f"Cached {symbol} ({len(data)} rows) from {source} to {file_path}"
        )

    def load_data(self, symbol: str, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  source: str = 'yfinance') -> pd.DataFrame:
        """
        Load data from cache, optionally filtering by date range

        Args:
            symbol: Stock ticker
            start_date: Optional filter (inclusive)
            end_date: Optional filter (inclusive)
            source: Data provider name

        Returns:
            DataFrame with OHLCV data
        """
        file_path = self._get_file_path(symbol, source)

        if not file_path.exists():
            raise FileNotFoundError(f"No cached data for {symbol} from {source}")

        # Load from Parquet
        data = pd.read_parquet(file_path)

        # Filter by date range if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        return data

    def is_cached(self, symbol: str, start_date: datetime,
                  end_date: datetime, source: str = 'yfinance') -> bool:
        """
        Check if requested date range is fully covered by cache

        Returns:
            True if cache covers entire range, False otherwise
        """
        if symbol not in self.metadata:
            return False

        meta = self.metadata[symbol]

        # Check source matches
        if meta['source'] != source:
            return False

        # Check date coverage
        cached_start = datetime.fromisoformat(meta['start_date'])
        cached_end = datetime.fromisoformat(meta['end_date'])

        # Cache must fully contain requested range
        return cached_start <= start_date and cached_end >= end_date

    def get_cache_info(self, symbol: str) -> Optional[Dict]:
        """Get metadata for cached symbol"""
        return self.metadata.get(symbol)

    def clear_cache(self, symbol: Optional[str] = None,
                   before_date: Optional[datetime] = None):
        """
        Clear cache entries

        Args:
            symbol: If specified, clear only this symbol. Otherwise clear all.
            before_date: If specified, clear entries cached before this date
        """
        if symbol:
            # Clear specific symbol
            if symbol in self.metadata:
                file_path = Path(self.metadata[symbol]['file_path'])
                if file_path.exists():
                    file_path.unlink()
                del self.metadata[symbol]
                self._save_metadata()
                self.logger.info(f"Cleared cache for {symbol}")
        else:
            # Clear all matching before_date
            to_remove = []
            for sym, meta in self.metadata.items():
                cached_at = datetime.fromisoformat(meta['cached_at'])
                if before_date is None or cached_at < before_date:
                    file_path = Path(meta['file_path'])
                    if file_path.exists():
                        file_path.unlink()
                    to_remove.append(sym)

            for sym in to_remove:
                del self.metadata[sym]

            self._save_metadata()
            self.logger.info(f"Cleared {len(to_remove)} cache entries")

    def validate_data_integrity(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Comprehensive data quality validation

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Schema validation
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(data.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")
            return False, errors

        # Empty data
        if len(data) == 0:
            errors.append("Empty DataFrame")
            return False, errors

        # OHLC relationship validation
        if not (data['high'] >= data['low']).all():
            errors.append("High < Low detected")

        if not (data['high'] >= data['open']).all():
            errors.append("High < Open detected")

        if not (data['high'] >= data['close']).all():
            errors.append("High < Close detected")

        if not (data['low'] <= data['open']).all():
            errors.append("Low > Open detected")

        if not (data['low'] <= data['close']).all():
            errors.append("Low > Close detected")

        # Check for negative values (prices should be positive)
        for col in ['open', 'high', 'low', 'close']:
            if (data[col] <= 0).any():
                errors.append(f"Non-positive values in {col}")

        # Volume should be non-negative
        if (data['volume'] < 0).any():
            errors.append("Negative volume detected")

        # Check for NaN values
        if data[required_cols].isnull().any().any():
            nan_cols = data[required_cols].columns[data[required_cols].isnull().any()].tolist()
            errors.append(f"NaN values in columns: {nan_cols}")

        return len(errors) == 0, errors

    def _get_file_path(self, symbol: str, source: str) -> Path:
        """Generate file path for symbol"""
        # Include source in filename to allow both yfinance and schwab caches
        return self.cache_dir / f"{symbol}_{source}.parquet"
```

**Key Features:**
- Metadata tracks date ranges to enable smart cache hits
- Parquet provides ~80% compression vs CSV
- Snappy compression for fast read/write
- Comprehensive data validation before saving
- Support for multiple data sources per symbol

#### `universe.py` - Stock universe management

```python
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

class UniverseManager:
    """Manages stock universes (e.g., top 20 by market cap)"""

    def __init__(self, universe_dir: str = 'data/universes'):
        self.universe_dir = Path(universe_dir)
        self.universe_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def get_top_n_by_market_cap(self, n: int = 20,
                               refresh_days: int = 7) -> List[str]:
        """
        Get top N US stocks by market cap

        Args:
            n: Number of stocks to return
            refresh_days: Re-fetch if cached data older than this

        Returns:
            List of stock symbols
        """
        universe_name = f'top{n}'

        # Check cache
        if not self.is_universe_stale(universe_name, refresh_days):
            return self.load_universe(universe_name)

        # Fetch fresh data
        self.logger.info(f"Fetching fresh top {n} stocks by market cap")
        symbols = self._fetch_top_n_stocks(n)

        # Save to cache
        self.save_universe(universe_name, symbols, {
            'criteria': 'market_cap',
            'n': n,
            'method': 'yfinance_screener'
        })

        return symbols

    def _fetch_top_n_stocks(self, n: int) -> List[str]:
        """
        Fetch top N stocks using yfinance or other source

        Note: This is a placeholder. Actual implementation could:
        1. Use yfinance screener
        2. Scrape from public lists (S&P 500, etc.)
        3. Use financial data API

        For now, returns a hardcoded list of large-cap stocks
        """
        # Hardcoded top 20 US stocks (as of 2024)
        # In production, this should fetch dynamically
        top_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK-B', 'V', 'UNH',
            'JNJ', 'WMT', 'JPM', 'MA', 'PG',
            'XOM', 'HD', 'CVX', 'LLY', 'ABBV'
        ]

        return top_stocks[:n]

    def save_universe(self, name: str, symbols: List[str],
                     metadata: Dict):
        """
        Save universe to JSON file

        Args:
            name: Universe identifier (e.g., 'top20')
            symbols: List of stock tickers
            metadata: Additional info (criteria, source, etc.)
        """
        file_path = self._get_universe_file_path(name)

        data = {
            'name': name,
            'symbols': symbols,
            'metadata': metadata,
            'created_at': datetime.now().isoformat(),
            'num_symbols': len(symbols)
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved universe '{name}' with {len(symbols)} symbols")

    def load_universe(self, name: str) -> List[str]:
        """Load universe from cache"""
        file_path = self._get_universe_file_path(name)

        if not file_path.exists():
            raise FileNotFoundError(f"Universe '{name}' not found")

        with open(file_path, 'r') as f:
            data = json.load(f)

        return data['symbols']

    def is_universe_stale(self, name: str, max_age_days: int = 7) -> bool:
        """
        Check if universe needs refreshing

        Returns:
            True if universe doesn't exist or is older than max_age_days
        """
        file_path = self._get_universe_file_path(name)

        if not file_path.exists():
            return True

        with open(file_path, 'r') as f:
            data = json.load(f)

        created_at = datetime.fromisoformat(data['created_at'])
        age = datetime.now() - created_at

        return age.days > max_age_days

    def _get_universe_file_path(self, name: str) -> Path:
        """Generate file path for universe"""
        return self.universe_dir / f"{name}.json"
```

**Design Notes:**
- Universe definitions cached as JSON (human-readable)
- Configurable refresh interval (default 7 days)
- Supports multiple universe types (top N, sector-specific, custom)
- Placeholder fetch logic - can be enhanced with real screener API

#### `provider.py` - Facade unifying all data operations

```python
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import logging

from Backtester.data.downloader import YFinanceDownloader, SchwabDownloader
from Backtester.data.storage import DataStorage
from Backtester.data.universe import UniverseManager

class DataProvider:
    """
    Facade providing unified interface to data subsystems

    This is the primary interface for all data access in the system.
    """

    def __init__(self, primary_source: str = 'yfinance',
                 fallback_source: Optional[str] = None):
        """
        Initialize data provider

        Args:
            primary_source: Default data source ('yfinance' or 'schwab')
            fallback_source: Backup source if primary fails
        """
        self.downloaders = {
            'yfinance': YFinanceDownloader(),
            # 'schwab': SchwabDownloader()  # Uncomment when implemented
        }

        self.primary_source = primary_source
        self.fallback_source = fallback_source

        self.storage = DataStorage()
        self.universe_manager = UniverseManager()
        self.logger = logging.getLogger(__name__)

    def get_universe_data(self, universe_name: str,
                         start_date: datetime,
                         end_date: datetime,
                         source: str = 'auto',
                         force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for all stocks in a universe

        Args:
            universe_name: Universe identifier (e.g., 'top20')
            start_date: Data start date
            end_date: Data end date
            source: 'auto', 'yfinance', or 'schwab'
            force_refresh: Bypass cache if True

        Returns:
            Dict mapping symbol -> DataFrame
        """
        # Parse universe name (e.g., 'top20' -> top 20 stocks)
        if universe_name.startswith('top'):
            n = int(universe_name[3:])
            symbols = self.universe_manager.get_top_n_by_market_cap(n)
        else:
            symbols = self.universe_manager.load_universe(universe_name)

        self.logger.info(
            f"Loading {len(symbols)} stocks for universe '{universe_name}'"
        )

        # Download data for each symbol
        results = {}
        for symbol in symbols:
            try:
                data = self.get_single_stock(
                    symbol, start_date, end_date, source, force_refresh
                )
                results[symbol] = data
            except Exception as e:
                self.logger.error(f"Failed to get data for {symbol}: {e}")
                # Continue with other symbols

        self.logger.info(
            f"Successfully loaded {len(results)}/{len(symbols)} stocks"
        )

        return results

    def get_single_stock(self, symbol: str,
                        start_date: datetime,
                        end_date: datetime,
                        source: str = 'auto',
                        force_refresh: bool = False) -> pd.DataFrame:
        """
        Get OHLCV data for a single stock

        Args:
            symbol: Stock ticker
            start_date: Data start date
            end_date: Data end date
            source: 'auto', 'yfinance', or 'schwab'
            force_refresh: Bypass cache if True

        Returns:
            DataFrame with OHLCV columns
        """
        # Determine source
        if source == 'auto':
            source = self.primary_source

        # Check cache first (unless force_refresh)
        if not force_refresh and self.storage.is_cached(
            symbol, start_date, end_date, source
        ):
            self.logger.debug(f"Cache hit for {symbol}")
            return self.storage.load_data(symbol, start_date, end_date, source)

        # Cache miss - download
        self.logger.info(f"Downloading {symbol} from {source}")

        try:
            downloader = self.downloaders[source]
            data = downloader.download_stock(symbol, start_date, end_date)

            # Save to cache
            self.storage.save_data(symbol, data, source)

            return data

        except Exception as e:
            # Try fallback source if configured
            if self.fallback_source and source != self.fallback_source:
                self.logger.warning(
                    f"Primary source failed for {symbol}, trying {self.fallback_source}"
                )
                return self.get_single_stock(
                    symbol, start_date, end_date,
                    self.fallback_source, force_refresh
                )
            else:
                raise

    def compare_sources(self, symbol: str,
                       start_date: datetime,
                       end_date: datetime) -> pd.DataFrame:
        """
        Compare data from yfinance and Schwab for the same symbol

        Returns:
            DataFrame with columns: date, yf_close, schwab_close, diff, diff_pct
        """
        # Download from both sources
        yf_data = self.get_single_stock(symbol, start_date, end_date, 'yfinance')
        schwab_data = self.get_single_stock(symbol, start_date, end_date, 'schwab')

        # Align dates
        common_dates = yf_data.index.intersection(schwab_data.index)

        comparison = pd.DataFrame({
            'yf_close': yf_data.loc[common_dates, 'close'],
            'schwab_close': schwab_data.loc[common_dates, 'close']
        })

        comparison['diff'] = comparison['schwab_close'] - comparison['yf_close']
        comparison['diff_pct'] = (comparison['diff'] / comparison['yf_close']) * 100

        return comparison
```

**Facade Benefits:**
- Single entry point for all data access
- Hides complexity of downloader/storage/universe subsystems
- Automatic fallback to secondary source
- Cache-first approach minimizes API calls
- Easy to swap implementations (Parquet → database, etc.)

### 1.2 Testing Strategy

#### `tests/test_data/test_downloader.py`

```python
import pytest
import pandas as pd
from datetime import datetime
from Backtester.data.downloader import YFinanceDownloader

class TestYFinanceDownloader:

    def test_download_valid_symbol(self):
        """Download should succeed for known symbols"""
        downloader = YFinanceDownloader()
        data = downloader.download_stock(
            'AAPL',
            datetime(2023, 1, 1),
            datetime(2023, 12, 31)
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_download_invalid_symbol(self):
        """Download should raise error for invalid symbols"""
        downloader = YFinanceDownloader()

        with pytest.raises(Exception):
            downloader.download_stock(
                'INVALIDXXX',
                datetime(2023, 1, 1),
                datetime(2023, 12, 31)
            )

    def test_date_range_validation(self):
        """Start date must be before end date"""
        downloader = YFinanceDownloader()

        with pytest.raises(Exception):
            downloader.download_stock(
                'AAPL',
                datetime(2023, 12, 31),  # End
                datetime(2023, 1, 1)     # Start (reversed!)
            )

    def test_schema_validation(self):
        """Validate OHLCV schema requirements"""
        downloader = YFinanceDownloader()

        # Valid data
        valid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02']))

        is_valid, errors = downloader.validate_response(valid_data)
        assert is_valid
        assert len(errors) == 0

    def test_invalid_ohlc_relationships(self):
        """Detect invalid OHLC relationships"""
        downloader = YFinanceDownloader()

        # Invalid: high < low
        invalid_data = pd.DataFrame({
            'open': [100],
            'high': [95],   # Invalid: lower than low
            'low': [99],
            'close': [101],
            'volume': [1000]
        }, index=pd.DatetimeIndex(['2023-01-01']))

        is_valid, errors = downloader.validate_response(invalid_data)
        assert not is_valid
        assert any('High < Low' in err for err in errors)
```

#### `tests/test_data/test_storage.py`

```python
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from Backtester.data.storage import DataStorage

@pytest.fixture
def temp_storage():
    """Create temporary storage directory for testing"""
    temp_dir = tempfile.mkdtemp()
    storage = DataStorage(cache_dir=temp_dir)
    yield storage
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_data():
    """Sample OHLCV data for testing"""
    return pd.DataFrame({
        'open': [100, 101, 102],
        'high': [102, 103, 104],
        'low': [99, 100, 101],
        'close': [101, 102, 103],
        'volume': [1000, 1100, 1200]
    }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03']))

class TestDataStorage:

    def test_save_and_load_roundtrip(self, temp_storage, sample_data):
        """Data should be preserved through save/load cycle"""
        temp_storage.save_data('AAPL', sample_data)
        loaded = temp_storage.load_data('AAPL')

        pd.testing.assert_frame_equal(sample_data, loaded)

    def test_cache_hit_detection(self, temp_storage, sample_data):
        """is_cached should return True when data covers range"""
        temp_storage.save_data('AAPL', sample_data)

        # Exact range
        assert temp_storage.is_cached(
            'AAPL',
            datetime(2023, 1, 1),
            datetime(2023, 1, 3)
        )

        # Subset range
        assert temp_storage.is_cached(
            'AAPL',
            datetime(2023, 1, 2),
            datetime(2023, 1, 2)
        )

    def test_cache_miss_detection(self, temp_storage, sample_data):
        """is_cached should return False when data doesn't cover range"""
        temp_storage.save_data('AAPL', sample_data)

        # Range extends beyond cached data
        assert not temp_storage.is_cached(
            'AAPL',
            datetime(2022, 12, 1),  # Before cached data
            datetime(2023, 1, 3)
        )

    def test_partial_date_range_load(self, temp_storage, sample_data):
        """Should load subset of cached data"""
        temp_storage.save_data('AAPL', sample_data)

        subset = temp_storage.load_data(
            'AAPL',
            start_date=datetime(2023, 1, 2),
            end_date=datetime(2023, 1, 2)
        )

        assert len(subset) == 1
        assert subset.index[0] == pd.Timestamp('2023-01-02')

    def test_metadata_tracking(self, temp_storage, sample_data):
        """Metadata should track cache info"""
        temp_storage.save_data('AAPL', sample_data, source='yfinance')

        info = temp_storage.get_cache_info('AAPL')
        assert info is not None
        assert info['source'] == 'yfinance'
        assert info['num_rows'] == 3
        assert 'cached_at' in info

    def test_clear_specific_symbol(self, temp_storage, sample_data):
        """Should clear only specified symbol"""
        temp_storage.save_data('AAPL', sample_data)
        temp_storage.save_data('MSFT', sample_data)

        temp_storage.clear_cache(symbol='AAPL')

        assert temp_storage.get_cache_info('AAPL') is None
        assert temp_storage.get_cache_info('MSFT') is not None
```

#### `tests/test_data/test_data_quality.py`

```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from Backtester.data.storage import DataStorage

class TestDataQuality:

    def test_missing_trading_days_detection(self):
        """Detect gaps in trading day sequence"""
        # Create data with 10-day gap
        dates = []
        dates.extend(pd.date_range('2023-01-01', '2023-01-10', freq='B'))
        dates.extend(pd.date_range('2023-01-20', '2023-01-31', freq='B'))

        data = pd.DataFrame({
            'open': np.random.rand(len(dates)) * 100,
            'high': np.random.rand(len(dates)) * 100,
            'low': np.random.rand(len(dates)) * 100,
            'close': np.random.rand(len(dates)) * 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

        # Function to detect gaps
        def detect_gaps(df, max_gap_days=5):
            date_diffs = df.index.to_series().diff()
            gaps = date_diffs[date_diffs > timedelta(days=max_gap_days)]
            return gaps

        gaps = detect_gaps(data)
        assert len(gaps) > 0

    def test_price_spike_detection(self):
        """Flag unrealistic single-day price movements"""
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='B')

        # Normal prices, then huge spike
        close_prices = [100] * len(dates)
        close_prices[10] = 200  # 100% spike

        data = pd.DataFrame({
            'open': close_prices,
            'high': close_prices,
            'low': close_prices,
            'close': close_prices,
            'volume': [1000] * len(dates)
        }, index=dates)

        # Detect spikes
        returns = data['close'].pct_change()
        spikes = returns[returns.abs() > 0.5]  # 50% threshold

        assert len(spikes) > 0

    def test_volume_zero_detection(self):
        """Flag zero volume on trading days"""
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='B')

        data = pd.DataFrame({
            'open': [100] * len(dates),
            'high': [102] * len(dates),
            'low': [99] * len(dates),
            'close': [101] * len(dates),
            'volume': [1000] * len(dates)
        }, index=dates)

        # Set some volumes to zero
        data.loc[data.index[5], 'volume'] = 0

        zero_volume_days = data[data['volume'] == 0]
        assert len(zero_volume_days) > 0

    def test_ohlc_consistency_validation(self):
        """Validate OHLC relationships"""
        storage = DataStorage()

        # Valid data
        valid_data = pd.DataFrame({
            'open': [100],
            'high': [102],  # >= all others
            'low': [98],    # <= all others
            'close': [101],
            'volume': [1000]
        }, index=pd.DatetimeIndex(['2023-01-01']))

        is_valid, errors = storage.validate_data_integrity(valid_data)
        assert is_valid

        # Invalid data
        invalid_data = pd.DataFrame({
            'open': [100],
            'high': [97],   # INVALID: less than low
            'low': [98],
            'close': [101],
            'volume': [1000]
        }, index=pd.DatetimeIndex(['2023-01-01']))

        is_valid, errors = storage.validate_data_integrity(invalid_data)
        assert not is_valid
```

#### `tests/test_data/test_source_comparison.py`

```python
import pytest
from datetime import datetime
from Backtester.data.provider import DataProvider

class TestSourceComparison:

    @pytest.mark.skipif(
        True,  # Skip until Schwab integration complete
        reason="Requires Schwab API implementation"
    )
    def test_yfinance_vs_schwab_prices(self):
        """Compare adjusted close prices between sources"""
        provider = DataProvider()

        comparison = provider.compare_sources(
            'AAPL',
            datetime(2023, 1, 1),
            datetime(2023, 12, 31)
        )

        # Most prices should be very close
        avg_diff_pct = comparison['diff_pct'].abs().mean()
        assert avg_diff_pct < 0.1  # Less than 0.1% average difference

        # Flag large divergences
        large_diffs = comparison[comparison['diff_pct'].abs() > 1.0]
        if len(large_diffs) > 0:
            print(f"Warning: {len(large_diffs)} days with >1% divergence")

    def test_source_fallback(self):
        """Verify fallback to secondary source on primary failure"""
        provider = DataProvider(
            primary_source='yfinance',
            fallback_source='yfinance'  # Use same for testing
        )

        # This should succeed even if primary fails
        # (In real scenario, primary and fallback would be different)
        data = provider.get_single_stock(
            'AAPL',
            datetime(2023, 1, 1),
            datetime(2023, 12, 31),
            source='auto'
        )

        assert data is not None
        assert len(data) > 0
```

### 1.3 Edge Cases to Handle

**Missing Data:**
- Weekends (expected, don't flag)
- Market holidays (expected, needs holiday calendar)
- Stock halts (unexpected gaps in trading days)
- Delisted stocks (historical data ends abruptly)
- IPO stocks (data starts mid-range)

**Corporate Actions:**
- Stock splits (verify adjusted prices are continuous)
- Dividends (verify ex-dividend drops are adjusted)
- Mergers/acquisitions (symbol may change or disappear)
- Spinoffs (parent stock adjustment)

**API Failures:**
- Rate limiting (429 errors)
- Network timeouts
- Invalid API responses
- API key expiration (Schwab)

**Data Quality Issues:**
- Flash crash artifacts (extreme intraday spikes)
- Bad ticks (single erroneous data point)
- Volume anomalies (wash trades, stock splits)
- Currency issues (ADRs may have FX conversion)

### 1.4 Phase 1 Success Criteria

✓ Can download 20 stocks from yfinance successfully
✓ Data cached in Parquet with metadata tracking
✓ All data quality tests pass (OHLC validation, spike detection)
✓ Cache hit/miss logic works correctly
✓ Universe manager maintains top 20 list with auto-refresh
✓ (Future) Source comparison shows <1% divergence between yfinance and Schwab

---

## Phase 2: Strategy Module

**Location:** `Backtester/strategies/`
**Goal:** Clean, testable signal generation framework
**Dependencies:** Phase 1 (requires data access)

### 2.1 Core Components

#### `base.py` - Abstract strategy interface

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict, Any
from Backtester.backtest.signals import SignalType

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""

    def __init__(self, **parameters):
        """
        Initialize strategy with parameters

        Args:
            **parameters: Strategy-specific parameters
                         Stored in self.parameters dict
        """
        self.parameters = parameters
        self.validate_parameters()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Transform OHLCV data into trading signals

        Args:
            data: DataFrame with columns [open, high, low, close, volume]
                  indexed by date (DatetimeIndex)

        Returns:
            Series of SignalType enum values (BUY=1, SELL=-1, HOLD=0)
            indexed by same dates as input data

        Implementation Requirements:
        - Must be VECTORIZED (process entire dataset at once)
        - Should not modify input data (read-only)
        - Must handle NaN values in indicator calculations
        - Should return same length as input (pad with HOLD if needed)
        """
        pass

    def validate_parameters(self):
        """
        Override to validate strategy-specific parameters

        Raises:
            ValueError if parameters are invalid
        """
        pass

    def generate_signals_debug(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Optional debug method returning signals + intermediate indicators

        Returns:
            (signals, indicators)
            - signals: pd.Series of SignalType
            - indicators: DataFrame with intermediate calculations

        Useful for strategy development and visualization
        """
        signals = self.generate_signals(data)
        return signals, pd.DataFrame()  # Default: no debug info

    def __str__(self):
        """String representation showing strategy name and parameters"""
        params_str = ', '.join(f'{k}={v}' for k, v in self.parameters.items())
        return f"{self.__class__.__name__}({params_str})"
```

#### `ma_crossover.py` - Trend-following strategy

```python
import pandas as pd
from Backtester.strategies.base import BaseStrategy
from Backtester.backtest.signals import SignalType

class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy (Trend Following)

    Signals:
    - BUY when short MA crosses above long MA (golden cross)
    - SELL when short MA crosses below long MA (death cross)
    - HOLD otherwise

    Parameters:
    - short_window: Period for short moving average (default: 50)
    - long_window: Period for long moving average (default: 200)

    Classic example: 50-day / 200-day MA crossover
    """

    def __init__(self, short_window: int = 50, long_window: int = 200):
        super().__init__(short_window=short_window, long_window=long_window)

    def validate_parameters(self):
        """Ensure short window < long window and both positive"""
        short = self.parameters['short_window']
        long = self.parameters['long_window']

        if short <= 0 or long <= 0:
            raise ValueError("Window sizes must be positive")

        if short >= long:
            raise ValueError(
                f"Short window ({short}) must be less than long window ({long})"
            )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals using MA crossover logic

        Implementation: Vectorized using pandas
        """
        # Calculate moving averages
        short_ma = data['close'].rolling(
            window=self.parameters['short_window']
        ).mean()

        long_ma = data['close'].rolling(
            window=self.parameters['long_window']
        ).mean()

        # Initialize all signals as HOLD
        signals = pd.Series(SignalType.HOLD, index=data.index)

        # Detect crossovers using vectorized comparison
        short_above = short_ma > long_ma
        short_above_prev = short_above.shift(1)

        # Golden cross: short crosses above long (BUY signal)
        golden_cross = short_above & ~short_above_prev
        signals[golden_cross] = SignalType.BUY

        # Death cross: short crosses below long (SELL signal)
        death_cross = ~short_above & short_above_prev
        signals[death_cross] = SignalType.SELL

        return signals

    def generate_signals_debug(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Return signals plus MA values for visualization"""
        signals = self.generate_signals(data)

        indicators = pd.DataFrame(index=data.index)
        indicators['short_ma'] = data['close'].rolling(
            self.parameters['short_window']
        ).mean()
        indicators['long_ma'] = data['close'].rolling(
            self.parameters['long_window']
        ).mean()
        indicators['signal'] = signals
        indicators['close'] = data['close']

        return signals, indicators
```

**Strategy Rationale:**
- Golden cross (50 above 200) = bullish momentum → enter long
- Death cross (50 below 200) = bearish momentum → exit position
- Simple, well-known, easy to validate
- Works well in trending markets, whipsaws in sideways markets

#### `rsi_strategy.py` - Mean-reversion strategy

```python
import pandas as pd
import numpy as np
from Backtester.strategies.base import BaseStrategy
from Backtester.backtest.signals import SignalType

class RSIMeanReversion(BaseStrategy):
    """
    RSI Mean Reversion Strategy

    Signals:
    - BUY when RSI < oversold threshold (e.g., 30)
    - SELL when RSI > overbought threshold (e.g., 70)
    - HOLD otherwise

    Parameters:
    - rsi_period: RSI calculation period (default: 14)
    - oversold: RSI level considered oversold (default: 30)
    - overbought: RSI level considered overbought (default: 70)

    Theory: RSI < 30 indicates oversold (likely to bounce)
            RSI > 70 indicates overbought (likely to pullback)
    """

    def __init__(self, rsi_period: int = 14,
                 oversold: float = 30,
                 overbought: float = 70):
        super().__init__(
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought
        )

    def validate_parameters(self):
        """Validate RSI parameters"""
        rsi_period = self.parameters['rsi_period']
        oversold = self.parameters['oversold']
        overbought = self.parameters['overbought']

        if rsi_period <= 0:
            raise ValueError("RSI period must be positive")

        if not (0 <= oversold <= 100):
            raise ValueError("Oversold threshold must be between 0 and 100")

        if not (0 <= overbought <= 100):
            raise ValueError("Overbought threshold must be between 0 and 100")

        if oversold >= overbought:
            raise ValueError(
                f"Oversold ({oversold}) must be less than overbought ({overbought})"
            )

    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI using vectorized pandas operations

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss over period

        Args:
            prices: Close price series
            period: RSI period (typically 14)

        Returns:
            RSI values (0-100)
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate rolling averages
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on RSI levels

        Implementation: Vectorized using pandas
        """
        # Calculate RSI
        rsi = self.calculate_rsi(
            data['close'],
            self.parameters['rsi_period']
        )

        # Initialize all signals as HOLD
        signals = pd.Series(SignalType.HOLD, index=data.index)

        # BUY when oversold
        oversold_condition = rsi < self.parameters['oversold']
        signals[oversold_condition] = SignalType.BUY

        # SELL when overbought
        overbought_condition = rsi > self.parameters['overbought']
        signals[overbought_condition] = SignalType.SELL

        return signals

    def generate_signals_debug(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Return signals plus RSI values for visualization"""
        rsi = self.calculate_rsi(
            data['close'],
            self.parameters['rsi_period']
        )

        signals = self.generate_signals(data)

        indicators = pd.DataFrame(index=data.index)
        indicators['rsi'] = rsi
        indicators['signal'] = signals
        indicators['close'] = data['close']
        indicators['oversold_line'] = self.parameters['oversold']
        indicators['overbought_line'] = self.parameters['overbought']

        return signals, indicators
```

**Strategy Rationale:**
- RSI < 30 = stock oversold, likely to bounce (contrarian buy)
- RSI > 70 = stock overbought, likely to pullback (contrarian sell)
- Mean-reversion philosophy (opposite of trend-following)
- Works well in range-bound markets, fails in strong trends

### 2.2 Testing Strategy

#### `tests/test_strategies/test_base_strategy.py`

```python
import pytest
import pandas as pd
from Backtester.strategies.base import BaseStrategy
from Backtester.backtest.signals import SignalType

class DummyStrategy(BaseStrategy):
    """Concrete implementation for testing abstract base"""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Simple logic: BUY if close > 100, SELL otherwise
        signals = pd.Series(SignalType.HOLD, index=data.index)
        signals[data['close'] > 100] = SignalType.BUY
        signals[data['close'] <= 100] = SignalType.SELL
        return signals

class TestBaseStrategy:

    def test_cannot_instantiate_abstract_class(self):
        """BaseStrategy should not be instantiable directly"""
        with pytest.raises(TypeError):
            BaseStrategy()

    def test_parameter_storage(self):
        """Parameters should be stored in self.parameters dict"""
        strategy = DummyStrategy(param1=10, param2='test')
        assert strategy.parameters['param1'] == 10
        assert strategy.parameters['param2'] == 'test'

    def test_generate_signals_returns_series(self):
        """generate_signals must return pd.Series"""
        strategy = DummyStrategy()
        data = pd.DataFrame({
            'close': [99, 101, 100],
            'open': [98, 100, 99],
            'high': [100, 102, 101],
            'low': [98, 100, 99],
            'volume': [1000, 1100, 1000]
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03']))

        signals = strategy.generate_signals(data)
        assert isinstance(signals, pd.Series)

    def test_signal_values_are_valid(self):
        """Signals should only contain valid SignalType values"""
        strategy = DummyStrategy()
        data = pd.DataFrame({
            'close': [99, 101, 100],
            'open': [98, 100, 99],
            'high': [100, 102, 101],
            'low': [98, 100, 99],
            'volume': [1000, 1100, 1000]
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03']))

        signals = strategy.generate_signals(data)

        # All values should be valid SignalType
        valid_values = {SignalType.BUY, SignalType.SELL, SignalType.HOLD}
        assert all(s in valid_values for s in signals)

    def test_string_representation(self):
        """__str__ should show strategy name and parameters"""
        strategy = DummyStrategy(window=50, threshold=0.5)
        str_repr = str(strategy)

        assert 'DummyStrategy' in str_repr
        assert 'window=50' in str_repr
        assert 'threshold=0.5' in str_repr
```

#### `tests/test_strategies/test_ma_crossover.py`

```python
import pytest
import pandas as pd
import numpy as np
from Backtester.strategies.ma_crossover import MovingAverageCrossover
from Backtester.backtest.signals import SignalType

class TestMovingAverageCrossover:

    def test_parameter_validation(self):
        """Short window must be less than long window"""
        # Valid parameters
        strategy = MovingAverageCrossover(short_window=50, long_window=200)
        assert strategy.parameters['short_window'] == 50

        # Invalid: short >= long
        with pytest.raises(ValueError):
            MovingAverageCrossover(short_window=200, long_window=50)

        # Invalid: negative window
        with pytest.raises(ValueError):
            MovingAverageCrossover(short_window=-10, long_window=200)

    def test_golden_cross_generates_buy(self):
        """Short MA crossing above long MA should generate BUY"""
        # Create synthetic data with clear golden cross
        # Prices fall then rise to create crossover

        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # First 50 days: falling prices (short MA will be below long MA)
        # Last 50 days: rising prices (short MA will cross above)
        prices = np.concatenate([
            np.linspace(100, 80, 50),  # Falling
            np.linspace(80, 120, 50)   # Rising
        ])

        data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': [1000] * 100
        }, index=dates)

        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        signals = strategy.generate_signals(data)

        # Should have at least one BUY signal
        assert (signals == SignalType.BUY).any()

    def test_death_cross_generates_sell(self):
        """Short MA crossing below long MA should generate SELL"""
        # Create synthetic data with death cross

        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # First 50 days: rising (short above long)
        # Last 50 days: falling (short crosses below)
        prices = np.concatenate([
            np.linspace(80, 120, 50),  # Rising
            np.linspace(120, 80, 50)   # Falling
        ])

        data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': [1000] * 100
        }, index=dates)

        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        signals = strategy.generate_signals(data)

        # Should have at least one SELL signal
        assert (signals == SignalType.SELL).any()

    def test_parallel_mas_generate_hold(self):
        """When MAs move parallel, should generate HOLD"""
        # Flat prices = parallel MAs
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = [100] * 100

        data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices,
            'low': prices,
            'volume': [1000] * 100
        }, index=dates)

        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        signals = strategy.generate_signals(data)

        # After MAs converge, most signals should be HOLD
        # (first 20 days have NaN, so check after that)
        assert (signals[20:] == SignalType.HOLD).most()

    def test_insufficient_data_handling(self):
        """Strategy should handle data shorter than window"""
        # Only 10 days of data, but long_window=200
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'close': range(100, 110),
            'open': range(100, 110),
            'high': range(101, 111),
            'low': range(99, 109),
            'volume': [1000] * 10
        }, index=dates)

        strategy = MovingAverageCrossover(short_window=50, long_window=200)
        signals = strategy.generate_signals(data)

        # Should return HOLD for all (MAs will be NaN)
        assert len(signals) == 10
        assert all(s == SignalType.HOLD or pd.isna(s) for s in signals)

    def test_debug_mode_returns_indicators(self):
        """Debug mode should return MA values"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': range(100, 200),
            'open': range(100, 200),
            'high': range(101, 201),
            'low': range(99, 199),
            'volume': [1000] * 100
        }, index=dates)

        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        signals, indicators = strategy.generate_signals_debug(data)

        assert 'short_ma' in indicators.columns
        assert 'long_ma' in indicators.columns
        assert len(signals) == len(indicators)

    @pytest.mark.integration
    def test_known_backtest_outcome(self):
        """Test against stored fixture with known outcome"""
        # Load fixture data
        # (This assumes fixture files created in Phase 1)
        fixture_path = 'tests/fixtures/aapl_2020_2024.parquet'

        try:
            data = pd.read_parquet(fixture_path)
        except FileNotFoundError:
            pytest.skip("Fixture file not available")

        # Load expected signals
        expected_signals_path = 'tests/fixtures/ma_50_200_signals_expected.csv'
        try:
            expected = pd.read_csv(expected_signals_path, index_col=0, parse_dates=True)
        except FileNotFoundError:
            pytest.skip("Expected signals fixture not available")

        # Generate signals
        strategy = MovingAverageCrossover(short_window=50, long_window=200)
        signals = strategy.generate_signals(data)

        # Compare
        pd.testing.assert_series_equal(
            signals,
            expected['signal'],
            check_names=False
        )
```

#### `tests/test_strategies/test_rsi_strategy.py`

```python
import pytest
import pandas as pd
import numpy as np
from Backtester.strategies.rsi_strategy import RSIMeanReversion
from Backtester.backtest.signals import SignalType

class TestRSIMeanReversion:

    def test_parameter_validation(self):
        """Validate RSI parameter constraints"""
        # Valid parameters
        strategy = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        assert strategy.parameters['rsi_period'] == 14

        # Invalid: oversold >= overbought
        with pytest.raises(ValueError):
            RSIMeanReversion(oversold=70, overbought=30)

        # Invalid: out of range
        with pytest.raises(ValueError):
            RSIMeanReversion(oversold=150)

    def test_rsi_calculation_accuracy(self):
        """RSI calculation should match known values"""
        # Use known price series with verified RSI values
        # (Can compare against ta-lib or tradingview)

        prices = pd.Series([
            44, 44.34, 44.09, 43.61, 44.33,
            44.83, 45.10, 45.42, 45.84, 46.08,
            45.89, 46.03, 45.61, 46.28, 46.28,
            46.00, 46.03, 46.41, 46.22, 45.64
        ])

        strategy = RSIMeanReversion(rsi_period=14)
        rsi = strategy.calculate_rsi(prices, period=14)

        # After 14 periods, RSI should be calculable
        assert not rsi.iloc[14:].isnull().all()

        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_oversold_generates_buy(self):
        """RSI < oversold threshold should generate BUY"""
        # Create price series that trends down (RSI will drop)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = np.linspace(100, 50, 50)  # Steady decline

        data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': [1000] * 50
        }, index=dates)

        strategy = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        signals = strategy.generate_signals(data)

        # Should have BUY signals when RSI drops below 30
        assert (signals == SignalType.BUY).any()

    def test_overbought_generates_sell(self):
        """RSI > overbought threshold should generate SELL"""
        # Create price series that trends up (RSI will rise)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = np.linspace(50, 100, 50)  # Steady rise

        data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': [1000] * 50
        }, index=dates)

        strategy = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        signals = strategy.generate_signals(data)

        # Should have SELL signals when RSI rises above 70
        assert (signals == SignalType.SELL).any()

    def test_rsi_boundary_conditions(self):
        """Test RSI exactly at thresholds"""
        # This is tricky - need to engineer prices to hit exact RSI values
        # For now, just test that signals work near boundaries

        dates = pd.date_range('2023-01-01', periods=30, freq='D')

        # Oscillating prices
        prices = [100 + 5 * np.sin(i * 0.3) for i in range(30)]

        data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'volume': [1000] * 30
        }, index=dates)

        strategy = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        signals = strategy.generate_signals(data)

        # No errors should occur
        assert len(signals) == 30

    def test_debug_mode_returns_rsi(self):
        """Debug mode should return RSI values"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = np.random.rand(50) * 100

        data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': [1000] * 50
        }, index=dates)

        strategy = RSIMeanReversion(rsi_period=14)
        signals, indicators = strategy.generate_signals_debug(data)

        assert 'rsi' in indicators.columns
        assert 'oversold_line' in indicators.columns
        assert 'overbought_line' in indicators.columns
```

### 2.3 Edge Cases

**Insufficient Data:**
- Data shorter than indicator period (e.g., 10 days but 50-day MA)
- Solution: Return HOLD for NaN periods, or raise clear error

**NaN Handling:**
- First N bars have NaN indicators
- Division by zero in RSI (no price changes)
- Solution: Use `.fillna(SignalType.HOLD)`

**Flat Prices:**
- All prices identical → RSI undefined (0/0)
- MAs converge → no crossovers
- Solution: Detect and return HOLD

**Extreme Volatility:**
- RSI pegged at 0 or 100 for extended periods
- Flash crashes create false signals
- Solution: Add signal confirmation logic (future enhancement)

### 2.4 Phase 2 Success Criteria

✓ MA crossover strategy reproduces known signals from fixture
✓ RSI calculation matches ta-lib or verified source
✓ All strategies pass parameter validation
✓ Debug mode returns intermediate indicators
✓ Strategies handle edge cases (insufficient data, NaN, flat prices)
✓ 100% test coverage on strategy logic

---

## Phase 3: Backtest Module

**Location:** `Backtester/backtest/`
**Goal:** Accurate simulation with realistic constraints
**Dependencies:** Phase 1 (data), Phase 2 (strategies)

### 3.1 Core Components

#### `signals.py` - Data structures

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class SignalType(Enum):
    """Trading signal types"""
    BUY = 1      # Enter long position
    SELL = -1    # Exit position
    HOLD = 0     # No action

@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    current_price: float

    @property
    def unrealized_pnl(self) -> float:
        """Profit/loss if closed at current price"""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Percentage return"""
        return (self.current_price / self.entry_price - 1) * 100

    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.current_price * self.quantity

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float           # Absolute profit/loss
    pnl_pct: float       # Percentage return
    commission_paid: float

    @property
    def hold_days(self) -> int:
        """Number of days position was held"""
        return (self.exit_date - self.entry_date).days

    @property
    def is_winner(self) -> bool:
        """Was this a profitable trade?"""
        return self.pnl > 0
```

#### `portfolio.py` - Position and cash management

```python
from typing import Dict, List, Optional
from datetime import datetime
import logging

from Backtester.backtest.signals import SignalType, Position, Trade

class Portfolio:
    """
    Manages cash and positions during backtest

    Responsibilities:
    - Track cash balance
    - Execute buy/sell orders
    - Calculate position sizes
    - Apply commissions
    - Record trade history
    """

    def __init__(self, initial_capital: float, commission_pct: float = 0.001):
        """
        Initialize portfolio

        Args:
            initial_capital: Starting cash amount
            commission_pct: Commission as percentage of trade value
                           Default 0.001 = 0.1% per trade
        """
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.commission_pct = commission_pct
        self.logger = logging.getLogger(__name__)

    def execute_signal(self, signal: SignalType, symbol: str,
                      price: float, date: datetime,
                      portfolio_value: float) -> None:
        """
        Execute trading signal

        Args:
            signal: BUY, SELL, or HOLD
            symbol: Stock ticker
            price: Execution price (typically close)
            date: Trade date
            portfolio_value: Total portfolio value (for position sizing)
        """
        if signal == SignalType.BUY:
            self._execute_buy(symbol, price, date, portfolio_value)
        elif signal == SignalType.SELL:
            self._execute_sell(symbol, price, date)
        # HOLD: do nothing

    def _execute_buy(self, symbol: str, price: float,
                    date: datetime, portfolio_value: float) -> None:
        """
        Execute buy order with position sizing and commission

        Position Sizing: 5% of total portfolio value
        """
        # Don't buy if already have position
        if symbol in self.positions:
            self.logger.debug(f"Already hold {symbol}, skipping buy signal")
            return

        # Calculate position size (5% of portfolio)
        allocation = portfolio_value * 0.05
        quantity = int(allocation / price)

        if quantity == 0:
            self.logger.debug(f"Position size too small for {symbol}")
            return

        # Calculate cost including commission
        cost_before_commission = quantity * price
        commission = cost_before_commission * self.commission_pct
        total_cost = cost_before_commission + commission

        # Check if sufficient cash
        if total_cost > self.cash:
            self.logger.debug(
                f"Insufficient cash for {symbol}: need ${total_cost:.2f}, have ${self.cash:.2f}"
            )
            return

        # Execute trade
        self.cash -= total_cost
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_date=date,
            current_price=price
        )

        self.logger.info(
            f"BUY {quantity} {symbol} @ ${price:.2f} (commission: ${commission:.2f})"
        )

    def _execute_sell(self, symbol: str, price: float, date: datetime) -> None:
        """Execute sell order and record trade"""
        # Can't sell if no position
        if symbol not in self.positions:
            self.logger.debug(f"No position in {symbol}, skipping sell signal")
            return

        pos = self.positions[symbol]

        # Calculate proceeds after commission
        proceeds_before_commission = pos.quantity * price
        commission = proceeds_before_commission * self.commission_pct
        proceeds = proceeds_before_commission - commission

        # Add to cash
        self.cash += proceeds

        # Calculate P&L
        cost_basis = pos.quantity * pos.entry_price
        total_commission = (cost_basis * self.commission_pct) + commission
        pnl = proceeds_before_commission - cost_basis - total_commission
        pnl_pct = (price / pos.entry_price - 1) * 100

        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission_paid=total_commission
        )
        self.trades.append(trade)

        # Remove position
        del self.positions[symbol]

        self.logger.info(
            f"SELL {pos.quantity} {symbol} @ ${price:.2f} "
            f"(P&L: ${pnl:.2f}, {pnl_pct:.1f}%)"
        )

    def update_positions(self, prices: Dict[str, float]) -> None:
        """Update current prices for all positions"""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]

    def get_total_value(self) -> float:
        """Calculate total portfolio value (cash + positions)"""
        positions_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        return self.cash + positions_value

    def close_all_positions(self, prices: Dict[str, float],
                           date: datetime) -> None:
        """Force close all positions (end of backtest)"""
        symbols_to_close = list(self.positions.keys())
        for symbol in symbols_to_close:
            if symbol in prices:
                self._execute_sell(symbol, prices[symbol], date)

    def get_trade_history(self) -> List[Trade]:
        """Get all completed trades"""
        return self.trades
```

#### `metrics.py` - Performance calculations

```python
import pandas as pd
import numpy as np
from typing import List, Dict
from Backtester.backtest.signals import Trade

class PerformanceMetrics:
    """Calculate backtest performance metrics"""

    @staticmethod
    def calculate_total_return(equity_curve: pd.Series,
                              initial_capital: float) -> float:
        """Total return as percentage"""
        final_value = equity_curve.iloc[-1]
        return (final_value / initial_capital - 1) * 100

    @staticmethod
    def calculate_annualized_return(equity_curve: pd.Series) -> float:
        """Annualized return (CAGR)"""
        total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        years = total_days / 365.25

        if years == 0:
            return 0.0

        return (total_return ** (1 / years) - 1) * 100

    @staticmethod
    def calculate_sharpe_ratio(equity_curve: pd.Series,
                              risk_free_rate: float = 0.0) -> float:
        """
        Sharpe ratio (annualized)

        = (Mean Return - Risk Free Rate) / Std Dev of Returns
        Annualized by multiplying by sqrt(252) for daily data
        """
        returns = equity_curve.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualize risk-free rate to daily
        daily_rf = risk_free_rate / 252

        excess_returns = returns - daily_rf
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()

        return sharpe

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Maximum drawdown as percentage

        Max peak-to-trough decline
        """
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax * 100
        return drawdown.min()

    @staticmethod
    def calculate_win_rate(trades: List[Trade]) -> float:
        """Percentage of profitable trades"""
        if not trades:
            return 0.0

        winning_trades = sum(1 for t in trades if t.pnl > 0)
        return (winning_trades / len(trades)) * 100

    @staticmethod
    def calculate_profit_factor(trades: List[Trade]) -> float:
        """
        Profit factor = Gross Profit / Gross Loss

        > 1.0 = profitable system
        < 1.0 = losing system
        """
        if not trades:
            return 0.0

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def calculate_average_trade(trades: List[Trade]) -> float:
        """Average P&L per trade"""
        if not trades:
            return 0.0
        return np.mean([t.pnl for t in trades])

    @staticmethod
    def calculate_average_winner(trades: List[Trade]) -> float:
        """Average winning trade P&L"""
        winners = [t.pnl for t in trades if t.pnl > 0]
        return np.mean(winners) if winners else 0.0

    @staticmethod
    def calculate_average_loser(trades: List[Trade]) -> float:
        """Average losing trade P&L"""
        losers = [t.pnl for t in trades if t.pnl < 0]
        return np.mean(losers) if losers else 0.0

    @staticmethod
    def calculate_all_metrics(equity_curve: pd.Series,
                             trades: List[Trade],
                             initial_capital: float,
                             risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate comprehensive set of metrics

        Returns:
            Dict with all performance metrics
        """
        return {
            'total_return': PerformanceMetrics.calculate_total_return(
                equity_curve, initial_capital
            ),
            'annualized_return': PerformanceMetrics.calculate_annualized_return(
                equity_curve
            ),
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(
                equity_curve, risk_free_rate
            ),
            'max_drawdown': PerformanceMetrics.calculate_max_drawdown(
                equity_curve
            ),
            'num_trades': len(trades),
            'win_rate': PerformanceMetrics.calculate_win_rate(trades),
            'profit_factor': PerformanceMetrics.calculate_profit_factor(trades),
            'avg_trade_pnl': PerformanceMetrics.calculate_average_trade(trades),
            'avg_win': PerformanceMetrics.calculate_average_winner(trades),
            'avg_loss': PerformanceMetrics.calculate_average_loser(trades),
        }
```

#### `engine.py` - Backtest orchestrator

```python
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from datetime import datetime
import logging

from Backtester.strategies.base import BaseStrategy
from Backtester.backtest.portfolio import Portfolio
from Backtester.backtest.metrics import PerformanceMetrics
from Backtester.backtest.signals import Trade

@dataclass
class BacktestResult:
    """Container for backtest results"""
    equity_curve: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]
    signals: Dict[str, pd.Series]  # symbol -> signal series

class BacktestEngine:
    """
    Main backtesting engine

    Orchestrates:
    1. Signal generation (vectorized)
    2. Chronological simulation
    3. Portfolio management
    4. Performance calculation
    """

    def __init__(self, initial_capital: float = 100000,
                 commission_pct: float = 0.001):
        """
        Initialize backtest engine

        Args:
            initial_capital: Starting capital (default $100k)
            commission_pct: Commission rate (default 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.logger = logging.getLogger(__name__)

    def run(self, strategy: BaseStrategy,
            data: Dict[str, pd.DataFrame],
            risk_free_rate: float = 0.0) -> BacktestResult:
        """
        Run backtest

        Args:
            strategy: Strategy instance
            data: Dict mapping symbol -> OHLCV DataFrame
            risk_free_rate: Annual risk-free rate for Sharpe calculation

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        self.logger.info(f"Starting backtest with {strategy}")

        # Step 1: Generate all signals (vectorized, fast)
        self.logger.info("Generating signals...")
        signals = {
            symbol: strategy.generate_signals(df)
            for symbol, df in data.items()
        }

        # Step 2: Get common dates across all symbols
        common_dates = self._get_common_dates(data)
        self.logger.info(f"Backtest period: {common_dates[0]} to {common_dates[-1]}")
        self.logger.info(f"Number of trading days: {len(common_dates)}")

        # Step 3: Initialize portfolio
        portfolio = Portfolio(self.initial_capital, self.commission_pct)
        equity_curve = []

        # Step 4: Chronological simulation
        self.logger.info("Running simulation...")
        for i, date in enumerate(common_dates):
            # Get current prices (use close)
            prices = {
                sym: data[sym].loc[date, 'close']
                for sym in data.keys()
            }

            # Update position valuations with current prices
            portfolio.update_positions(prices)

            # Get portfolio value BEFORE executing new signals
            # (Important: position sizing based on current value)
            portfolio_value = portfolio.get_total_value()

            # Execute signals for this date
            for symbol in data.keys():
                signal = signals[symbol].loc[date]
                portfolio.execute_signal(
                    signal, symbol, prices[symbol],
                    date, portfolio_value
                )

            # Record equity after signal execution
            equity_curve.append({
                'date': date,
                'value': portfolio.get_total_value()
            })

            # Progress logging every 50 days
            if (i + 1) % 50 == 0:
                self.logger.debug(
                    f"Day {i+1}/{len(common_dates)}: "
                    f"Value=${portfolio.get_total_value():.2f}"
                )

        # Step 5: Close all positions at end
        final_date = common_dates[-1]
        final_prices = {
            sym: data[sym].loc[final_date, 'close']
            for sym in data.keys()
        }
        portfolio.close_all_positions(final_prices, final_date)

        # Step 6: Convert equity curve to Series
        equity_series = pd.Series(
            [e['value'] for e in equity_curve],
            index=[e['date'] for e in equity_curve]
        )

        # Step 7: Calculate metrics
        self.logger.info("Calculating performance metrics...")
        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_series,
            portfolio.trades,
            self.initial_capital,
            risk_free_rate
        )

        # Log summary
        self.logger.info(f"Backtest complete:")
        self.logger.info(f"  Total Return: {metrics['total_return']:.2f}%")
        self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        self.logger.info(f"  Num Trades: {metrics['num_trades']}")

        return BacktestResult(
            equity_curve=equity_series,
            trades=portfolio.trades,
            metrics=metrics,
            signals=signals
        )

    def _get_common_dates(self, data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """
        Find intersection of all date ranges

        Only backtest on dates where all symbols have data
        """
        date_sets = [set(df.index) for df in data.values()]
        common = set.intersection(*date_sets)
        return pd.DatetimeIndex(sorted(common))
```

### 3.2 Storage Format Decision

**Hybrid Approach: JSON + Parquet**

```
results/
  backtest_20260106_143022/
    metadata.json          # Strategy, params, metrics, run info
    equity_curve.parquet   # Time series data (fast queries)
    trades.parquet         # Trade history (tabular data)
    signals/               # Per-symbol signals
      AAPL.parquet
      MSFT.parquet
      ...
```

**Rationale:**
- **Parquet** for time-series (equity curve, signals) → columnar, fast
- **JSON** for metadata → human-readable, version-controllable
- **Hybrid** for trades → can use either (Parquet slightly better for large datasets)

**Pros:**
- Fast access to equity curve for plotting
- Human-readable config and metrics
- Easy to version control metadata
- Efficient storage for large backtests

**Cons:**
- Slightly more complex than single-file approach
- Need to manage directory structure

### 3.3 Testing Strategy

#### `tests/test_backtest/test_portfolio.py`

```python
import pytest
from datetime import datetime
from Backtester.backtest.portfolio import Portfolio
from Backtester.backtest.signals import SignalType

class TestPortfolio:

    def test_initial_state(self):
        """Portfolio should start with cash, no positions"""
        portfolio = Portfolio(initial_capital=100000)
        assert portfolio.cash == 100000
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0

    def test_buy_execution(self):
        """Buy should decrease cash and create position"""
        portfolio = Portfolio(initial_capital=100000)

        portfolio.execute_signal(
            SignalType.BUY,
            'AAPL',
            100.0,  # price
            datetime(2023, 1, 1),
            100000  # portfolio value
        )

        # Should have position
        assert 'AAPL' in portfolio.positions
        pos = portfolio.positions['AAPL']

        # Quantity should be ~5% of portfolio / price
        # 5% of 100k = 5000, at $100 = 50 shares
        assert pos.quantity == 50

        # Cash should decrease by cost + commission
        cost = 50 * 100 * 1.001  # Include 0.1% commission
        assert portfolio.cash == pytest.approx(100000 - cost, abs=0.01)

    def test_insufficient_cash(self):
        """Cannot buy if not enough cash"""
        portfolio = Portfolio(initial_capital=1000)  # Only $1k

        portfolio.execute_signal(
            SignalType.BUY,
            'AAPL',
            1000.0,  # Very expensive stock
            datetime(2023, 1, 1),
            1000
        )

        # Should not create position
        assert 'AAPL' not in portfolio.positions
        assert portfolio.cash == 1000  # Unchanged

    def test_commission_calculation(self):
        """Commission should be applied to both sides"""
        portfolio = Portfolio(initial_capital=100000, commission_pct=0.01)  # 1%

        # Buy
        portfolio.execute_signal(
            SignalType.BUY,
            'AAPL',
            100.0,
            datetime(2023, 1, 1),
            100000
        )

        cash_after_buy = portfolio.cash

        # Sell
        portfolio.execute_signal(
            SignalType.SELL,
            'AAPL',
            100.0,  # Same price (no profit before commission)
            datetime(2023, 1, 2),
            100000
        )

        # Should have lost money due to commission (round-trip)
        assert portfolio.cash < 100000

    def test_sell_execution(self):
        """Sell should close position and record trade"""
        portfolio = Portfolio(initial_capital=100000)

        # Buy
        portfolio.execute_signal(
            SignalType.BUY,
            'AAPL',
            100.0,
            datetime(2023, 1, 1),
            100000
        )

        # Sell at profit
        portfolio.execute_signal(
            SignalType.SELL,
            'AAPL',
            110.0,  # 10% gain
            datetime(2023, 1, 10),
            100000
        )

        # Position should be closed
        assert 'AAPL' not in portfolio.positions

        # Should have 1 trade
        assert len(portfolio.trades) == 1
        trade = portfolio.trades[0]

        assert trade.symbol == 'AAPL'
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.pnl > 0  # Profitable

    def test_sell_nonexistent_position(self):
        """Selling without position should be no-op"""
        portfolio = Portfolio(initial_capital=100000)

        portfolio.execute_signal(
            SignalType.SELL,
            'AAPL',
            100.0,
            datetime(2023, 1, 1),
            100000
        )

        # No trade created
        assert len(portfolio.trades) == 0

    def test_position_update(self):
        """update_positions should change current_price"""
        portfolio = Portfolio(initial_capital=100000)

        portfolio.execute_signal(
            SignalType.BUY,
            'AAPL',
            100.0,
            datetime(2023, 1, 1),
            100000
        )

        # Update price
        portfolio.update_positions({'AAPL': 105.0})

        pos = portfolio.positions['AAPL']
        assert pos.current_price == 105.0
        assert pos.unrealized_pnl > 0
```

(Continue with test_engine.py and test_metrics.py following similar patterns...)

### 3.4 Phase 3 Success Criteria

✓ Can run multi-stock backtest end-to-end
✓ Known backtest reproduces within 0.01% tolerance
✓ All metrics calculated correctly (verified against manual calculations)
✓ Results saved in hybrid Parquet/JSON format
✓ Portfolio properly handles edge cases (insufficient cash, double buys)
✓ Commission accounting is accurate

---

## Phase 4: Visualization Module

**Location:** `Backtester/visualization/`
**Goal:** Intuitive, interactive reports
**Dependencies:** Phase 3 (requires backtest results)

### 4.1 Core Components

[Content from earlier, formatted as above with charts.py and reports.py implementation]

### 4.2 Phase 4 Success Criteria

✓ HTML report opens in browser with all charts
✓ Charts are interactive (zoom, pan, hover)
✓ Can compare 2+ strategies side-by-side
✓ Report generation takes <5 seconds
✓ All metrics displayed in readable format

---

## Phase 5: Utils Module

**Location:** `Backtester/utils.py`
**Goal:** Shared utilities to avoid duplication
**Dependencies:** All phases (extracted from common patterns)

### 5.1 Candidate Functions

[Date utilities, validation utilities, file utilities as described earlier]

### 5.2 Phase 5 Success Criteria

✓ All duplicate code removed from modules
✓ Utils functions have 100% test coverage
✓ Clear documentation for each utility

---

## Implementation Order & Dependencies

```
Phase 1: Data Module (Foundational - no dependencies)
   ↓ Required by Strategy testing
Phase 2: Strategy Module (Independent once data available)
   ↓ Required by Backtest
Phase 3: Backtest Module (Depends on Data + Strategy)
   ↓ Produces results for visualization
Phase 4: Visualization Module (Consumes backtest results)

Phase 5: Utils (Extracted as patterns emerge across phases)
```

**Parallel Work Opportunities:**
- Phase 4 (Visualization) can start before Phase 3 is 100% complete
- Use mock/fixture backtest results to develop charts
- Phase 5 (Utils) can be extracted incrementally during other phases

---

## Success Criteria by Phase

### Phase 1 Complete:
✓ Download 20 stocks from yfinance successfully
✓ Data cached in Parquet with metadata tracking
✓ All data quality tests pass
✓ Cache hit/miss logic works correctly
✓ Universe manager maintains top 20 with auto-refresh
✓ Source comparison infrastructure ready (Schwab integration pending)

### Phase 2 Complete:
✓ MA crossover reproduces known signals
✓ RSI matches verified calculations
✓ All strategies pass vectorization tests
✓ Debug mode returns intermediate indicators
✓ Edge cases handled gracefully

### Phase 3 Complete:
✓ Multi-stock backtest runs end-to-end
✓ Known backtest reproduces exactly
✓ All metrics correct
✓ Results saved properly
✓ 100% test coverage on portfolio logic

### Phase 4 Complete:
✓ HTML report with interactive charts
✓ Strategy comparison working
✓ Fast report generation (<5s)
✓ Professional presentation quality

### Phase 5 Complete:
✓ No code duplication
✓ All utils tested
✓ Clear documentation

---

## Testing Philosophy

### Fixture-Based Testing
- Store known-good data in `tests/fixtures/`
- Version control expected outputs
- Test for exact reproduction (deterministic backtests)

**Example Fixtures:**
```
tests/fixtures/
  aapl_2020_2024.parquet              # Historical data
  ma_50_200_signals_expected.csv       # Expected signals
  ma_crossover_backtest_expected.json  # Expected metrics
  rsi_14_30_70_signals_expected.csv
```

### Property-Based Testing
- Use `hypothesis` for edge case generation
- Test invariants:
  - Total equity = cash + sum(position values)
  - Total P&L = sum of all trades
  - Portfolio value always >= 0
  - Commission always > 0

### Integration Tests
- End-to-end workflows in `tests/integration/`
- Test full pipeline: download → signal → backtest → report
- Use small date ranges for speed

### Performance Tests
- Benchmark critical paths
- Targets:
  - Signal generation: <1s for 5 years daily data
  - Backtest simulation: <5s for 20 stocks × 5 years
  - Report generation: <5s

**Benchmark Script:**
```python
import time

def benchmark_signal_generation():
    # 5 years, 1250 trading days
    start = time.time()
    strategy.generate_signals(data)
    elapsed = time.time() - start

    assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"
```

### Test Organization
```
tests/
  fixtures/                  # Known-good data
    aapl_2020_2024.parquet
    ma_50_200_signals.csv
    backtest_expected.json

  test_data/                 # Phase 1
    test_downloader.py
    test_storage.py
    test_universe.py
    test_provider.py
    test_data_quality.py
    test_source_comparison.py

  test_strategies/           # Phase 2
    test_base_strategy.py
    test_ma_crossover.py
    test_rsi_strategy.py

  test_backtest/             # Phase 3
    test_signals.py
    test_portfolio.py
    test_engine.py
    test_metrics.py

  test_visualization/        # Phase 4
    test_charts.py
    test_reports.py

  test_utils.py              # Phase 5

  integration/               # End-to-end
    test_full_pipeline.py
    test_multi_strategy.py

  performance/               # Benchmarks
    benchmark_vectorization.py
    benchmark_backtest.py
```

---

## Risk Mitigation

### Data Risks
- **API failures**: Fallback to secondary source, retry logic
- **Data quality**: Comprehensive validation before caching
- **Corporate actions**: Test split/dividend adjustment carefully

### Strategy Risks
- **Lookahead bias**: Pre-generate all signals before simulation
- **Overfitting**: Use out-of-sample testing (not in initial phases)
- **NaN handling**: Explicit tests for edge cases

### Backtest Risks
- **Unrealistic execution**: Use market-on-close, apply commission
- **Position sizing errors**: Comprehensive portfolio tests
- **Rounding errors**: Use appropriate precision, validate totals

### Performance Risks
- **Slow vectorization**: Benchmark and optimize
- **Memory issues**: Monitor data size, use chunking if needed
- **Disk space**: Parquet compression, cache management

---

## Next Steps After Implementation

1. **Phase 6: Advanced Strategies**
   - Combine multiple indicators
   - Machine learning strategies
   - Multi-timeframe analysis

2. **Phase 7: Optimization**
   - Parameter sweeps
   - Walk-forward analysis
   - Genetic algorithms for parameter tuning

3. **Phase 8: Production Deployment**
   - Signal generation automation
   - Real-time data feeds
   - Trade execution integration

4. **Phase 9: Advanced Analytics**
   - Monte Carlo simulation
   - Risk attribution
   - Factor analysis

---

This plan provides a comprehensive roadmap with clear milestones, testable deliverables, and architectural alignment with ARCHITECTURE.md. Each phase builds on the previous, with well-defined success criteria and minimal coupling between modules.
