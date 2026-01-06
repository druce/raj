# Architecture Guide: Building a Python Backtesting System

## Purpose of This Document

This document describes the architectural design of Backtester, a modular backtesting system for technical trading strategies. It's written for developers and AI agents who want to understand how to build a robust, extensible backtesting framework using Python, pandas, ta-lib, and plotly.

The focus is on **architectural decisions, design patterns, and structural principles** rather than implementation details.

---

## Table of Contents

1. [Core Architectural Philosophy](#core-architectural-philosophy)
2. [System Overview](#system-overview)
3. [Layer-by-Layer Architecture](#layer-by-layer-architecture)
4. [Design Patterns and Rationale](#design-patterns-and-rationale)
5. [Data Flow Through the System](#data-flow-through-the-system)
6. [Critical Design Decisions](#critical-design-decisions)
7. [Extension Points](#extension-points)
8. [Performance Considerations](#performance-considerations)

---

## Core Architectural Philosophy

### Separation of Concerns

The system is built on strict separation of concerns:

1. **Data layer** is concerned with downloading from data sources like Yahoo Finance and Schwab, and storing data in Parquet.
2. **Strategy layer** transforms data into signals (no execution logic
3. **Backtest layer** transforms signals into trade history
4. **Visualization layer** charts historical performance

This separation enables:
- Testing each layer independently
- Swapping implementations (e.g., different data providers, or CSVs or a sql database like DuckDB or SQLite instead of Parquet)
- Parallel development of features
- Understandable mental model of system behavior

### Vectorization Over Event-Driven

The system uses a **vectorized signal generation** approach rather than bar-by-bar event handlers:

```python
# Vectorized approach (what Backtester uses)
def generate_signals(self, data: pd.DataFrame) -> pd.Series:
    """Return ALL signals at once using pandas operations"""
    return pd.Series(where(condition, BUY, SELL))

# NOT event-driven in a loop (what Backtester avoids)
def on_bar(self, bar):
    """Process one bar at a time"""
    if self.state.condition:
        return BUY
```

**Rationale:**
- Leverages pandas/numpy vectorization for 10-100x speed improvements. (if we were using minute data we might want to leverage a multithreaded dataframe implementation like Polars, Dark or Modin.)
- Makes strategy logic easier to understand and debug
- Enables strategy testing without running full backtests
- Natural fit for technical indicators from ta-lib

**Trade-off:** Harder to implement strategies with complex state machines, but 95% of technical strategies don't need them.

### Immutability Where Possible

Data flows through the system without mutation:
- OHLCV data is read-only after loading
- Strategies return new signal series without modifying input
- Portfolio state is isolated within the backtest engine

This prevents subtle bugs from shared mutable state.

---

## System Overview

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│     (CLI scripts: download_data, run_backtest, etc.)        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    Backtesting Engine                        │
│  • Coordinates data, strategy, and portfolio                 │
│  • Orchestrates simulation loop                              │
│  • Calculates performance metrics                            │
└───────┬──────────────────────┬─────────────────────┬─────────┘
        │                      │                     │ manages
        ▼                      ▼                     ▼
┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐
│    Strategy     │ ◀│  Data Provider   │▶ │    Dataviz        │
│  (Pluggable)    │  │  (Facade)        │  │ (Charts/Reports)  │
└─────────────────┘  └─ ──┬────┬────┬───┘  └───────────────────┘
                          │    │    │              
                          ▼    ▼    ▼              
                     ┌─────┬─────┬──────┐       
                     │Down │Store│Univ. │
                     │load │     │Mgr   │
                     └─────┴─────┴──────┘
       




```

### Data Structures

```python
# Input: OHLCV DataFrame (standardized format)
pd.DataFrame({
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': int
}, index=DatetimeIndex)

# Signal: Discrete trading decision
class SignalType(Enum):
    BUY = 1      # Enter long position
    SELL = -1    # Exit position
    HOLD = 0     # No action

# Position: Current holdings
@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    current_price: float

# Trade: Completed transaction
@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
```

**Design rationale:** These structures define clear contracts between layers. A strategy never sees a Position or Trade - it only produces signals based on market data. The backtest engine never modifies OHLCV data.

---

## Layer-by-Layer Architecture

### Layer 1: Data Infrastructure

**Purpose:** Reliably acquire and cache historical market data

**Components:**

1. **DataDownloader** - Fetches data from external sources (yfinance)
2. **DataStorage** - Parquet-based caching with metadata tracking
3. **UniverseManager** - Defines and caches stock universes
4. **DataProvider** - Unified facade over the above

**Key Design Decision: Facade Pattern**

```python
class DataProvider:
    def __init__(self):
        self.downloader = DataDownloader()
        self.storage = DataStorage()
        self.universe_manager = UniverseManager()

    def download_universe_data(self, universe_name, start_date, end_date):
        """Single method hides complexity of 3 subsystems"""
        symbols = self.universe_manager.get_top_n_by_market_cap(20)

        results = {}
        for symbol in symbols:
            if self.storage.is_cached(symbol, start_date, end_date):
                data = self.storage.load_data(symbol)
            else:
                data = self.downloader.download_stock(symbol)
                self.storage.save_data(symbol, data)
            results[symbol] = data

        return results
```

**Rationale:** The rest of the system only interacts with `DataProvider`, never with downloader/storage/universe directly. This means:
- We can swap yfinance for Alpha Vantage without changing strategy code
- We can change caching strategy (Redis instead of Parquet) in one place
- Testing is easier - mock one object instead of three

**Storage Implementation Details:**

The system uses Parquet files because:
- **Columnar format** is ideal for time-series OHLCV data
- **Compression** reduces storage by ~80% vs CSV
- **Metadata** (min/max dates) enables smart cache invalidation
- **pandas native support** makes it seamless to use

```python
# Smart caching logic
def is_cached(self, symbol, start_date, end_date):
    """Check if cache covers requested date range"""
    meta = self.metadata[symbol]
    cached_start = meta['start_date']
    cached_end = meta['end_date']

    # Only download if we need data outside cached range
    if start_date >= cached_start and end_date <= cached_end:
        return True
    return False
```



Caching prevents redundant downloads - if you have 2020-2024 cached and request 2021-2022, you get instant cache hit.

Downloads should implement retry for transient errors.

---

### Layer 2: Strategy Framework

**Purpose:** Define a clean interface for implementing trading logic

**Core Abstraction:**

```python
class BaseStrategy(ABC):
    """Abstract base class for all strategies"""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Transform OHLCV data into trading signals

        Input:  DataFrame with columns [open, high, low, close, volume]
        Output: Series of SignalType enum values (BUY/SELL/HOLD)

        This method must be vectorized - it processes the entire
        dataset at once, not bar-by-bar.
        """
        pass
```

**Design Rationale: Strategy Pattern**

Every trading strategy inherits from `BaseStrategy` and implements one method. This enables:

1. **Polymorphism** - Backtest engine works with any strategy
2. **Type safety** - All strategies have the same interface
3. **Testability** - Test signal generation independently
4. **Composability** - Combine multiple strategies easily

**Example Implementation:**

```python
class MovingAverageCrossover(BaseStrategy):
    def __init__(self, short_window=50, long_window=200):
        super().__init__(short_window=short_window, long_window=long_window)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized signal generation using pandas"""

        # Calculate indicators for entire dataset at once
        short_ma = data['close'].rolling(self.parameters['short_window']).mean()
        long_ma = data['close'].rolling(self.parameters['long_window']).mean()

        # Generate all signals in one vectorized operation
        signals = pd.Series(SignalType.HOLD, index=data.index)
        signals[short_ma > long_ma] = SignalType.BUY
        signals[short_ma < long_ma] = SignalType.SELL

        return signals
```

**Assumption:** daily change changes reflect ex-dividends, splits and corporate actions. Implicitly, we assume we reinvest cash dividends in the stock, or potentially sell spinoff and re-invest in the stock. Yfinance provides adjusted series. For other providers, we may need to take the raw data they provide and adjust in a similar way. (or make a different assumption on how to compute signals and execute in consideration of dividends corporate actions). When you do this, old share prices are not the historical price, but the price that would have bought a fraction of stock that would have become a single share with all distributions reinvested up to the current date. The series is really a total return index that lines up with current stock price for data since the last ex-dividend/corporate action. Makes sense if you are thinking about a portfolio allocated by weight, as opposed to a full-blown corporate accounting system (in which case might want to see if e.g. Zipline or Backtrader are more appropriate engines)

**Why Vectorization Matters:**

```python
# Vectorized (what we do)
signals = pd.Series(SignalType.HOLD, index=data.index)
signals[condition] = SignalType.BUY  # Processes 1000s of rows instantly

# Event-driven (what we avoid)
for i in range(len(data)):
    if condition[i]:
        signals[i] = SignalType.BUY  # 100x slower
```

For a 5-year daily backtest (1250 rows), vectorization is 50-500x faster.

**Extension**: 

- for debugging, we may want to return the signals plus intermediate values like moving averages used to generate final signals
- in this case if debug=True, return a tuple of signals and a dataframe of intermediate values

---

### Layer 3: Backtesting Engine

**Purpose:** Simulate trading in historical market conditions

**Components:**

1. **Portfolio** - Tracks cash, positions, executes trades
2. **BacktestEngine** - Main simulation coordinator
3. **PerformanceMetrics** - Calculate Sharpe, drawdown, etc.

**Architecture:**

```python
class BacktestEngine:
    def run(self, strategy: BaseStrategy, data: Dict[str, pd.DataFrame]):
        """
        Simulation loop:
        1. Generate all signals upfront (vectorized)
        2. Iterate through dates chronologically
        3. Execute signals using portfolio manager
        4. Track equity curve
        5. Calculate metrics
        """

        # Step 1: Vectorized signal generation
        signals = {
            symbol: strategy.generate_signals(df)
            for symbol, df in data.items()
        }

        # Step 2: Get common dates across all symbols
        common_dates = self._get_common_dates(data)

        # Step 3: Initialize portfolio
        portfolio = Portfolio(initial_capital, commission)
        equity_curve = []

        # Step 4: Chronological simulation
        for date in common_dates:
            # Get current prices
            prices = {sym: data[sym].loc[date, 'close'] for sym in data}

            # Update position valuations
            portfolio.update_positions(prices)

            # Execute signals for this date
            for symbol in data.keys():
                signal = signals[symbol].loc[date]
                portfolio.execute_signal(signal, prices[symbol])

            # Record equity
            equity_curve.append(portfolio.get_total_value())

        # Step 5: Calculate metrics
        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve,
            portfolio.get_trade_history(),
            initial_capital
        )

        return BacktestResult(equity_curve, trades, metrics, signals)
```

**Performance metrics:**

- total return
- annualized return
- sharpe ratio
- max drawdown
- number of trades
- win/loss rate
- average trade gain/loss
- average winning trade
- average losing trade
- best/worst day/week/month
- others tbd



**Key Design Decision: Pre-generate Signals**

Notice that signals are generated BEFORE the simulation loop, not during it:

```python
# Generate ALL signals at once (vectorized)
signals = {symbol: strategy.generate_signals(df) for symbol, df in data.items()}

# Then simulate chronologically
for date in dates:
    signal = signals[symbol].loc[date]  # Lookup, not compute
    portfolio.execute_signal(signal, price)
```

**Why this architecture?**

1. **Speed:** Signal generation uses vectorized pandas (fast), simulation loop is pure Python (slow). Do the slow part once.

2. **Lookahead bias prevention:** Even though we have all signals, we only execute them chronologically. We can't accidentally use future data.

3. **Debuggability:** You can inspect all signals before running the backtest to verify strategy logic.

4. **Testability:** Test signal generation separately from execution logic.

**Portfolio Management:**

The Portfolio class enforces realistic trading constraints:

```python
class Portfolio:
    def execute_signal(self, signal, price, portfolio_value):
        if signal.type == BUY:
            # Position sizing: 5% of portfolio per stock
            allocation = portfolio_value * 0.05
            quantity = int(allocation / price)

            # Don't buy if already have position
            if symbol in self.positions:
                return

            # Apply commission
            cost = quantity * price * (1 + self.commission)

            # Don't buy if insufficient cash
            if cost > self.cash:
                return

            self.cash -= cost
            self.positions[symbol] = Position(...)

        elif signal.type == SELL:
            # Exit entire position
            # Track trade for metrics
            # Apply commission
```

This prevents common backtesting mistakes:
- Over-allocating capital
- Ignoring transaction costs
- Taking positions you can't afford

---

### Layer 4: Visualization

**Purpose:** Present backtest results in intuitive, interactive formats

**Components:**

1. **ChartGenerator** - Creates Plotly charts (equity curve, drawdown, signals)
2. **ReportGenerator** - Generates HTML reports with embedded charts

**Design Principle: Separation of Computation and Presentation**

Visualization layer is **read-only** - it never modifies data such as backtest results:

```python
class ChartGenerator:
    @staticmethod
    def create_equity_curve(equity_series: pd.Series):
        """
        Input: Series of portfolio values
        Output: Plotly Figure object

        No business logic, just formatting
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            mode='lines',
            name='Portfolio Value'
        ))
        return fig
```

**Why Plotly?**

1. **Interactive:** Hover tooltips, zoom, pan work out of the box
2. **Portable:** Saves to standalone HTML files
3. **Professional:** Publication-quality charts with minimal code
4. **Consistent:** All charts use same theme

**Chart Generation:**

- Equity Curve
- Drawdowns
- Daily/weekly/monthly returns
- distribution of daily/weekly/monthly retuns

**Report Generation:**

```python
class ReportGenerator:
    def generate_backtest_report(self, result: BacktestResult):
        """
        Combine metrics and charts into single HTML file

        Architecture: Template pattern
        - HTML template with placeholders
        - Fill in metrics table
        - Embed Plotly charts as JavaScript
        - Write standalone file
        """
```

- table of chosen metrics
- compare 2 strategies eg buy/hold vs. MA crossover
- output html with charts + report



---

## Design Patterns and Rationale

### 1. Facade Pattern (DataProvider)

**Problem:** Data acquisition involves three complex subsystems

**Solution:** Single interface hides complexity

```python
# Without facade - user juggles 3 objects
downloader = DataDownloader()
storage = DataStorage()
universe_mgr = UniverseManager()
symbols = universe_mgr.get_top_20()
for s in symbols:
    if not storage.is_cached(s):
        data = downloader.download(s)
        storage.save(s, data)

# With facade - one simple call
provider = DataProvider()
data = provider.download_universe_data('top20')
```

### 2. Strategy Pattern (BaseStrategy)

**Problem:** Need to support unlimited trading strategies

**Solution:** Define interface, implement variants

```python
# Engine works with any strategy
def backtest(strategy: BaseStrategy, data):
    signals = strategy.generate_signals(data)
    # ... rest of backtest

# Can plug in any implementation
ma_strategy = MovingAverageCrossover(50, 200)
rsi_strategy = RSIStrategy(14)
ml_strategy = MachineLearningStrategy(model)

# All work with same engine
backtest(ma_strategy, data)
backtest(rsi_strategy, data)
backtest(ml_strategy, data)
```

### 3. Repository Pattern (DataStorage)

**Problem:** Want to swap storage backends without changing code

**Solution:** Abstract storage behind interface

```python
class DataStorage:
    """Interface that hides implementation"""
    def save_data(self, symbol, data): ...
    def load_data(self, symbol): ...
    def is_cached(self, symbol): ...

# Current: Parquet files
# Future: Could be Redis, MongoDB, PostgreSQL
# Rest of system doesn't need to change
```

### 4. Dependency Injection

**Problem:** Hard-coded dependencies make testing difficult

**Solution:** Inject dependencies, don't create them

```python
# Good - dependencies injected
class BacktestEngine:
    def __init__(self, initial_capital, commission):
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, strategy, data):
        portfolio = Portfolio(self.initial_capital, self.commission)

# Can easily test with different values
engine_low_commission = BacktestEngine(100000, 0.001)
engine_high_commission = BacktestEngine(100000, 0.01)

# Bad - hard-coded values
class BacktestEngine:
    def run(self, strategy, data):
        portfolio = Portfolio(100000, 0.001)  # Can't change without editing code
```

---

## Data Flow Through the System

### End-to-End Example: Running a Moving Average Backtest

```python
# 1. User initiates request
strategy = MovingAverageCrossover(short=50, long=200)
```

**Data Layer:**
```python
# 2. Fetch data through facade
provider = DataProvider()
data = provider.download_universe_data('top20', '2020-01-01', '2024-12-31')

# Behind the scenes:
#   a. UniverseManager checks cache, finds top 20 list
#   b. For each symbol:
#       - DataStorage checks if date range is cached
#       - If cached: load from Parquet
#       - If not: DataDownloader fetches from yfinance
#       - Save to cache for future use
#   c. Returns: Dict[str, pd.DataFrame]
```

**Strategy Layer:**
```python
# 3. Generate signals (vectorized)
signals = strategy.generate_signals(data['AAPL'])

# Behind the scenes:
#   a. Calculate 50-day MA using pandas rolling
#   b. Calculate 200-day MA using pandas rolling
#   c. Compare MAs across entire dataset at once
#   d. Return Series of BUY/SELL/HOLD signals
```

**Backtest Layer:**
```python
# 4. Run simulation
engine = BacktestEngine(initial_capital=100000, commission=0.001)
result = engine.run(strategy, data)

# Behind the scenes:
#   a. Generate signals for all stocks
#   b. Find common date range
#   c. Initialize Portfolio with cash
#   d. For each date chronologically:
#       - Update position prices
#       - Look up signals for this date
#       - Execute BUY/SELL through Portfolio
#       - Record equity
#   e. Close all positions at end
#   f. Calculate metrics (Sharpe, drawdown, etc.)
#   g. Return BacktestResult
```

**Visualization Layer:**
```python
# 5. Generate charts
from Backtester.visualization.charts import ChartGenerator

fig = ChartGenerator.create_equity_curve(result.equity_curve)
fig.show()

# Behind the scenes:
#   a. Create Plotly Figure
#   b. Add trace for equity curve
#   c. Format axes, add grid
#   d. Return interactive chart
```

### Data Transformations

```
yfinance API response (JSON)
    ↓
DataDownloader.download_stock()
    ↓
pd.DataFrame (OHLCV with DatetimeIndex)
    ↓
DataStorage.save_data()
    ↓
Parquet file on disk
    ↓
DataProvider.download_universe_data()
    ↓
Dict[str, pd.DataFrame]
    ↓
Strategy.generate_signals()
    ↓
pd.Series (SignalType enum values)
    ↓
BacktestEngine.run()
    ↓
BacktestResult (equity_curve, trades, metrics)
    ↓
ChartGenerator.create_charts()
    ↓
Plotly Figure objects
    ↓
ReportGenerator.generate_html()
    ↓
Standalone HTML file
```

---

## Critical Design Decisions

### 1. Vectorized vs Event-Driven

**Decision:** Use vectorized signal generation

**Rationale:**
- 50-500x faster for typical technical strategies
- Easier to debug (inspect all signals at once)
- Natural fit for pandas/numpy/ta-lib
- Enables backtesting thousands of parameter combinations

**Trade-off:** Complex state machines are harder to implement

**When this breaks down:** Strategies that need to track complex state (e.g., "buy on 3rd consecutive green candle after a drawdown > 10%") are awkward to vectorize. Solution: Hybrid approach or numba JIT compilation.

### 2. Long-Only Portfolio

**Decision:** Only support long positions (no shorting)

**Rationale:**
- 95% of retail traders only go long
- Short positions have different margin requirements
- Simplifies position sizing logic
- Easier to explain and reason about

**Trade-off:** Can't backtest short-selling or pairs trading strategies

**Extension path:** Add `PositionType.LONG` and `PositionType.SHORT` enum, update commission calculation.

### 3. Equal-Weight Position Sizing

**Decision:** Default to 5% of portfolio per position

**Rationale:**
- Simple to understand
- Natural diversification (max 20 positions)
- Works well for universe of 20 stocks
- No optimization needed

**Trade-off:** Not risk-adjusted (volatile stocks get same allocation as stable ones)

**Extension path:** Override `Portfolio._calculate_quantity()` to use Kelly criterion, risk parity, or volatility targeting.

### 4. Market-On-Close Orders

**Default:** All orders execute at close price

**Rationale:**
- Most retail traders use daily data
- Close price is the most reliable (highest volume)
- Avoids intraday complexity
- Matches how most people think ("buy at end of day")

**Trade-off:** Ignores slippage and overnight gaps

**Extension path:** 

- Also support next day at open execution
- Add slippage parameter and logic
- Add custom function logic for execution price
- consider dynamics of shorts, short rebate, fees, and fully support short positions
- consider dynamics of leverage 
- consider constraints, like if you universe is 60 stocks , position size is 5%, maybe if all stocks have buy signals you can only buy up to 150% leverage

### 5. Parquet Storage

**Decision:** Use Parquet files for caching

**Rationale:**
- Columnar format perfect for time-series
- 80% compression vs CSV
- Fast range queries (read 2022 data without loading full file)
- Native pandas integration

**Trade-off:** Binary format (can't inspect with text editor)

**Alternative considered:** CSV files (human-readable but slow and large), SQLite (query flexibility but overhead for simple use case).

### 6. Percentage-Based Commission

**Decision:** Commission = % of trade value (default 0.1%)

**Rationale:**
- Matches most brokers (e.g., 0.1% per trade)
- Scales naturally with position size
- Simple to calculate

**Trade-off:** Some brokers charge per-share or have minimums

**Extension path:** Add `CommissionModel` class with methods for different schemes.

---

## Extension Points

### Adding a New Data Source

**Current:** yfinance API

**To add:** Alpha Vantage, Polygon.io, CSV files, etc.

```python
# 1. Implement downloader interface
class AlphaVantageDownloader:
    def download_stock(self, symbol, start_date, end_date) -> pd.DataFrame:
        """Return DataFrame with OHLCV columns"""
        # Fetch from Alpha Vantage API
        # Normalize to standard format
        return df

# 2. Inject into DataProvider
provider = DataProvider()
provider.downloader = AlphaVantageDownloader()

# Rest of system works unchanged
```

**Key insight:** As long as the downloader returns a DataFrame with the right schema, nothing else needs to change.

### Adding a New Technical Indicator

**Using ta-lib:**

```python
import talib

class BollingerBandsStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # ta-lib operates on numpy arrays
        upper, middle, lower = talib.BBANDS(
            data['close'].values,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2
        )

        # Vectorized signal logic
        signals = pd.Series(SignalType.HOLD, index=data.index)
        signals[data['close'] < lower] = SignalType.BUY
        signals[data['close'] > upper] = SignalType.SELL

        return signals
```

**Custom indicator:**

```python
def calculate_custom_indicator(df):
    """Custom vectorized indicator"""
    return (df['high'] - df['low']) / df['close']

class CustomStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        indicator = calculate_custom_indicator(data)

        signals = pd.Series(SignalType.HOLD, index=data.index)
        signals[indicator > 0.05] = SignalType.BUY
        signals[indicator < 0.02] = SignalType.SELL

        return signals
```

### Adding a New Performance Metric

```python
# 1. Add method to PerformanceMetrics class
class PerformanceMetrics:
    @staticmethod
    def calculate_calmar_ratio(equity_curve: pd.Series) -> float:
        """Annualized return / Max drawdown"""
        ann_return = PerformanceMetrics.calculate_annualized_return(equity_curve)
        max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        return ann_return / max_dd

    @staticmethod
    def calculate_all_metrics(equity_curve, trades, initial_capital):
        metrics = {
            # ... existing metrics
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(equity_curve)
        }
        return metrics
```

### Adding Multi-Asset Support

**Current:** Stocks only

**Extension:** Add crypto, forex, futures

```python
# 1. Add asset type field
@dataclass
class Position:
    symbol: str
    asset_type: AssetType  # NEW
    quantity: float  # Change from int (crypto has decimals)
    # ...

# 2. Asset-specific commission models
class CommissionModel:
    def calculate(self, asset_type, quantity, price):
        if asset_type == AssetType.STOCK:
            return quantity * price * 0.001
        elif asset_type == AssetType.CRYPTO:
            return quantity * price * 0.002  # Higher fees
        elif asset_type == AssetType.FOREX:
            return quantity * price * 0.00005  # Spread-based
```

---

## Performance Considerations

### Vectorization is Critical

**Slow (event-driven):**
```python
signals = []
for i in range(len(data)):
    if data['close'].iloc[i] > data['open'].iloc[i]:
        signals.append(SignalType.BUY)
    else:
        signals.append(SignalType.SELL)
```

**Fast (vectorized):**
```python
signals = pd.Series(SignalType.HOLD, index=data.index)
signals[data['close'] > data['open']] = SignalType.BUY
signals[data['close'] <= data['open']] = SignalType.SELL
```

**Benchmark:** For 10,000 rows, vectorized is ~200x faster.

### Cache Everything Reasonable

The system caches:
1. **Downloaded data** (Parquet files) - avoid API rate limits
2. **Universe definitions** (JSON files) - market cap rankings change slowly
3. **Metadata** (JSON file) - avoid filesystem scans

What we DON'T cache:
- Signals (cheap to regenerate, change with parameters)
- Backtest results (depend on many inputs)
- Charts (fast to render)

### Parallelize Where Possible

**Current:** Sequential processing of symbols

**Optimization opportunity:**

```python
from concurrent.futures import ThreadPoolExecutor

def download_universe_data(self, universe_name, start_date, end_date):
    symbols = self.universe_manager.get_top_n(20)

    # Download in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(self._get_single_stock, sym): sym
            for sym in symbols
        }

        results = {}
        for future in futures:
            symbol = futures[future]
            results[symbol] = future.result()

    return results
```

This can reduce initial download time from 60s to 15s for 20 stocks.

### Memory Efficiency

**Current approach:** Load all data into memory

**Works well for:**
- Daily data (small: 1 year = 252 rows)
- Small universes (20 stocks)
- Date ranges (5 years = 1250 rows per stock)

**Memory usage:** 20 stocks × 5 years × 6 columns × 8 bytes = ~480 KB

**Breaks down at:**
- Minute data (390 bars/day × 252 days × 5 years = ~500K rows)
- Large universes (500+ stocks)

**Solution for large datasets:** Chunked processing with Dask or iterator-based backtest loop.

---

## Comparison with Alternative Architectures

### Event-Driven Frameworks (e.g., Zipline, Backtrader)

**Their approach:**
```python
class Strategy:
    def on_bar(self, bar):
        if self.moving_average > bar.close:
            self.buy(symbol, quantity=100)
```

**Pros:**
- Natural for complex state machines
- Easy to understand for beginners
- Matches real-time trading

**Cons:**
- Slow (pure Python loops)
- Hard to debug (can't see all signals at once)
- Requires framework-specific API

**Backtester's approach:**
```python
class Strategy:
    def generate_signals(self, data):
        ma = data['close'].rolling(50).mean()
        return pd.Series(where(ma > data['close'], BUY, SELL))
```

**Pros:**
- Fast (vectorized)
- Easy to test (pure function)
- Framework-independent (just pandas)

**Cons:**
- Harder for complex state
- Doesn't match real-time trading flow

**Verdict:** For technical strategies on daily data, vectorized is superior. For complex discretionary strategies or real-time systems, event-driven may be better.

### Research-Oriented Platforms (e.g., QuantConnect, Quantopian)

**Their approach:** Integrated cloud platform with data + execution + backtesting

**Pros:**
- No data sourcing needed
- Professional infrastructure
- Live trading integration

**Cons:**
- Vendor lock-in
- Monthly fees
- Limited customization

**Backtester's approach:** Modular local system

**Pros:**
- Free and open
- Full control and customization
- Educational (see how everything works)

**Cons:**
- DIY data sourcing
- No live trading built-in
- Requires Python knowledge

**Verdict:** Use Backtester for learning, research, and full control. Use platforms for production trading.

---

## Conclusion

The Backtester architecture demonstrates key principles for building backtesting systems:

1. **Separate layers** - data, strategy, execution, visualization
2. **Vectorize computation** - leverage pandas/numpy for 100x speedups
3. **Use design patterns** - Facade, Strategy, Repository for clean code
4. **Cache aggressively** - avoid redundant API calls
5. **Make signals immutable** - generate once, execute chronologically
6. **Test components independently** - each layer has clear contracts

This architecture scales from simple moving average strategies to complex multi-factor models while maintaining code clarity and performance.

**For AI agents building similar systems:**
- Start with data layer (caching is critical)
- Make strategy interface your core abstraction
- Separate signal generation from execution
- Vectorize everything possible
- Add visualization last (it's just formatting)

The pattern "load data → generate signals → simulate execution → calculate metrics → visualize" applies to nearly all backtesting problems, regardless of asset class or strategy type.
