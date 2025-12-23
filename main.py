"""Main entry point for the trading system."""
import logging

from raj.backtest.engine import BacktestEngine
from raj.backtest.metrics import PerformanceMetrics
from raj.data.provider import DataProvider
from raj.strategies.examples.buy_hold import BuyAndHoldStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run a complete trading system demo."""
    print("\n" + "="*60)
    print("RAJ TRADING SYSTEM - DEMO")
    print("="*60 + "\n")

    # Step 1: Load data
    print("Step 1: Loading data for top 20 stocks...")
    provider = DataProvider()

    data = provider.get_universe_data(
        universe_name='top20',
        start_date='2020-01-01'
    )

    if len(data) == 0:
        print("\nNo cached data found. Please run:")
        print("  python examples/download_data.py")
        return

    print(f"  Loaded {len(data)} symbols")
    print(f"  Symbols: {', '.join(sorted(data.keys()))}")

    # Step 2: Create strategy
    print("\nStep 2: Creating trading strategy...")
    strategy = BuyAndHoldStrategy()
    print(f"  Strategy: {strategy.name}")

    # Step 3: Run backtest
    print("\nStep 3: Running backtest...")
    engine = BacktestEngine(
        initial_capital=100_000,
        commission=0.001
    )

    result = engine.run(strategy, data, show_progress=True)

    # Step 4: Display results
    print("\nStep 4: Backtest Results")
    PerformanceMetrics.print_metrics(result.metrics)

    # Step 5: Next steps
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Run a backtest with visualization:")
    print("   python examples/run_backtest.py --show-charts")
    print("\n2. Generate HTML report:")
    print("   python examples/run_backtest.py --generate-report")
    print("\n3. Generate trading signals:")
    print("   python examples/generate_signals.py")
    print("\n4. Explore strategies in Jupyter:")
    print("   jupyter notebook notebooks/strategy_exploration.ipynb")
    print("\n5. Create your own strategy:")
    print("   - Inherit from raj.strategies.base.BaseStrategy")
    print("   - Implement generate_signals() method")
    print("   - See notebooks/strategy_exploration.ipynb for examples")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
