"""Run a backtest with a trading strategy."""
import argparse
import logging

from raj.backtest.engine import BacktestEngine
from raj.backtest.metrics import PerformanceMetrics
from raj.config import DEFAULT_COMMISSION, DEFAULT_INITIAL_CAPITAL, DEFAULT_START_DATE, REPORTS_DIR
from raj.data.provider import DataProvider
from raj.strategies.examples.buy_hold import BuyAndHoldStrategy
from raj.utils.dates import get_today
from raj.visualization.charts import ChartGenerator
from raj.visualization.reports import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run backtest example."""
    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument(
        '--start-date',
        type=str,
        default=DEFAULT_START_DATE,
        help=f'Start date (YYYY-MM-DD), default: {DEFAULT_START_DATE}'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD), default: today'
    )
    parser.add_argument(
        '--universe',
        type=str,
        default='top20',
        help='Universe name, default: top20'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help=f'Initial capital, default: {DEFAULT_INITIAL_CAPITAL}'
    )
    parser.add_argument(
        '--commission',
        type=float,
        default=DEFAULT_COMMISSION,
        help=f'Commission rate, default: {DEFAULT_COMMISSION}'
    )
    parser.add_argument(
        '--show-charts',
        action='store_true',
        help='Display interactive charts'
    )
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate HTML report'
    )

    args = parser.parse_args()

    # Use today if no end date specified
    end_date = args.end_date or get_today()

    logger.info("Loading data...")
    provider = DataProvider()
    data = provider.get_universe_data(
        universe_name=args.universe,
        start_date=args.start_date,
        end_date=end_date
    )

    if len(data) == 0:
        logger.error("No data available. Please run download_data.py first.")
        return

    logger.info(f"Loaded data for {len(data)} symbols")

    # Create strategy
    logger.info("Creating strategy...")
    strategy = BuyAndHoldStrategy()

    # Create backtest engine
    logger.info("Running backtest...")
    engine = BacktestEngine(
        initial_capital=args.initial_capital,
        commission=args.commission
    )

    # Run backtest
    result = engine.run(strategy, data)

    # Print metrics
    PerformanceMetrics.print_metrics(result.metrics)

    # Show charts if requested
    if args.show_charts:
        logger.info("Generating charts...")
        dashboard = ChartGenerator.create_dashboard(result)
        ChartGenerator.show_figure(dashboard)

    # Generate report if requested
    if args.generate_report:
        logger.info("Generating report...")
        report_path = REPORTS_DIR / f"backtest_{strategy.name}_{get_today()}.html"
        generated_path = ReportGenerator.generate_backtest_report(result, str(report_path))
        logger.info(f"Report saved to: {generated_path}")


if __name__ == '__main__':
    main()
