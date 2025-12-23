"""Download data for top 20 stocks."""
import argparse
import logging

from raj.config import DEFAULT_START_DATE
from raj.data.provider import DataProvider
from raj.utils.dates import get_today

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Download data for top 20 stocks."""
    parser = argparse.ArgumentParser(description='Download stock data')
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
        '--force-refresh',
        action='store_true',
        help='Force re-download even if cached'
    )
    parser.add_argument(
        '--refresh-universe',
        action='store_true',
        help='Refresh universe definition'
    )

    args = parser.parse_args()

    # Use today if no end date specified
    end_date = args.end_date or get_today()

    logger.info(f"Downloading data for universe: {args.universe}")
    logger.info(f"Date range: {args.start_date} to {end_date}")

    # Create data provider
    provider = DataProvider()

    # Download data
    data = provider.get_universe_data(
        universe_name=args.universe,
        start_date=args.start_date,
        end_date=end_date,
        force_refresh=args.force_refresh,
        refresh_universe=args.refresh_universe
    )

    # Print summary
    logger.info(f"\nDownload Summary:")
    logger.info(f"  Symbols downloaded: {len(data)}")
    logger.info(f"  Symbols: {', '.join(sorted(data.keys()))}")

    for symbol, df in data.items():
        logger.info(f"  {symbol}: {len(df)} rows, "
                   f"{df.index.min().date()} to {df.index.max().date()}")

    logger.info(f"\nData cached successfully!")


if __name__ == '__main__':
    main()
