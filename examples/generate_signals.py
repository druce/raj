"""Generate trading signals for today."""
import argparse
import json
import logging

from raj.config import DEFAULT_START_DATE, OUTPUT_DIR
from raj.data.provider import DataProvider
from raj.strategies.examples.buy_hold import BuyAndHoldStrategy
from raj.utils.dates import get_today

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Generate trading signals."""
    parser = argparse.ArgumentParser(description='Generate trading signals')
    parser.add_argument(
        '--universe',
        type=str,
        default='top20',
        help='Universe name, default: top20'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=DEFAULT_START_DATE,
        help=f'Start date for data (YYYY-MM-DD), default: {DEFAULT_START_DATE}'
    )
    parser.add_argument(
        '--output-json',
        action='store_true',
        help='Save signals to JSON file'
    )

    args = parser.parse_args()

    today = get_today()

    logger.info(f"Generating signals for {today}")
    logger.info(f"Universe: {args.universe}")

    # Load data (update to today)
    logger.info("Loading data...")
    provider = DataProvider()

    # Update cached data to include today
    logger.info("Updating data cache...")
    data = provider.get_universe_data(
        universe_name=args.universe,
        start_date=args.start_date,
        end_date=today
    )

    if len(data) == 0:
        logger.error("No data available. Please run download_data.py first.")
        return

    logger.info(f"Loaded data for {len(data)} symbols")

    # Create strategy
    strategy = BuyAndHoldStrategy()

    # Generate signals for all symbols
    logger.info("Generating signals...")
    signals_today = {}

    for symbol, df in data.items():
        if len(df) == 0:
            continue

        # Generate signals for entire dataset
        signals = strategy.generate_signals(df)

        # Get today's signal
        if len(signals) > 0:
            latest_date = signals.index[-1]
            latest_signal = signals.iloc[-1]
            signals_today[symbol] = {
                'date': str(latest_date.date()),
                'signal': latest_signal.name,
                'price': float(df.iloc[-1]['close'])
            }

    # Display signals
    print("\n" + "="*60)
    print(f"TRADING SIGNALS FOR {today}")
    print("="*60)
    print(f"Strategy: {strategy.name}")
    print(f"Signals generated: {len(signals_today)}")
    print()

    # Group by signal type
    buy_signals = {k: v for k, v in signals_today.items() if v['signal'] == 'BUY'}
    sell_signals = {k: v for k, v in signals_today.items() if v['signal'] == 'SELL'}
    hold_signals = {k: v for k, v in signals_today.items() if v['signal'] == 'HOLD'}

    if buy_signals:
        print(f"\nBUY Signals ({len(buy_signals)}):")
        for symbol, info in sorted(buy_signals.items()):
            print(f"  {symbol:6s} @ ${info['price']:.2f}")

    if sell_signals:
        print(f"\nSELL Signals ({len(sell_signals)}):")
        for symbol, info in sorted(sell_signals.items()):
            print(f"  {symbol:6s} @ ${info['price']:.2f}")

    if hold_signals:
        print(f"\nHOLD Signals ({len(hold_signals)}):")
        for symbol, info in sorted(hold_signals.items()):
            print(f"  {symbol:6s} @ ${info['price']:.2f}")

    print("\n" + "="*60 + "\n")

    # Save to JSON if requested
    if args.output_json:
        output_file = OUTPUT_DIR / f"signals_{today}.json"
        output_data = {
            'date': today,
            'strategy': strategy.name,
            'signals': signals_today
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Signals saved to: {output_file}")


if __name__ == '__main__':
    main()
