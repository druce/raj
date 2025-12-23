"""Stock universe management."""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from raj.config import DEFAULT_TOP_N_STOCKS, UNIVERSE_DIR, UNIVERSE_REFRESH_DAYS

logger = logging.getLogger(__name__)


class UniverseManager:
    """Manages stock universes (e.g., top N stocks by market cap)."""

    def __init__(self, universe_dir: Path = UNIVERSE_DIR):
        """Initialize universe manager.

        Args:
            universe_dir: Directory for storing universe definitions
        """
        self.universe_dir = universe_dir
        self.universe_dir.mkdir(parents=True, exist_ok=True)

    def get_top_n_by_market_cap(
        self,
        n: int = DEFAULT_TOP_N_STOCKS,
        refresh: bool = False
    ) -> List[str]:
        """Get top N stocks by market capitalization.

        Args:
            n: Number of stocks to return
            refresh: Force refresh even if cached

        Returns:
            List of stock symbols
        """
        # Check for cached universe
        universe_name = f"top{n}"
        if not refresh:
            cached = self._load_cached_universe(universe_name)
            if cached:
                return cached

        logger.info(f"Fetching top {n} stocks by market cap...")

        # Get top stocks - using a simplified approach
        # In production, you might want to use a more sophisticated method
        symbols = self._fetch_top_stocks(n)

        # Save universe
        self.save_universe(universe_name, symbols)

        return symbols

    def _fetch_top_stocks(self, n: int) -> List[str]:
        """Fetch top N stocks by market cap.

        This is a simplified implementation that uses a predefined list
        of large-cap stocks. For production use, consider using an
        external API or data source.

        Args:
            n: Number of stocks to return

        Returns:
            List of stock symbols
        """
        # List of major US stocks by approximate market cap
        # This is a simplified approach - in production, use a dynamic source
        major_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "BRK-B", "LLY", "V",
            "UNH", "XOM", "WMT", "JPM", "JNJ",
            "MA", "PG", "AVGO", "HD", "CVX",
            "MRK", "ABBV", "COST", "KO", "PEP"
        ]

        # Fetch market caps for these stocks
        stocks_with_mcap = []
        for symbol in major_stocks[:n*2]:  # Fetch extra in case some fail
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                mcap = info.get('marketCap')
                if mcap:
                    stocks_with_mcap.append((symbol, mcap))
            except Exception as e:
                logger.warning(f"Could not get market cap for {symbol}: {e}")
                continue

        # Sort by market cap and take top N
        stocks_with_mcap.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in stocks_with_mcap[:n]]

        logger.info(f"Found {len(top_symbols)} stocks")
        return top_symbols

    def _load_cached_universe(self, universe_name: str) -> Optional[List[str]]:
        """Load cached universe if it exists and is recent.

        Args:
            universe_name: Name of the universe

        Returns:
            List of symbols if cache is valid, None otherwise
        """
        # Find the most recent universe file
        pattern = f"{universe_name}_*.json"
        universe_files = sorted(self.universe_dir.glob(pattern), reverse=True)

        if not universe_files:
            return None

        latest_file = universe_files[0]

        # Parse date from filename
        try:
            date_str = latest_file.stem.split('_')[1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Check if cache is still fresh
            if datetime.now() - file_date > timedelta(days=UNIVERSE_REFRESH_DAYS):
                logger.info(f"Cached universe is stale ({file_date.date()})")
                return None

            # Load universe
            with open(latest_file, 'r') as f:
                data = json.load(f)

            logger.info(f"Loaded cached universe from {latest_file.name}")
            return data['symbols']

        except Exception as e:
            logger.warning(f"Error loading cached universe: {e}")
            return None

    def save_universe(self, name: str, symbols: List[str]):
        """Save a universe to disk.

        Args:
            name: Universe name
            symbols: List of stock symbols
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{name}_{date_str}.json"
        filepath = self.universe_dir / filename

        data = {
            'name': name,
            'date': date_str,
            'symbols': symbols,
            'count': len(symbols)
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved universe '{name}' with {len(symbols)} symbols to {filename}")

    def load_universe(self, name: str) -> Optional[List[str]]:
        """Load a universe by name (latest version).

        Args:
            name: Universe name

        Returns:
            List of symbols, or None if not found
        """
        # Find the most recent universe file
        pattern = f"{name}_*.json"
        universe_files = sorted(self.universe_dir.glob(pattern), reverse=True)

        if not universe_files:
            logger.warning(f"Universe '{name}' not found")
            return None

        latest_file = universe_files[0]

        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            return data['symbols']
        except Exception as e:
            logger.error(f"Error loading universe '{name}': {e}")
            return None

    def list_universes(self) -> List[str]:
        """List all available universes.

        Returns:
            List of universe names
        """
        universe_files = self.universe_dir.glob("*.json")
        names = set()
        for f in universe_files:
            # Extract universe name (before the date)
            name = '_'.join(f.stem.split('_')[:-1])
            names.add(name)
        return sorted(names)
