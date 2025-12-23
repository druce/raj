"""Unified data provider interface."""
import logging
from typing import Dict, Optional

import pandas as pd

from raj.config import DEFAULT_START_DATE
from raj.data.downloader import DataDownloader
from raj.data.storage import DataStorage
from raj.data.universe import UniverseManager

logger = logging.getLogger(__name__)


class DataProvider:
    """Unified interface for accessing stock data with caching."""

    def __init__(self):
        """Initialize the data provider."""
        self.downloader = DataDownloader()
        self.storage = DataStorage()
        self.universe_manager = UniverseManager()

    def get_data(
        self,
        symbols: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Get data for specified symbols with smart caching.

        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: Force re-download even if cached

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        start_date = start_date or DEFAULT_START_DATE
        results = {}

        for symbol in symbols:
            # Check if we should use cache
            if not force_refresh and self.storage.is_cached(symbol, start_date, end_date):
                logger.info(f"Loading {symbol} from cache")
                data = self.storage.load_data(symbol, start_date, end_date)
                if data is not None:
                    results[symbol] = data
                    continue

            # Download fresh data
            logger.info(f"Downloading {symbol}")
            try:
                data = self.downloader.download_stock(symbol, start_date, end_date)
                self.storage.save_data(symbol, data)
                results[symbol] = data
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                continue

        return results

    def get_universe_data(
        self,
        universe_name: str = "top20",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
        refresh_universe: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Get data for a predefined universe.

        Args:
            universe_name: Name of the universe (e.g., 'top20')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: Force re-download data even if cached
            refresh_universe: Force refresh universe definition

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        # Get universe symbols
        if universe_name.startswith('top'):
            # Extract number from 'topN'
            try:
                n = int(universe_name[3:])
            except ValueError:
                n = 20
            symbols = self.universe_manager.get_top_n_by_market_cap(n, refresh=refresh_universe)
        else:
            symbols = self.universe_manager.load_universe(universe_name)
            if symbols is None:
                raise ValueError(f"Universe '{universe_name}' not found")

        logger.info(f"Getting data for {len(symbols)} stocks in universe '{universe_name}'")

        # Get data for all symbols
        return self.get_data(symbols, start_date, end_date, force_refresh)

    def update_data(
        self,
        symbols: Optional[list[str]] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Update cached data with latest available data.

        Args:
            symbols: List of symbols to update (None = all cached symbols)
            end_date: End date for update (None = today)

        Returns:
            Dictionary mapping symbols to updated DataFrames
        """
        if symbols is None:
            symbols = self.storage.list_cached_symbols()

        if not symbols:
            logger.warning("No symbols to update")
            return {}

        logger.info(f"Updating {len(symbols)} symbols")

        results = {}
        for symbol in symbols:
            try:
                # Get cache info to determine start date
                cache_info = self.storage.get_cache_info(symbol)
                if cache_info:
                    # Start from the day after last cached date
                    start_date = cache_info['end_date']
                else:
                    start_date = DEFAULT_START_DATE

                # Download new data
                data = self.downloader.download_stock(symbol, start_date, end_date)

                # Load existing cached data
                existing_data = self.storage.load_data(symbol)

                if existing_data is not None:
                    # Combine old and new data
                    combined_data = pd.concat([existing_data, data])
                    # Remove duplicates, keeping last occurrence
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    # Sort by date
                    combined_data = combined_data.sort_index()
                else:
                    combined_data = data

                # Save updated data
                self.storage.save_data(symbol, combined_data)
                results[symbol] = combined_data

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                continue

        return results

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data.

        Args:
            symbol: Symbol to clear (None = clear all)
        """
        self.storage.clear_cache(symbol)

    def get_available_symbols(self) -> list[str]:
        """Get list of symbols with cached data.

        Returns:
            List of stock symbols
        """
        return self.storage.list_cached_symbols()
