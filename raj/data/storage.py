"""Data storage and caching using Parquet format."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from raj.config import CACHE_DIR, CACHE_METADATA_FILE

logger = logging.getLogger(__name__)


class DataStorage:
    """Manages caching of OHLCV data using Parquet format."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        """Initialize data storage.

        Args:
            cache_dir: Directory for cached data
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = CACHE_METADATA_FILE
        self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_cache_path(self, symbol: str) -> Path:
        """Get the cache file path for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Path to the Parquet file
        """
        return self.cache_dir / f"{symbol}.parquet"

    def save_data(self, symbol: str, data: pd.DataFrame):
        """Save data to Parquet cache.

        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
        """
        cache_path = self._get_cache_path(symbol)

        # Save to Parquet with compression
        data.to_parquet(cache_path, compression='snappy')

        # Update metadata
        self.metadata[symbol] = {
            'last_updated': datetime.now().isoformat(),
            'start_date': str(data.index.min().date()),
            'end_date': str(data.index.max().date()),
            'rows': len(data)
        }
        self._save_metadata()

        logger.info(f"Saved {len(data)} rows to cache for {symbol}")

    def load_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load data from Parquet cache.

        Args:
            symbol: Stock symbol
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with OHLCV data, or None if not cached
        """
        cache_path = self._get_cache_path(symbol)

        if not cache_path.exists():
            return None

        try:
            data = pd.read_parquet(cache_path)

            # Filter by date range if specified
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]

            logger.info(f"Loaded {len(data)} rows from cache for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error loading cached data for {symbol}: {e}")
            return None

    def is_cached(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> bool:
        """Check if data is cached for the given symbol and date range.

        Args:
            symbol: Stock symbol
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            True if data is fully cached, False otherwise
        """
        if symbol not in self.metadata:
            return False

        cache_path = self._get_cache_path(symbol)
        if not cache_path.exists():
            return False

        # If no date range specified, cache exists
        if not start_date and not end_date:
            return True

        # Check if cache covers the requested date range
        meta = self.metadata[symbol]
        cached_start = pd.to_datetime(meta['start_date'])
        cached_end = pd.to_datetime(meta['end_date'])

        if start_date and pd.to_datetime(start_date) < cached_start:
            return False
        if end_date and pd.to_datetime(end_date) > cached_end:
            return False

        return True

    def get_cache_info(self, symbol: str) -> Optional[Dict]:
        """Get cache metadata for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with cache information, or None if not cached
        """
        return self.metadata.get(symbol)

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data.

        Args:
            symbol: Stock symbol to clear, or None to clear all
        """
        if symbol:
            # Clear specific symbol
            cache_path = self._get_cache_path(symbol)
            if cache_path.exists():
                cache_path.unlink()
            if symbol in self.metadata:
                del self.metadata[symbol]
                self._save_metadata()
            logger.info(f"Cleared cache for {symbol}")
        else:
            # Clear all cached data
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            self.metadata = {}
            self._save_metadata()
            logger.info("Cleared all cached data")

    def list_cached_symbols(self) -> list[str]:
        """Get list of all cached symbols.

        Returns:
            List of stock symbols that have cached data
        """
        return list(self.metadata.keys())
