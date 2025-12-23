"""Data downloader using yfinance."""
import logging
from typing import Dict, Optional
from datetime import datetime

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataDownloader:
    """Downloads OHLCV data from yfinance."""

    def __init__(self):
        """Initialize the data downloader."""
        pass

    def download_stock(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Download OHLCV data for a single stock.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format (default: 2020-01-01)
            end_date: End date in YYYY-MM-DD format (default: today)

        Returns:
            DataFrame with OHLCV data indexed by date

        Raises:
            ValueError: If no data is available for the symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if data.empty:
                raise ValueError(f"No data available for {symbol}")

            # Rename columns to lowercase for consistency
            data.columns = [col.lower() for col in data.columns]

            # Keep only OHLCV columns
            data = data[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Downloaded {len(data)} rows for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            raise

    def download_multiple(
        self,
        symbols: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Download OHLCV data for multiple stocks.

        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        iterator = tqdm(symbols, desc="Downloading") if show_progress else symbols

        for symbol in iterator:
            try:
                data = self.download_stock(symbol, start_date, end_date)
                results[symbol] = data
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")
                continue

        return results

    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get the market capitalization for a stock.

        Args:
            symbol: Stock symbol

        Returns:
            Market cap in USD, or None if unavailable
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('marketCap')
        except Exception as e:
            logger.warning(f"Could not get market cap for {symbol}: {e}")
            return None

    def get_info(self, symbol: str) -> Dict:
        """Get ticker info from yfinance.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with ticker information
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {e}")
            return {}
