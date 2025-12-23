"""Date utility functions."""
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        datetime object
    """
    return pd.to_datetime(date_str)


def format_date(date: datetime) -> str:
    """Format datetime to YYYY-MM-DD string.

    Args:
        date: datetime object

    Returns:
        Date string
    """
    return date.strftime('%Y-%m-%d')


def get_today() -> str:
    """Get today's date as string.

    Returns:
        Today's date in YYYY-MM-DD format
    """
    return format_date(datetime.now())


def days_ago(n: int) -> str:
    """Get date N days ago.

    Args:
        n: Number of days

    Returns:
        Date string in YYYY-MM-DD format
    """
    date = datetime.now() - timedelta(days=n)
    return format_date(date)


def date_range(start: str, end: str, freq: str = 'D') -> pd.DatetimeIndex:
    """Generate date range.

    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)

    Returns:
        DatetimeIndex
    """
    return pd.date_range(start=start, end=end, freq=freq)
