"""Configuration constants for the trading system."""
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
UNIVERSE_DIR = DATA_DIR / "universes"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Trading parameters
DEFAULT_COMMISSION = 0.001  # 0.1% per trade
DEFAULT_INITIAL_CAPITAL = 100_000
DEFAULT_START_DATE = "2020-01-01"

# Data parameters
DEFAULT_TOP_N_STOCKS = 20
CACHE_METADATA_FILE = CACHE_DIR / "metadata.json"

# Universe refresh parameters
UNIVERSE_REFRESH_DAYS = 7  # Refresh universe if older than 7 days
