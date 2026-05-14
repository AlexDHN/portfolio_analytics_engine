import os
import pandas as pd
from utils.logger import get_logger
from loaders.bdf_api import get_bdf_series
from loaders.sg_pee import get_sg_pee_data
from config.settings import (
    BDF_BASE_URL,
    BDF_HEADERS,
    KEY_INFLATION,
    KEY_LIVRET_A,
    ISIN_PEE,
    DRIVER_PATH,
)

logger = get_logger("Main")


def ensure_cache_dir(path: str = "cache"):
    """Ensure cache directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created cache directory at '{path}'")


def main():
    logger.info("=== Starting financial data load ===")
    ensure_cache_dir()

    data_sources = {
        "Livret A": {
            "func": get_bdf_series,
            "params": {
                "series_key": KEY_LIVRET_A,
                "base_url": BDF_BASE_URL,
                "headers": BDF_HEADERS,
                "start_date": "2020-01-01",
                "cache_path": "cache/livret_a.pkl",
            },
        },
        "Inflation": {
            "func": get_bdf_series,
            "params": {
                "series_key": KEY_INFLATION,
                "base_url": BDF_BASE_URL,
                "headers": BDF_HEADERS,
                "start_date": "2020-01-01",
                "cache_path": "cache/inflation.pkl",
            },
        },
        "PEE": {
            "func": get_sg_pee_data,
            "params": {
                "isin": ISIN_PEE,
                "driver_path": DRIVER_PATH,
                "headless": True,
                "cache_path": "cache/pee.pkl",
            },
        },
    }

    results = {}

    for label, info in data_sources.items():
        try:
            logger.info(f"Loading {label} data...")
            df = info["func"](**info["params"])
            logger.info(f"{label} data loaded successfully ({len(df)} rows)")
            results[label] = df
        except Exception as e:
            logger.error(f"Failed to load {label} data: {e}")
            results[label] = None

    logger.info("=== Financial data load completed ===")

    # Optionally log previews at DEBUG level
    for label, df in results.items():
        if df is not None:
            logger.debug(f"{label} data preview:\n{df.head()}")

    return results


if __name__ == "__main__":
    main()
