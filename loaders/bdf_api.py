import os
import time
import requests
import pandas as pd
from services.cache import load_cache, save_cache
from utils.logger import get_logger

logger = get_logger("BDF_API")
CACHE_MAX_AGE = 10  # 86400  # 24h en secondes


def get_bdf_series(
    series_key: str,
    base_url: str,
    headers: dict,
    start_date: str = "2020-01-01",
    cache_path: str = None,
    max_cache_age: int = CACHE_MAX_AGE,
) -> pd.Series:
    """
    Fetch a time series from Banque de France Webstat API with optional caching.

    Args:
        series_key (str): Unique series ID, e.g. "ICP.M.FR.N.000000.4.INX"
        base_url (str): Webstat API endpoint URL.
        headers (dict): Required headers (with API key if needed).
        start_date (str): Filter for data after this date.
        cache_path (str): Optional path to load/save cached data.
        max_cache_age (int): Max age in seconds before cache is invalid.

    Returns:
        pd.Series: Time-indexed series of values.

    Raises:
        ValueError: If required columns are missing or data is empty.
        HTTPError: If the request fails.
    """

    # Try loading from cache if specified and valid
    if cache_path:
        cached_series = load_cache(cache_path, max_age_seconds=max_cache_age)
        if cached_series is not None:
            logger.info(f"[CACHE HIT] Loaded series '{series_key}' from {cache_path}")
            return cached_series

    logger.info(f"[API CALL] Requesting Banque de France series '{series_key}'")

    params = {
        "select": "time_period_end,obs_value",
        "where": f"series_key='{series_key}' and time_period_start>date'{start_date}'",
        "order_by": "time_period_start",
    }

    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"[ERROR] Failed to fetch BDF series '{series_key}': {e}")
        raise

    if not data:
        logger.warning(f"[EMPTY] No data returned for series '{series_key}'")
        raise ValueError(f"No data returned for BDF series '{series_key}'")

    df = pd.DataFrame(data)

    if "time_period_end" not in df or "obs_value" not in df:
        logger.error("[ERROR] Missing required columns in API response")
        raise ValueError("Missing 'time_period_end' or 'obs_value' in BDF response.")

    df["time_period_end"] = pd.to_datetime(df["time_period_end"])
    df.set_index("time_period_end", inplace=True)

    series = df["obs_value"].astype(float).sort_index()

    if cache_path:
        save_cache(cache_path, series)
        logger.info(f"[CACHE SAVE] Series '{series_key}' cached to {cache_path}")

    return series
