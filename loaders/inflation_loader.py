import pandas as pd
from config.settings import InflationConfig

# ton client API existant
from loaders.bdf_api import get_bdf_series


def load_inflation_series(config: InflationConfig) -> pd.Series:
    """
    Inflation Data Loader
    ======================

    Pure functional loader for inflation time series.

    Overview
    --------
    This function is responsible for retrieving raw inflation data
    from the external Banque de France (or equivalent) API layer.

    It acts as a **stateless adapter** between:
        - external data source (BDF API)
        - internal financial services (InflationService)

    Design principles
    -----------------
    - Pure function (no side effects)
    - Stateless (no caching, no mutation)
    - Deterministic given the same config
    - Delegates all I/O to lower-level API client

    Responsibilities
    ---------------
    - Fetch inflation time series using API credentials
    - Apply minimal validation (non-empty dataset check)
    - Return a clean pandas Series for downstream services

    Non-responsibilities
    --------------------
    This function MUST NOT:
    - cache data (handled by InflationService)
    - manage threads or concurrency
    - persist data to disk
    - transform or normalize series beyond basic validation

    Parameters
    ----------
    config : InflationConfig
        Configuration object containing:
        - API key / dataset identifier
        - base URL of the data provider
        - request headers
        - start date for historical extraction

    Returns
    -------
    pd.Series
        Inflation time series indexed by date.

    Raises
    ------
    ValueError
        If the API returns an empty or invalid dataset.

    Notes
    -----
    This function is intentionally minimal and should remain stable
    even if caching or orchestration logic evolves elsewhere.
    """

    df = get_bdf_series(
        config.key,
        config.base_url,
        config.headers,
        start_date=config.start_date,
    )

    if df is None or df.empty:
        raise ValueError("Inflation dataset is empty")

    return df
