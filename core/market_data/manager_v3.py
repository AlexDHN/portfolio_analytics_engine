# core/market_data/manager.py

import os
import pandas as pd
import yfinance as yf
from typing import List, Optional, Union
from utils.logger import get_logger
from services.cache import load_cache, save_cache

CACHE_MAX_AGE = 3600  # 1 hour * 24 * 30  # 30 days


class MarketDataManager:
    """
    MarketDataManager
    =================
    Professional-grade manager for downloading, caching, and updating market data
    using Yahoo Finance.

    Features:
    - Download one or multiple tickers directly
    - Maintain MultiIndex columns (level 0 = ticker, level 1 = field)
    - Incremental updates with cache
    - Automatic forward-fill of missing data
    - Single call to download all tickers for efficiency
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        cache_dir: str = "cache/market_data",
        cache_name: str = "all_tickers.pkl",
        force_refresh: bool = False,
    ):
        self.tickers = sorted(set(tickers or []))
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(self.cache_dir, cache_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.logger = get_logger(self.__class__.__name__)
        self._data: pd.DataFrame = pd.DataFrame()

        # Initial load
        if self.tickers:
            self.logger.info(f"Initializing with tickers: {self.tickers}")
            self._data = self._load_market_data(
                self.tickers, force_refresh=force_refresh
            )
        else:
            self.logger.warning("No tickers provided. Manager initialized empty.")

    # -------------------------------------------------------------------------
    # Core data loading
    # -------------------------------------------------------------------------
    def _load_market_data(
        self,
        tickers: List[str],
        force_refresh: bool = False,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        auto_adjust: bool = False,
        ffill: bool = True,
    ) -> pd.DataFrame:
        """
        Load or update market data for given tickers.

        Logic:
        1. Load cache if exists.
        2. Determine missing or outdated tickers.
        3. Remove last row of cache to overwrite last date.
        4. Download all tickers in a single yf.download call.
        5. Merge cache and downloaded data, forward-fill missing values.
        6. Save updated cache.

        Returns a MultiIndex DataFrame: level 0 = ticker, level 1 = field.
        """

        if isinstance(tickers, str):
            tickers = [tickers]

        tickers = list(set(tickers))

        cached_df = None

        if not force_refresh and os.path.exists(self.cache_path):
            cached_df = load_cache(self.cache_path, max_age_seconds=CACHE_MAX_AGE)

        # Remove last row to overwrite last date
        if cached_df is not None and not cached_df.empty:
            existing_tickers = set(cached_df.columns.levels[0])
            old_tickers = [t for t in tickers if t in existing_tickers]
            new_tickers = [t for t in tickers if t not in existing_tickers]
            if new_tickers == []:
                return cached_df
            else:
                return self._load_market_data(tickers, force_refresh=True)
                # cached_df = cached_df.iloc[:-1]

        download_start = pd.Timestamp(start)

        """
        if cached_df is not None and not cached_df.empty:
            download_start = cached_df.index.max()
        """

        self.logger.info(
            f"Downloading tickers {tickers} from {download_start.date()} to {end or 'today'}"
        )

        # Single download for all tickers
        new_data = yf.download(
            tickers=tickers,
            start=download_start,
            end=end,
            actions=True,
            auto_adjust=auto_adjust,
            group_by="Ticker",
            threads=True,
            progress=False,
        )

        # Merge cache and new data
        combined = (
            pd.concat([cached_df, new_data], axis=0)
            if cached_df is not None
            else new_data
        )

        # Remove duplicate indices if any
        combined = combined[~combined.index.duplicated(keep="last")]

        if ffill:
            combined = combined.ffill()

        # Save cache
        save_cache(self.cache_path, combined)
        self.logger.info(f"Cache updated: {len(combined)} rows for {tickers}")

        return combined

    # -------------------------------------------------------------------------
    # Public access methods
    # -------------------------------------------------------------------------
    def get_market_data(
        self, tickers: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Retrieve market data for one or multiple tickers.
        Automatically downloads missing tickers if needed.
        Returns MultiIndex DataFrame: level 0 = ticker, level 1 = field.
        """
        if tickers is None:
            return self._data

        if isinstance(tickers, str):
            tickers = [tickers]

        missing = [t for t in tickers if t not in self._data.columns.levels[0]]
        if missing:
            self.logger.info(f"Downloading missing tickers: {missing}")
            new_df = self._load_market_data(missing)
            self._data = pd.concat([self._data, new_df], axis=1)
            self._data = self._data[~self._data.index.duplicated(keep="last")]
            self.tickers = sorted(set(self.tickers + missing))
            save_cache(self.cache_path, self._data)

        return self._data[tickers] if len(tickers) > 1 else self._data[tickers[0]]

    def refresh(self) -> pd.DataFrame:
        """
        Force a full refresh of all tracked tickers.
        """
        self.logger.info("Performing full data refresh...")
        self._data = self._load_market_data(self.tickers, force_refresh=True)
        return self._data
