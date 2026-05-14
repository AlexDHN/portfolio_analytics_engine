import os
import time
import threading
import logging
from typing import Callable, Optional

import pandas as pd

from config.settings import InflationConfig
from services.cache import load_cache, save_cache


class InflationService:
    """
    InflationService
    ================

    Thread-safe hybrid caching system for inflation time series.

    Overview
    --------
    This service provides access to inflation data using a 3-layer cache:

        1. Memory cache (fast, in-process)
        2. Disk cache (persistent across sessions)
        3. Remote API loader (source of truth)

    It is designed for financial applications such as:
    - portfolio analytics
    - macroeconomic adjustment (real returns)
    - backtesting frameworks
    - multi-threaded investment simulations

    Design principles
    -----------------
    - Read-heavy optimized (inflation is immutable in most workflows)
    - Thread-safe initialization (single-flight loading)
    - TTL-based invalidation (cache freshness control)
    - Fail-safe disk cache (never breaks execution)
    - Optional asynchronous refresh for stale data

    Cache policy
    ------------
    If force_refresh=False:
        1. Memory cache (if valid)
        2. Disk cache (if valid)
        3. API loader

    If force_refresh=True:
        → bypass ALL caches and call API directly

    Notes
    -----
    Inflation data is assumed to be updated periodically (not real-time).
    """

    def __init__(
        self,
        loader: Callable[[InflationConfig], pd.Series],
        config: InflationConfig,
        cache_path: str = "cache/inflation.pkl",
        max_age_seconds: int = 60,  # 3600,
        auto_refresh_async: bool = False,
    ):
        self._loader = loader
        self._config = config

        # -------------------------
        # MEMORY CACHE STATE
        # -------------------------
        self._cache: Optional[pd.Series] = None
        self._cache_timestamp: Optional[float] = None

        # -------------------------
        # CACHE CONFIG
        # -------------------------
        self._cache_path = cache_path
        self._max_age = max_age_seconds
        self._auto_refresh_async = auto_refresh_async

        # -------------------------
        # THREAD SAFETY
        # -------------------------
        self._lock = threading.Lock()

        # ensure cache directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # ============================================================
    # PUBLIC API
    # ============================================================
    def get_inflation_rates(
        self,
        logger: Optional[logging.Logger] = None,
        force_refresh: bool = False,
    ) -> pd.Series:
        """
        Retrieve inflation time series with hybrid caching.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used for debug/info tracing.
        force_refresh : bool, default False
            If True, bypasses memory and disk cache and reloads from API.

        Returns
        -------
        pd.Series
            Inflation time series indexed by date.

        Cache resolution order
        ----------------------
        If force_refresh=False:
            1. Memory cache (if valid)
            2. Disk cache (if valid)
            3. API loader

        If force_refresh=True:
            → API loader only
        """

        # ========================================================
        # 0. FORCE REFRESH (bypass all caches)
        # ========================================================
        if force_refresh:
            if logger:
                logger.info("InflationService: force refresh → API call")

            data = self._load_from_api(logger)
            self._update_cache(data)
            self._save_to_disk(data, logger)
            return data

        # ========================================================
        # 1. MEMORY CACHE FAST PATH
        # ========================================================
        if self._cache is not None and not self._is_stale():
            if logger:
                logger.debug("InflationService: memory cache hit")
            return self._cache

        # ========================================================
        # 2. THREAD-SAFE INITIALIZATION / REFRESH
        # ========================================================
        with self._lock:

            # double-check after lock
            if self._cache is not None and not self._is_stale():
                return self._cache

            if logger:
                logger.info("InflationService: cache miss → resolving source")

            # ====================================================
            # 3. DISK CACHE
            # ====================================================
            disk_data = self._load_from_disk(logger)

            if disk_data is not None:
                self._update_cache(disk_data)

                if logger:
                    logger.info("InflationService: loaded from disk cache")

                # optional async refresh if stale
                if self._is_stale() and self._auto_refresh_async:
                    self._refresh_async(logger)

                return self._cache

            # ====================================================
            # 4. API LOAD (SOURCE OF TRUTH)
            # ====================================================
            data = self._load_from_api(logger)
            self._update_cache(data)
            self._save_to_disk(data, logger)

            if logger:
                logger.info("InflationService: cached (memory + disk)")

            return self._cache

    # ============================================================
    # CORE LOADER
    # ============================================================
    def _load_from_api(self, logger: Optional[logging.Logger]) -> pd.Series:
        """
        Call external loader (source of truth).
        """
        if logger:
            logger.info("InflationService: loading from API...")

        data = self._loader(self._config)

        if data is None or data.empty:
            raise ValueError("Inflation data is empty")

        return data

    # ============================================================
    # CACHE HELPERS
    # ============================================================
    def _update_cache(self, data: pd.Series) -> None:
        """
        Update in-memory cache.
        """
        self._cache = data
        self._cache_timestamp = time.time()

    def _is_stale(self) -> bool:
        """
        Check if memory cache is expired (TTL).
        """
        if self._cache_timestamp is None:
            return True
        return (time.time() - self._cache_timestamp) > self._max_age

    def _load_from_disk(self, logger: Optional[logging.Logger]) -> Optional[pd.Series]:
        """
        Load inflation data from disk cache safely.

        Returns None if:
        - cache is missing
        - cache is corrupted
        - cache is expired
        """
        try:
            data = load_cache(self._cache_path, max_age_seconds=self._max_age)

            if data is not None and logger:
                logger.debug("InflationService: disk cache hit")

            return data

        except Exception:
            return None

    def _save_to_disk(self, data: pd.Series, logger: Optional[logging.Logger]) -> None:
        """
        Persist inflation cache to disk (fail-safe).
        """
        try:
            save_cache(self._cache_path, data)

            if logger:
                logger.debug("InflationService: saved to disk cache")

        except Exception:
            pass

    # ============================================================
    # ASYNC REFRESH
    # ============================================================
    def _refresh_async(self, logger: Optional[logging.Logger]) -> None:
        """
        Background refresh of inflation cache (non-blocking).
        """

        def _job():
            try:
                if logger:
                    logger.info("InflationService: async refresh started")

                data = self._load_from_api(logger)

                self._update_cache(data)
                self._save_to_disk(data, logger)

                if logger:
                    logger.info("InflationService: async refresh completed")

            except Exception as e:
                if logger:
                    logger.warning(f"InflationService: async refresh failed: {e}")

        threading.Thread(target=_job, daemon=True).start()
