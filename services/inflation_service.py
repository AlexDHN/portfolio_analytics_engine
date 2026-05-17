import os
import threading
import logging
from typing import Callable, Optional

import pandas as pd

from config.settings import InflationConfig
from services.cache import (
    load_cache,
    save_cache,
    is_cache_expired,
    is_cache_nearly_expired,
)


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
        1. Memory cache — valid if is_cache_expired() confirms disk is fresh
        2. Disk cache   — loaded via load_cache() (handles TTL + corruption)
        3. API loader   — source of truth, persisted to disk + memory

    If force_refresh=True:
        → bypass ALL caches and call API directly

    Cache utility responsibilities
    ------------------------------
    - is_cache_expired()        : lightweight TTL check (mtime only), used
                                  for fast path validation without deserializing
    - is_cache_nearly_expired() : proactive refresh trigger at 80% TTL,
                                  used to warm cache before expiration
    - load_cache()              : full load (TTL check + deserialization),
                                  used once at disk cache resolution step
    - save_cache()              : persistence to disk

    Notes
    -----
    Inflation data is assumed to be updated periodically (not real-time).
    The single source of truth for TTL validity is the disk file mtime,
    accessed via is_cache_expired() and load_cache() from services.cache.
    """

    def __init__(
        self,
        loader: Callable[[InflationConfig], pd.Series],
        config: InflationConfig,
        cache_path: str = "cache/inflation.pkl",
        max_age_seconds: int = 3600,
        auto_refresh_async: bool = False,
    ):
        """
        Initialize the InflationService.

        Parameters
        ----------
        loader : Callable[[InflationConfig], pd.Series]
            External function responsible for fetching inflation data
            from a remote source (API, database, etc.).
        config : InflationConfig
            Configuration object passed to the loader.
        cache_path : str, default "cache/inflation.pkl"
            Path to the disk cache file.
        max_age_seconds : int, default 3600
            Maximum age of the cache in seconds before it is considered
            stale. Applies to both memory and disk cache layers via
            is_cache_expired() and load_cache().
        auto_refresh_async : bool, default False
            If True, triggers a background thread to refresh the cache
            when the disk cache exceeds 80% of its TTL, without blocking
            the caller.
        """
        self._loader = loader
        self._config = config

        # -------------------------
        # MEMORY CACHE STATE
        # -------------------------
        self._cache: Optional[pd.Series] = None

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
            1. Memory cache — valid if is_cache_expired() confirms disk
               is still fresh. Lightweight check, no deserialization.
            2. Disk cache   — loaded via load_cache() which handles TTL
               check and deserialization in a single call.
            3. API loader   — source of truth, result persisted to both
               disk and memory cache.

        If force_refresh=True:
            → API loader only, result persisted to disk + memory.
        """

        # ========================================================
        # 0. FORCE REFRESH (bypass all caches)
        # ========================================================
        if force_refresh:
            if logger:
                logger.info("InflationService: force refresh → API call")

            data = self._load_from_api(logger)
            self._cache = data
            self._save_to_disk(data, logger)
            return data

        # ========================================================
        # 1. MEMORY CACHE FAST PATH
        # ========================================================
        # is_cache_expired() provides a lightweight mtime-only check —
        # no deserialization. load_cache() is reserved for the actual
        # disk load at step 3.
        if self._cache is not None and not is_cache_expired(
            self._cache_path, self._max_age
        ):
            if logger:
                logger.debug("InflationService: memory cache hit")

            # Proactive async refresh if nearing expiration (> 80% TTL)
            if self._auto_refresh_async and is_cache_nearly_expired(
                self._cache_path, self._max_age
            ):
                self._refresh_async(logger)

            return self._cache

        # ========================================================
        # 2. THREAD-SAFE INITIALIZATION / REFRESH
        # ========================================================
        with self._lock:

            # double-check after lock acquisition
            if self._cache is not None and not is_cache_expired(
                self._cache_path, self._max_age
            ):
                return self._cache

            if logger:
                logger.info("InflationService: cache miss → resolving source")

            # ====================================================
            # 3. DISK CACHE
            # load_cache() handles TTL check + deserialization.
            # Returns None if missing, expired, or corrupted.
            # ====================================================
            disk_data = self._load_from_disk(logger)

            if disk_data is not None:
                self._cache = disk_data

                if logger:
                    logger.info("InflationService: loaded from disk cache")

                return self._cache

            # ====================================================
            # 4. API LOAD (SOURCE OF TRUTH)
            # ====================================================
            data = self._load_from_api(logger)
            self._cache = data
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

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used for tracing.

        Returns
        -------
        pd.Series
            Raw inflation time series from the remote source.

        Raises
        ------
        ValueError
            If the loader returns None or an empty Series.
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
    def _load_from_disk(self, logger: Optional[logging.Logger]) -> Optional[pd.Series]:
        """
        Load inflation data from disk cache via load_cache().

        Delegates all TTL and validity checks to load_cache(), which
        internally calls is_cache_expired() based on file mtime, then
        deserializes the object if valid.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used for tracing.

        Returns
        -------
        pd.Series or None
            Cached data if valid, None if missing, expired or corrupted.
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
        Persist inflation data to disk cache (fail-safe).

        Parameters
        ----------
        data : pd.Series
            Inflation time series to persist.
        logger : logging.Logger, optional
            Logger used for tracing.
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
        Trigger a background refresh of the inflation cache (non-blocking).

        Spawns a daemon thread that reloads data from the API and updates
        both memory and disk cache. Invoked proactively when the disk cache
        exceeds 80% of its TTL, so the cache is warmed before expiration.
        Failures are logged but never propagate to the caller.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used for tracing.
        """

        def _job():
            try:
                if logger:
                    logger.info("InflationService: async refresh started")

                data = self._load_from_api(logger)
                self._cache = data
                self._save_to_disk(data, logger)

                if logger:
                    logger.info("InflationService: async refresh completed")

            except Exception as e:
                if logger:
                    logger.warning(f"InflationService: async refresh failed: {e}")

        threading.Thread(target=_job, daemon=True).start()
