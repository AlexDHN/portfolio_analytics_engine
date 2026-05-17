from typing import Optional

from config.settings import (
    InflationConfig,
    MIN_DATE,
    KEY_INFLATION,
    BDF_BASE_URL,
    BDF_HEADERS,
)

from loaders.inflation_loader import load_inflation_series
from services.inflation_service import InflationService

# ============================================================
# SINGLETON STATE (APPLICATION SCOPE)
# ============================================================
_inflation_service: Optional[InflationService] = None


def get_inflation_service() -> InflationService:
    """
    InflationService Factory (Singleton Provider)
    ============================================

    Overview
    --------
    This factory is responsible for creating and providing a **singleton**
    instance of `InflationService` across the entire application.

    It ensures that:
        - only one service instance exists (process-wide singleton)
        - configuration is centralized
        - dependencies are properly wired
        - inflation data is not redundantly loaded

    Architecture role
    ------------------
    This module sits at the **composition root** of the application:

        Config → Loader → Service → Application

    Responsibilities
    ----------------
    - Build InflationConfig from global settings
    - Inject dependencies into InflationService
    - Guarantee singleton lifecycle
    - Provide global access point for inflation data service

    Non-responsibilities
    --------------------
    This factory MUST NOT:
    - perform caching itself (handled by InflationService)
    - call external APIs directly
    - transform inflation data
    - manage threading or concurrency

    Singleton behavior
    ------------------
    - First call initializes the service
    - Subsequent calls return the same instance
    - State is shared across the application process

    Returns
    -------
    InflationService
        Singleton instance of the inflation service.

    Notes
    -----
    This pattern is safe in:
        - backtesting engines
        - portfolio simulators
        - multi-threaded investment pipelines

    but assumes a single-process runtime context.
    """

    global _inflation_service

    # ------------------------------------------------------------
    # FAST PATH (already initialized)
    # ------------------------------------------------------------
    if _inflation_service is not None:
        return _inflation_service

    # ------------------------------------------------------------
    # BUILD CONFIGURATION OBJECT
    # ------------------------------------------------------------
    config = InflationConfig(
        key=KEY_INFLATION,
        base_url=BDF_BASE_URL,
        headers=BDF_HEADERS,
        start_date=MIN_DATE,
    )

    # ------------------------------------------------------------
    # SERVICE INITIALIZATION (dependency wiring)
    # ------------------------------------------------------------
    _inflation_service = InflationService(
        loader=load_inflation_series,
        config=config,
    )

    return _inflation_service
