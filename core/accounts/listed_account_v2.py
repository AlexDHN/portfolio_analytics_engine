# core/accounts/listed_account.py

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict
from core.accounts.base_account import InvestmentAccount
from core.investments.listed_v2 import ListedInvestment
from core.market_data.manager_v3 import MarketDataManager
from factories.inflation_factory import get_inflation_service
from utils.logger import get_logger


class ListedAccount(InvestmentAccount):
    """
    ListedAccount
    =============

    Concrete implementation of InvestmentAccount for listed instruments
    (e.g., PEA, CTO, or other brokerage accounts).

    Overview
    --------
    This class orchestrates the full initialization pipeline for a listed
    investment account:

    - Transaction preprocessing and validation
    - Shared resource initialization (MarketDataManager, inflation rates)
    - Parallel construction of ListedInvestment instances

    It acts as the **composition root** for all listed investments in the
    account, resolving shared dependencies once and injecting them into
    each investment — avoiding redundant loading across parallel workers.

    Design principles
    -----------------
    - Shared dependencies (MarketDataManager, inflation rates) are resolved
      once at the account level and injected into each ListedInvestment.
    - Parallel instantiation via ThreadPoolExecutor ensures scalable
      performance for large multi-asset portfolios.
    - Thread-safe by design: all shared objects are read-only after
      initialization and safe to pass across worker threads.

    Parameters
    ----------
    transactions : pd.DataFrame
        MultiIndexed DataFrame containing raw transaction data for all listed
        instruments. The index must be structured as:
            - level 0 : Ticker or instrument identifier
            - level 1 : Transaction date (datetime or string)
    stock_data_fs : pd.DataFrame
        Fractional share (rompus) reference data indexed by ticker and date.
        Used during corporate action processing (e.g., stock splits).
    max_workers : Optional[int], default=None
        Maximum number of worker threads for parallel investment initialization.
        If None, defaults to the number of available CPU cores.
    force_refresh : bool, default=False
        If True, bypasses disk cache and forces a fresh download of market
        data via MarketDataManager.

    Raises
    ------
    ValueError
        If required transaction columns are missing or data is malformed.

    Attributes
    ----------
    logger : logging.Logger
        Class-specific logger for diagnostics and lifecycle tracing.
    stock_data_fs : pd.DataFrame
        Fractional share reference data shared across all investments.
    market_data_manager : MarketDataManager
        Shared market data manager initialized once for all tickers in
        the account. Provides OHLCV and corporate action data.
    inflation_rates : pd.Series
        Inflation time series resolved once via InflationService and
        injected into each ListedInvestment. Indexed by date.
    tickers : pd.Index
        Unique list of ticker symbols detected from transaction data.

    Notes
    -----
    Shared dependency resolution order in __init__:
        1. Preprocess and validate transactions
        2. Initialize MarketDataManager (market data, once for all tickers)
        3. Resolve inflation rates (once via InflationService singleton)
        4. Build all ListedInvestment instances in parallel (injecting 2 & 3)
    """

    REQUIRED_COLUMNS = [
        "value",
        "count",
        "tax",
        "abondement",
        "indice",
        "broker_fees",
        "other_fees",
    ]

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(
        self,
        transactions: pd.DataFrame,
        stock_data_fs: pd.DataFrame,
        max_workers: Optional[int] = None,
        force_refresh: bool = False,
    ):
        """
        Initialize the ListedAccount and all its ListedInvestment instances.

        Shared dependencies are resolved once at this level and injected
        into each investment, avoiding redundant loading across parallel
        worker threads.

        Parameters
        ----------
        transactions : pd.DataFrame
            Raw MultiIndexed transaction data (level 0: ticker, level 1: date).
        stock_data_fs : pd.DataFrame
            Fractional share reference data indexed by ticker and date.
        max_workers : Optional[int], default=None
            Number of parallel threads for investment construction.
            Defaults to available CPU cores if None.
        force_refresh : bool, default=False
            Forces market data refresh, bypassing disk cache.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.max_workers = max_workers
        self.stock_data_fs = stock_data_fs

        # --- 1. Preprocess and validate transactions ---
        df = self._preprocess_transactions(transactions)

        # --- 2. Detect all relevant tickers ---
        self.tickers = df.index.get_level_values(0).unique()
        self.logger.debug("Detected tickers: %s", list(self.tickers))

        all_tickers = self.tickers.union(df["indice"].unique())

        # --- 3. Initialize MarketDataManager once for all tickers ---
        self.market_data_manager = MarketDataManager(
            tickers=list(all_tickers), force_refresh=force_refresh
        )
        self.logger.info(
            "Successfully initialized MarketDataManager with %d tickers",
            len(all_tickers),
        )

        # --- 4. Resolve inflation rates once for all investments ---
        # Resolved here at the account level to avoid N redundant calls
        # across parallel ListedInvestment workers. Injected as a read-only
        # pd.Series — safe to share across threads.
        self.inflation_rates = get_inflation_service().get_inflation_rates(self.logger)
        self.logger.info(
            "Inflation rates resolved for %d periods", len(self.inflation_rates)
        )

        # --- 5. Initialize via abstract parent (triggers _build_investments_parallel) ---
        super().__init__(df)

    # -------------------------------------------------------------------------
    # Data Preprocessing
    # -------------------------------------------------------------------------
    def _preprocess_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess and validate raw transaction data for ListedInvestment
        compatibility.

        Steps
        -----
        1. Validate that all required columns are present.
        2. Sort by index and copy to avoid mutating the input.
        3. Fill missing fee values with 0.0.
        4. Initialize adjusted and tracking columns.
        5. Add an empty date placeholder column.

        Parameters
        ----------
        transactions : pd.DataFrame
            Raw transactions DataFrame to preprocess.

        Returns
        -------
        pd.DataFrame
            Preprocessed transactions ready for InvestmentAccount consumption.

        Raises
        ------
        ValueError
            If one or more required columns are missing from the input.
        """
        # === 1. Validate required columns ===
        missing_cols = set(self.REQUIRED_COLUMNS) - set(transactions.columns)
        if missing_cols:
            self.logger.error("Missing required columns: %s", missing_cols)
            raise ValueError(
                f"Missing required columns in transactions DataFrame: {missing_cols}"
            )
        self.logger.debug("All required columns are present")

        # === 2. Sort by index and copy to avoid mutating input ===
        df = transactions.sort_index().copy()
        self.logger.debug("Transactions sorted by index")

        # === 3. Fill missing values for fee columns ===
        fee_cols = ["other_fees", "broker_fees", "tax"]
        df[fee_cols] = df[fee_cols].fillna(0.0)

        # === 4. Initialize adjusted and tracking columns ===
        df = df.assign(
            count_nm=df["count"],
            count_adj=df["count"],
            value_adj=df["value"],
            tax_adj=df["tax"],
            abond_adj=df["abondement"],
            fees_adj=df["broker_fees"] + df["other_fees"],
            fs_adj=0.0,  # Fractional shares initialized to zero
            empty_date=pd.NaT,  # Placeholder for future date values
        )

        # === 5. Log result and return ===
        self.logger.info("Preprocessing completed: %d rows prepared", len(df))
        return df

    # -------------------------------------------------------------------------
    # Parallel Investment Construction
    # -------------------------------------------------------------------------
    def _build_investments_parallel(self) -> pd.Series:
        """
        Construct all ListedInvestment instances in parallel using thread workers.

        Each worker receives:
        - Its corresponding transaction subset (ticker-scoped)
        - Fractional share data for that ticker (empty DataFrame if absent)
        - The shared MarketDataManager (read-only, thread-safe)
        - The pre-resolved inflation rates (read-only pd.Series, thread-safe)

        Shared dependencies are injected rather than resolved per-worker,
        ensuring a single load per account initialization regardless of the
        number of tickers.

        Returns
        -------
        pd.Series
            Series of successfully initialized ListedInvestment instances,
            indexed by ticker symbol. Failed initializations are logged and
            excluded from the result.
        """
        investments: Dict[str, ListedInvestment] = {}

        self.logger.info(
            "Starting parallel construction of %d ListedInvestment objects "
            "with %s workers",
            len(self.transactions.index.get_level_values(0).unique()),
            self.max_workers or "default",
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    ListedInvestment,
                    name,
                    group.droplevel(0),
                    (
                        self.stock_data_fs.loc[name]
                        if name in self.stock_data_fs.index.unique(0)
                        else pd.DataFrame()
                    ),
                    self.market_data_manager,
                    self.inflation_rates,  # injected once, shared across all workers
                ): name
                for name, group in self.transactions.groupby(level=0)
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    investments[name] = future.result()
                    self.logger.debug("Successfully initialized investment: %s", name)
                except Exception as e:
                    self.logger.error("Failed to initialize %s: %s", name, e)

        self.logger.info(
            "Successfully built %d / %d listed investments",
            len(investments),
            len(self.tickers),
        )
        return pd.Series(investments)
