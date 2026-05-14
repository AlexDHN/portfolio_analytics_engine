# core/accounts/listed_account.py

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict
from core.accounts.base_account import InvestmentAccount
from core.investments.listed_v2 import ListedInvestment
from core.market_data.manager_v3 import MarketDataManager
from utils.logger import get_logger


class ListedAccount(InvestmentAccount):
    """
    Concrete implementation of InvestmentAccount for listed instruments
    (e.g., PEA, CTO, or other brokerage accounts).

    This class handles the preprocessing, initialization, and parallel
    construction of `ListedInvestment` instances. It also manages integration
    with a shared `MarketDataManager` to efficiently fetch and cache price data
    across multiple tickers.

    The design supports large portfolios by leveraging parallel execution for
    investment instantiation, ensuring scalable performance while maintaining
    data integrity and reusability of market data resources.

    Parameters
    ----------
    transactions : pd.DataFrame
        MultiIndexed DataFrame containing raw transaction data for all listed
        instruments. The index must be structured as:
            - level 0 : Ticker or instrument identifier
            - level 1 : Transaction date (datetime or string)

    stock_data_fs : pd.DataFrame
        Fractional share (rompus) reference data indexed by ticker and date.

    max_workers : Optional[int], default=None
        Maximum number of worker threads to use for parallel investment
        initialization. If None, defaults to the system’s available CPU cores.

    Raises
    ------
    ValueError
        If required transaction columns are missing or data is malformed.

    Attributes
    ----------
    logger : logging.Logger
        Class-specific logger for diagnostics and lifecycle tracing.

    stock_data_fs : pd.DataFrame
        Fractional share reference data used for handling stock splits and
        non-integer share quantities.

    market_data_manager : MarketDataManager
        Shared data manager instance used to retrieve and cache market data
        for all tickers referenced in the account.

    tickers : pd.Index
        Unique list of tickers (investment identifiers) detected from
        transaction data.

    Notes
    -----
    - Each investment is initialized as a `ListedInvestment`, receiving both
      fractional share data and a shared `MarketDataManager` instance.
    - Parallelization via ThreadPoolExecutor significantly reduces
      initialization time for large multi-asset portfolios.
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
        self.logger = get_logger(self.__class__.__name__)
        self.max_workers = max_workers
        self.stock_data_fs = stock_data_fs

        # Preprocess and validate transactions
        df = self._preprocess_transactions(transactions)

        # Detect all relevant tickers
        self.tickers = df.index.get_level_values(0).unique()
        self.logger.debug("Detected tickers: %s", list(self.tickers))

        all_tickers = self.tickers.union(df["indice"].unique())

        # Initialize MarketDataManager once for all tickers
        self.market_data_manager = MarketDataManager(
            tickers=list(all_tickers), force_refresh=force_refresh
        )
        self.logger.info(
            "Successfully initialized MarketDataManager with %d tickers",
            len(all_tickers),
        )

        # Initialize via abstract parent
        super().__init__(df)

    # -------------------------------------------------------------------------
    # Data Preprocessing
    # -------------------------------------------------------------------------
    def _preprocess_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw transaction DataFrame for ListedInvestment compatibility.

        Steps:
        - Validate required columns exist
        - Fill missing values for fees
        - Initialize adjusted and tracking columns
        - Add an empty date placeholder
        - Sort and return a clean DataFrame

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
            If required columns are missing in the input DataFrame.
        """
        # === 1. Validate required columns ===
        missing_cols = set(self.REQUIRED_COLUMNS) - set(transactions.columns)
        if missing_cols:
            self.logger.error("Missing required columns: %s", missing_cols)
            raise ValueError(
                f"Missing required columns in transactions DataFrame: {missing_cols}"
            )
        self.logger.debug("All required columns are present")

        # === 2. Sort by index and make a copy to avoid mutating input ===
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
    # Parallel Investment Initialization
    # -------------------------------------------------------------------------
    def _build_investments_parallel(self) -> pd.Series:
        """
        Construct all ListedInvestment instances in parallel using thread workers.

        Each worker initializes a `ListedInvestment` with:
        - Its corresponding transaction subset
        - Fractional share data (if available)
        - The shared `MarketDataManager` instance for market data access

        Returns
        -------
        pd.Series
            Series of successfully initialized `ListedInvestment` instances,
            indexed by ticker symbol.
        """
        investments: Dict[str, ListedInvestment] = {}

        self.logger.info(
            "Starting parallel construction of ListedInvestment objects for %d tickers with %d workers",
            len(self.transactions.index.get_level_values(0).unique()),
            self.max_workers,
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

        self.logger.info("Successfully built %d listed investments", len(investments))
        return pd.Series(investments)
