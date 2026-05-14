# core/accounts/regulated_saving_account.py

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict
from core.accounts.base_account import InvestmentAccount
from core.investments.fixed_rate import FixedRateInvestment
from utils.logger import get_logger


class RegulatedSavingAccount(InvestmentAccount):
    """
    Concrete implementation of InvestmentAccount for regulated savings products
    (e.g., Livret A, LDDS).

    This class handles preprocessing of transactions specific to regulated
    savings accounts and parallel construction of `FixedRateInvestment`
    instances.

    Parameters
    ----------
    transactions : pd.DataFrame
        MultiIndexed DataFrame containing all transactions related to the account.
        The index must have at least two levels:
            - level 0 : Investment identifier (e.g., product name)
            - level 1 : Transaction date

    max_workers : Optional[int], default=None
        Maximum number of worker threads to use for parallel investment
        initialization. If None, defaults to the system’s available CPU cores.

    Attributes
    ----------
    logger : logging.Logger
        Class-specific logger for diagnostics and lifecycle tracing.

    max_workers : int
        Number of threads used for parallel investment construction.

    Notes
    -----
    - Each investment is initialized as a `FixedRateInvestment`.
    - Parallelization ensures scalable performance for accounts with
      multiple savings products.
    """

    def __init__(self, transactions: pd.DataFrame, max_workers: Optional[int] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.max_workers = max_workers

        # Preprocess transactions: keep necessary columns and initialize adjusted values
        df = transactions[["value", "empty_date", "indice"]].sort_index().copy()
        df["value_adj"] = df["value"]
        df[["rl_interest_adj", "th_interest_adj"]] = 0.0

        self.logger.debug(
            "Preprocessed %d transactions for regulated savings account", len(df)
        )

        super().__init__(df)

    # -------------------------------------------------------------------------
    # Parallel Investment Initialization
    # -------------------------------------------------------------------------
    def _build_investments_parallel(self) -> pd.Series:
        """
        Construct all FixedRateInvestment instances in parallel using ThreadPoolExecutor.

        Returns
        -------
        pd.Series
            Series of successfully initialized `FixedRateInvestment` instances,
            indexed by product name.
        """
        investments: Dict[str, FixedRateInvestment] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(FixedRateInvestment, name, group.droplevel(0)): name
                for name, group in self.transactions.groupby(level=0)
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    investments[name] = future.result()
                    self.logger.debug("Successfully initialized investment: %s", name)
                except Exception as e:
                    self.logger.error("Failed to initialize investment %s: %s", name, e)

        self.logger.info(
            "Successfully built %d regulated savings investments", len(investments)
        )
        return pd.Series(investments)
