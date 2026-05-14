# core/accounts/base_account.py

from abc import ABC, abstractmethod
import pandas as pd
import os


class InvestmentAccount(ABC):
    """
    Abstract base class representing a portfolio account composed of multiple
    individual investment instruments.

    This class provides the core orchestration logic for initializing, aggregating,
    and analyzing multiple investment objects derived from transactional data.
    Subclasses are responsible for implementing their own *parallel investment
    build logic* as well as specifying the concrete `Investment` subclass used.

    The design emphasizes performance and scalability by allowing parallel
    construction of investment instances (e.g., via ThreadPoolExecutor),
    which can significantly reduce initialization time for large portfolios.

    Parameters
    ----------
    transactions : pd.DataFrame
        MultiIndexed DataFrame containing all transactions related to the account.
        The index must have at least two levels:
            - level 0 : Investment identifier (e.g., ticker, product name)
            - level 1 : Date or datetime of each transaction.

    Raises
    ------
    ValueError
        If the transaction DataFrame does not have a MultiIndex with at least
        two levels.

    Attributes
    ----------
    transactions : pd.DataFrame
        The preprocessed transaction data for the entire account.
    max_workers : int
        Number of CPU workers used for parallel investment building, determined
        automatically from the system’s available cores.
    investments : pd.Series
        Series of instantiated `Investment` objects, indexed by investment name.
        Each instance encapsulates the computation logic and data for one instrument.
    investments_timeline : pd.DataFrame
        Aggregated timeline across all investments, consolidating their positions
        or valuations over time.
    investments_closed_positions : pd.DataFrame
        Combined dataset of closed or realized investment positions across
        all underlying instruments.

    Notes
    -----
    - Subclasses must implement:
        * `_build_investments_parallel()`: to define how investments are
          constructed (possibly in parallel).
    - This class assumes that each investment class exposes the following attributes:
        * `positions_timeline` : pd.DataFrame
        * `closed_positions` : pd.DataFrame
    """

    def __init__(self, transactions: pd.DataFrame):
        if (
            not isinstance(transactions.index, pd.MultiIndex)
            or transactions.index.nlevels < 2
        ):
            raise ValueError(
                "The DataFrame must be indexed with a MultiIndex (InvestmentName, Date)."
            )

        self.max_workers = os.cpu_count()

        self.transactions = transactions
        self.investments = self._build_investments_parallel()
        self.investments_timeline = self._investments_timeline()
        self.investments_closed_positions = self._investments_closed_positions()

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def _build_investments_parallel(self) -> pd.Series:
        """
        Construct and initialize Investment objects, potentially in parallel.

        Must be implemented by subclasses, typically using a ThreadPoolExecutor
        or ProcessPoolExecutor to parallelize the creation of investment instances
        when dealing with large portfolios.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Aggregation Helpers
    # -------------------------------------------------------------------------
    def _investments_timeline(self) -> pd.DataFrame:
        """
        Aggregate the position timelines across all investments.

        Returns
        -------
        pd.DataFrame
            Concatenated positions timeline of all investments, indexed by date.
        """
        return pd.concat(
            self.investments.apply(lambda inv: inv.positions_timeline).to_dict(),
            axis=1,
        )

    def _investments_closed_positions(self) -> pd.DataFrame:
        """
        Aggregate all closed positions from the underlying investments.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame of all realized (closed) investment positions.
            Returns an empty DataFrame if no closed positions are found.
        """
        dfs = [
            inv.closed_positions
            for inv in self.investments
            if hasattr(inv, "closed_positions") and not inv.closed_positions.empty
        ]
        return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()
