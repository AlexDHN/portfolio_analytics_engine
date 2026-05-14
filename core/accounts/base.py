from abc import ABC, abstractmethod
import pandas as pd
from typing import Type, Optional
from core.investments.base import Investment
from core.investments.listed import ListedInvestment


class InvestmentAccount(ABC):
    """
    Abstract base class representing an account composed of multiple investment instruments.
    """

    def __init__(
        self, transactions: pd.DataFrame, df_data_fs: Optional[pd.DataFrame] = None
    ):
        """
        Initialize an investment account from a multi-indexed DataFrame,
        where level 0 is the investment identifier and level 1 is typically a datetime.

        Parameters
        ----------
        transactions : pd.DataFrame
            A MultiIndexed DataFrame containing the transaction history
            for all underlying investments.
        """
        if (
            not isinstance(transactions.index, pd.MultiIndex)
            or transactions.index.nlevels < 2
        ):
            raise ValueError(
                "The DataFrame must be indexed with a MultiIndex (InvestmentName, Date)."
            )

        self.investments: pd.Series = self._build_investments(transactions, df_data_fs)
        self.investments_timeline: pd.DataFrame = self._investments_timeline()
        self.investments_closed_positions: pd.DataFrame = (
            self._investments_closed_positions()
        )

    @abstractmethod
    def _investment_class(self) -> Type[Investment]:
        """
        Return the Investment subclass to instantiate (e.g., FixedRateInvestment).
        Must be implemented by each subclass.
        """
        pass

    def _build_investments(
        self, transactions: pd.DataFrame, df_data_fs: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Construct a Series of investment instances from the transaction data.

        Parameters
        ----------
        transactions : pd.DataFrame
            MultiIndexed DataFrame with transactions grouped by investment name.
            Expected to contain at least identifiers for tickers and transaction details.
        df_data_fs : Optional[pd.DataFrame], optional
            DataFrame containing fractional share (rompus) values, indexed by ticker and date.
            If provided, it will be passed to investments that require fractional share allocation
            (e.g., when handling stock splits).

        Returns
        -------
        pd.Series
            Series of Investment objects indexed by investment name.
        """
        cls = self._investment_class()
        if cls == ListedInvestment:
            return transactions.groupby(level=0).apply(
                lambda g: cls(
                    g.name,
                    g.droplevel(0),
                    (
                        df_data_fs.loc[g.name]
                        if g.name in df_data_fs.index.unique(0)
                        else pd.DataFrame()
                    ),
                )
            )
        else:
            return transactions.groupby(level=0).apply(
                lambda g: cls(g.name, g.droplevel(0))
            )

    def _investments_timeline(self) -> pd.DataFrame:
        """
        Aggregate settled positions from all underlying investments.

        Returns
        -------
        pd.DataFrame
        """

        return pd.concat(
            self.investments.apply(lambda inv: inv.positions_timeline).to_dict(),
            axis=1,
        )

    def _investments_closed_positions(self) -> pd.DataFrame:
        """
        Aggregate settled positions from all underlying investments.

        Returns
        -------
        pd.DataFrame
        """

        dfs = [
            inv.closed_positions
            for inv in self.investments
            if hasattr(inv, "closed_positions") and not inv.closed_positions.empty
        ]
        return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()
