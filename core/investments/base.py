from abc import ABC, abstractmethod
import pandas as pd
import logging


class Investment(ABC):
    """
    Abstract base class for all types of financial investments.

    Defines the minimal interface and common attributes for concrete investment
    classes such as listed stocks, ETFs, or fixed-rate savings accounts.

    Attributes
    ----------
    name : str
        Investment identifier (ticker, account type, etc.).
    transactions : pd.DataFrame
        DataFrame storing the transaction history for this investment.
    positions_timeline : pd.DataFrame
        Time-indexed DataFrame capturing the evolution of positions, values, and interest/fees.
    closed_positions : pd.DataFrame | pd.Series
        Final processed positions after all adjustments.
    logger : logging.Logger
        Logger instance for the investment, available for subclasses.
    """

    def __init__(self, name: str = "Unnamed"):
        self.name: str = name
        self.transactions: pd.DataFrame = pd.DataFrame()
        self.positions_timeline: pd.DataFrame = pd.DataFrame()
        self.closed_positions: pd.DataFrame | pd.Series = pd.DataFrame()
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def _prepare_empty_positions_timeline(self) -> pd.DataFrame:
        """
        Prepare an empty investment timeline DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by relevant dates or intervals with columns
            for transaction values, fees, interests, etc.

        Notes
        -----
        - Must be implemented by all concrete subclasses.
        - Defines the structure for `_process_positions_timeline`.
        """
        pass

    @abstractmethod
    def _process_positions_timeline(self) -> list[pd.DataFrame]:
        pass
