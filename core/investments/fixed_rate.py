# core/investments/fixed_rate.py

from core.investments.base import Investment
import logging
import pandas as pd
import numpy as np
from collections import deque
from typing import ClassVar
from config.settings import TODAY, MIN_DATE, KEY_LIVRET_A, BDF_BASE_URL, BDF_HEADERS
from loaders.bdf_api import get_bdf_series
from utils.logger import get_logger
from utils.math_utils import smart_rounding_banking


class FixedRateInvestment(Investment):
    """
    Represents a fixed-rate investment, such as a French Livret savings account.

    This class manages deposits and withdrawals, calculates accrued interest
    (both realized and theoretical), handles short-value corrections (negative
    withdrawals exceeding balances), and builds a timeline of investment states
    over time.

    Class Attributes
    ----------------
    _interest_rates : pd.Series
        Cached time series of applicable interest rates for the investment.
        Indexed by interval periods (bimensual).

    Instance Attributes
    -------------------
    name : str
        Name of the investment (e.g., "Livret A").
    transactions : pd.DataFrame
        Transaction history (deposits and withdrawals) including columns:
        'value_adj', 'rl_interest_adj', 'th_interest_adj', 'empty_date', 'indice'.
    interest_rates : pd.Series
        Time-indexed interest rates used to calculate accrued interest.
    positions_timeline : pd.DataFrame
        Interval-based timeline showing transaction values and interest accrual
        over time.
    closed_positions : pd.Series
        Final invested positions adjusted for accrued interest.
    logger : logging.Logger
        Logger instance for this investment.
    """

    _interest_rates: ClassVar[pd.Series] = None

    def __init__(self, name: str, transactions: pd.DataFrame):
        self.logger = get_logger(self.__class__.__name__)
        self.name = name
        self.transactions = transactions
        self.logger.info("Initializing fixed-rate investment for %s", self.name)

        self.interest_rates = self.__class__._get_interest_rates(logger=self.logger)
        self.positions_timeline = self._prepare_empty_positions_timeline()
        self.closed_positions = self._process_positions_timeline()

    @classmethod
    def _get_interest_rates(cls, logger: logging.Logger = None) -> pd.Series:
        """
        Load and cache the interest rate series for the investment from Banque de France.

        The method retrieves historical interest rates for the Livret product,
        splits them into bimensual intervals, and caches the result as a class attribute.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance to record progress and errors.

        Returns
        -------
        pd.Series
            Series of interest rates indexed by bimensual intervals.
            Rates are expressed as daily fractions for application to transaction values.

        Raises
        ------
        Exception
            If data retrieval or processing fails.

        Notes
        -----
        - Cached rates are reused for all instances of this class.
        - Adjusts intervals to start and end within valid historical dates.
        """
        if cls._interest_rates is None:
            if logger:
                logger.info("Loading interest rates from Banque de France…")

            try:
                start_date = MIN_DATE - pd.DateOffset(months=1)
                end_date = TODAY

                monthly_start = pd.date_range(start=start_date, end=end_date, freq="MS")
                fifteenth = monthly_start + pd.DateOffset(days=14)
                last = monthly_start + pd.offsets.MonthEnd(0)

                start_dates = pd.concat(
                    [
                        pd.Series(monthly_start),
                        pd.Series(fifteenth + pd.Timedelta(days=1)),
                    ]
                )
                end_dates = pd.concat([pd.Series(fifteenth), pd.Series(last)])

                interval_index = pd.IntervalIndex.from_arrays(
                    start_dates.sort_values(), end_dates.sort_values(), closed="both"
                )

                if interval_index[0].right < MIN_DATE:
                    interval_index = interval_index[1:]
                if interval_index[-1].left > TODAY:
                    interval_index = interval_index[:-1]

                df_la = get_bdf_series(
                    KEY_LIVRET_A,
                    BDF_BASE_URL,
                    BDF_HEADERS,
                    start_date=interval_index.left[0],
                )

                df_la.index = df_la.index.to_period("M").to_timestamp()

                interest_rates = (
                    pd.DataFrame(
                        {
                            "interval_index": interval_index,
                            "month": interval_index.left.to_period("M").to_timestamp(),
                        }
                    )
                    .merge(
                        df_la.rename("interest"),
                        left_on="month",
                        right_index=True,
                        how="left",
                    )
                    .set_index("interval_index")["interest"]
                    / 100
                    / 24
                )

                cls._interest_rates = interest_rates
                if logger:
                    logger.info("Interest rates successfully loaded and cached.")

            except Exception as e:
                if logger:
                    logger.exception("Error while loading interest rates: %s", str(e))
                raise
        else:
            if logger:
                logger.info("Using cached interest rates.")

        return cls._interest_rates

    def _prepare_empty_positions_timeline(self) -> pd.DataFrame:
        """
        Initialize an empty interval-based DataFrame for tracking investments and interest.

        The DataFrame is indexed by bimensual intervals and has columns corresponding
        to individual positions (transactions) and metrics: 'value_adj', 'rl_interest_adj',
        'th_interest_adj'.

        Returns
        -------
        pd.DataFrame
            Empty timeline DataFrame, ready for value and interest propagation.
        """
        # Colonnes extraites des transactions filtrées sur 'value_adj' positive
        cols = (
            self.transactions[self.transactions["value_adj"] > 0][
                ["value_adj", "rl_interest_adj", "th_interest_adj"]
            ]
            .T.unstack()
            .index
        )

        df_positions_timeline = pd.DataFrame(
            columns=cols, index=self.interest_rates.index
        )

        return df_positions_timeline

    @staticmethod
    def _reconcile_short_positions(
        transactions: pd.DataFrame, name: str
    ) -> pd.DataFrame:
        """
        Reconcile negative (short) value positions using FIFO logic against prior positive ones.

        For each negative `value_adj` row in the transaction history, this function matches it
        with prior positive positions in FIFO order. Interests (`rl_interest_adj`, `th_interest_adj`)
        are reallocated proportionally. Positions that are fully offset are marked with an
        `empty_date`.

        Parameters
        ----------
        transactions : pd.DataFrame
            A transaction history DataFrame, indexed by date, with the following columns:
            - 'value_adj' : float
            - 'rl_interest_adj' : float
            - 'th_interest_adj' : float
            - 'empty_date' : datetime64[ns] (can be NaT)
            - 'indice' : int (or similar identifier)

        typ : str
            Type of the Livret active_transactions_posount (e.g., 'Livret A', 'LDDS', etc.). Used for indexing the result.

        ticker : str
            Identifier of the Livret instrument. Also used in the result's index.

        Returns
        -------
        pd.DataFrame
            A DataFrame of compensation transactions, indexed by a MultiIndex (type, ticker, date),
            with columns:
            - 'value'
            - 'rl_interest_adj'
            - 'th_interest_adj'
            - 'short_date'
            - 'value_short'
            - 'indice'

        Raises
        ------
        ValueError
            If a negative value cannot be offset due to insufficient available positive positions.

        Notes
        -----
        - Mutates the `transactions` DataFrame in-place by updating 'value_adj',
        'rl_interest_adj', 'th_interest_adj', and 'empty_date'.
        - Matching logic is strictly FIFO (first-in, first-out).
        """

        # Extraction des colonnes nécessaires
        value_adj = transactions["value_adj"].to_numpy()
        rl_interest = transactions["rl_interest_adj"].to_numpy()
        th_interest = transactions["th_interest_adj"].to_numpy()
        value = (
            value_adj + rl_interest + th_interest
        )  # transactions[['value_adj', 'rl_interest_adj', 'th_interest_adj']].sum(axis=1).to_numpy()
        index = transactions.index.to_numpy()
        empty_dates = transactions["empty_date"].to_numpy()
        indices = transactions["indice"].to_numpy()

        remaining_longs = deque()
        results = []
        result_idx = []

        def prop(base, share, total):
            if base == 0:
                return 0.0
            return round(
                (share / base) * total + 1e-8, 2
            )  # correction flottante pour arrondir proprement à 2 décimales

        # Parcours des lignes pour détecter les positions négatives à compenser
        for i, qty in enumerate(value):
            if qty > 0:
                # Si c’est une position longue → on l’ajoute à la file FIFO
                remaining_longs.append((i, qty))
            elif qty < 0:
                # Si position négative : on cherche à la compenser avec les longues précédentes
                needed = -qty
                short_idx = i

                # On vérifie s’il y a assez de positions longues disponibles
                available = sum(q for _, q in remaining_longs)
                if needed > available:
                    raise ValueError(
                        f"Not enough long positions to offset short at {index[i]} "
                        f"(need {needed}, have {available})"
                    )

                # On cherche les positions longues nécessaires pour compenser
                total_matched = 0
                matched = []

                while needed > 0 and remaining_longs:
                    long_idx, long_qty = remaining_longs.popleft()
                    match_qty = min(long_qty, needed)
                    needed -= match_qty
                    total_matched += match_qty
                    # value[long_idx] -= match_qty
                    matched.append((long_idx, match_qty))

                    # Si on n’a pas tout utilisé, on remet le reste dans la file
                    if long_qty > match_qty:
                        remaining_longs.appendleft((long_idx, long_qty - match_qty))
                    else:
                        empty_dates[long_idx] = index[short_idx]

                value_adj[short_idx] = (
                    0  # On marque la position courte comme "neutralisée"
                )
                # value[short_idx] = 0

                # On construit les lignes de transaction de compensation
                for long_idx, match_val in matched:
                    comp = {
                        "value": prop(value[long_idx], match_val, value_adj[long_idx]),
                        "rl_interest_adj": prop(
                            value[long_idx], match_val, rl_interest[long_idx]
                        ),
                        "th_interest_adj": prop(
                            value[long_idx], match_val, th_interest[long_idx]
                        ),
                        "short_date": index[short_idx],
                        "value_short": match_val,
                        "indice": indices[long_idx],
                    }

                    # Mise à jour des intérêts

                    value_adj[long_idx] -= comp["value"]
                    rl_interest[long_idx] -= comp["rl_interest_adj"]
                    th_interest[long_idx] -= comp["th_interest_adj"]
                    results.append(comp)
                    result_idx.append((name, index[long_idx]))

        # Mise à jour du DataFrame d’origine
        transactions.loc[:, "value_adj"] = value_adj
        transactions.loc[:, "rl_interest_adj"] = rl_interest
        transactions.loc[:, "th_interest_adj"] = th_interest
        transactions.loc[:, "empty_date"] = empty_dates

        return pd.DataFrame(
            results,
            index=pd.MultiIndex.from_tuples(result_idx, names=["ticker", "date"]),
        )

    def _apply_value_and_interest(
        self,
        applied_idx: pd.DatetimeIndex,
        prev_interval_pos: int,
        pos: int,
    ):
        """
        Apply current 'value_adj' and 'rl_interest_adj' from transactions-level data to the interval DataFrame.

        This function propagates the latest position-level value and realized interest
        to each time slice in `positions_timeline` between `prev_interval_pos` and `pos`.

        Parameters
        ----------
        positions_timeline : pd.DataFrame
            Interval-based DataFrame (multi-indexed by time and position).
            Columns are a MultiIndex with position keys and ['value_adj', 'rl_interest_adj'].

        transactions : pd.DataFrame
            Position-level DataFrame indexed by position keys.

        applied_idx : Index
            Index of positions to which values should be applied.

        prev_interval_pos : int
            Start position in `positions_timeline.index` for the update interval (inclusive).

        pos : int or None
            End position in `positions_timeline.index` for the update interval (exclusive). If None,
            applies to the end of the DataFrame.

        Returns
        -------
        None
            Updates `positions_timeline` in place.
        """

        transactions_vals = self.transactions.loc[
            applied_idx, ["value_adj", "rl_interest_adj"]
        ].to_numpy()
        self.positions_timeline.loc[
            self.positions_timeline.index[prev_interval_pos:pos],
            (applied_idx, ["value_adj", "rl_interest_adj"]),
        ] = transactions_vals.ravel()

    def _apply_theoretical_interest(
        self,
        applied_idx: pd.DatetimeIndex,
        prev_interest_pos: int,
        pos: int,
        dtype_th: np.dtype,
    ):
        """
        Calculate and apply theoretical interest to positions over a given time interval.

        The function computes the theoretical interest based on the previous period's
        'value_adj' and the current 'rl_interest_adj', then updates both the transactions-level
        and interval-level data structures.

        Parameters
        ----------
        positions_timeline : pd.DataFrame
            MultiIndexed DataFrame with interval rows and (position, metric) columns,
            where metric includes 'value_adj' and 'th_interest_adj'.

        transactions : pd.DataFrame
            Position-level DataFrame. Must include:
            - 'rl_interest_adj'
            - 'th_interest_adj'

        applied_idx : Index
            Index of positions to apply interest computation to.

        prev_interest_pos : int
            Starting index in `positions_timeline.index` (inclusive) for interest computation.

        pos : int
            Ending index in `positions_timeline.index` (exclusive) for interest computation.

        interest_rates : pd.Series
            Series of interest rates to apply over the interval range. Must align with
            positions_timeline index.

        dtype_th : np.dtype
            Data type to cast theoretical interest to (e.g., for precision control).

        Returns
        -------
        None
            Modifies `positions_timeline` and `transactions` in place.
        """

        if (pos > prev_interest_pos) or (pos == -1):
            interest_values = (
                self.positions_timeline.loc[
                    self.positions_timeline.index[prev_interest_pos - 1 : pos],
                    (applied_idx, "value_adj"),
                ]
                .shift(1)
                .iloc[1:]
                .add(self.transactions.loc[applied_idx, "rl_interest_adj"].values)
                .mul(self.interest_rates.iloc[prev_interest_pos:pos], axis=0)
                .cumsum()
                .add(
                    self.transactions.loc[applied_idx, "th_interest_adj"].values,
                    fill_value=0,
                )
            )

            self.positions_timeline.loc[
                self.positions_timeline.index[prev_interest_pos:pos],
                (applied_idx, "th_interest_adj"),
            ] = interest_values.values
            self.transactions.loc[applied_idx, "th_interest_adj"] = (
                self.positions_timeline.loc[
                    self.positions_timeline.index[prev_interest_pos:pos][-1],
                    (applied_idx, "th_interest_adj"),
                ]
                .astype(dtype_th)
                .values
            )

    def _process_positions_timeline(
        self,
    ) -> list[pd.DataFrame]:
        """
        Process interest and value adjustments for a Livret investment portfolio over time.

        This function iterates over time intervals to apply:
        - Adjusted values (`value_adj`)
        - Realized interest (`rl_interest_adj`)
        - Theoretical interest (`th_interest_adj`)

        It also handles short positions by applying a FIFO-based compensation logic, and updates
        both the transactions-level and interval-level tracking DataFrames active_transactions_posordingly.

        Parameters
        ----------
        transactions : pd.DataFrame
            Position-level DataFrame indexed by identifiers. Must contain:
            'value_adj', 'rl_interest_adj', 'th_interest_adj'.
        positions_timeline : pd.DataFrame
            MultiIndexed DataFrame (by time interval and position). Must contain:
            'value_adj', 'rl_interest_adj', and 'th_interest_adj'. Modified in-place.
        interest_la : pd.Series
            Time-indexed interest rates to apply over the interval periods.

        Returns
        -------
        list[pd.DataFrame] | pd.DataFrame
            List of compensation DataFrames for matched short positions, or an empty DataFrame
            if no negative positions exist.

        Raises
        ------
        ValueError
            If there are insufficient positive positions to offset a negative transaction.

        Notes
        -----
        - Updates `transactions` and `positions_timeline` in-place.
        - Applies banking rounding to realized interest after theoretical interest calculations.
        - Combines FIFO logic and interval propagation to track investment accurately.
        """

        long_idx = (
            self.transactions.index
        )  # Index of active positions (e.g., a DatetimeIndex).

        interval_idx = self.positions_timeline.index

        interest_idx = interval_idx.left[
            (interval_idx.left > long_idx[0])
            & (interval_idx.left.month == 1)
            & (interval_idx.left.day == 1)
        ]
        interest_pos = interval_idx.get_indexer(
            interest_idx
        )  # Array of positions where theoretical interest calculations are to be applied (e.g., start of year).
        all_idx = self.transactions.index[1:].union(interest_idx)
        all_pos = interval_idx.get_indexer(
            all_idx
        )  # List or array of position indices (integer indices within `positions_timeline`) to process chronologically.
        short = []
        transactions_pos = 1
        active_transactions_pos = 1
        prev_interval_pos = interval_idx.get_indexer([long_idx[0]])[0]
        prev_interest_pos = prev_interval_pos
        dtype_th = self.transactions["th_interest_adj"].dtype
        dtype_rl = self.transactions["rl_interest_adj"].dtype

        for pos in all_pos:
            applied_idx = long_idx[:transactions_pos]

            self._apply_value_and_interest(applied_idx, prev_interval_pos, pos)

            if interest_pos.size > 0 and interest_pos[0] == pos:
                interest_pos = interest_pos[1:]

                self._apply_theoretical_interest(
                    applied_idx,
                    prev_interest_pos,
                    pos,
                    dtype_th,
                )

                self.transactions.loc[applied_idx, "rl_interest_adj"] = (
                    smart_rounding_banking(
                        self.transactions.loc[applied_idx, "rl_interest_adj"]
                        + (
                            self.transactions.loc[
                                applied_idx, "th_interest_adj"
                            ].astype(dtype_rl)
                        )
                    )
                )

                self.transactions.loc[applied_idx, "th_interest_adj"] = 0

                prev_interest_pos = pos

            else:
                # Traitement de la vente
                if self.transactions.loc[long_idx[transactions_pos], "value_adj"] < 0:

                    self._apply_theoretical_interest(
                        applied_idx,
                        prev_interest_pos,
                        pos,
                        dtype_th,
                    )
                    prev_interest_pos = pos

                    df_compensations = self._reconcile_short_positions(
                        self.transactions.iloc[: active_transactions_pos + 1], self.name
                    )
                    short.append(df_compensations)
                    """
                    self.transactions.loc[long_idx[transactions_pos], "value_adj"] = (
                        0  # Simu vente
                    )
                    """
                    null_idx = self.transactions[
                        self.transactions["value_adj"] == 0
                    ].index
                    n = len(long_idx)
                    long_idx = long_idx.difference(null_idx)
                    transactions_pos -= n - len(long_idx)

                transactions_pos += 1
                active_transactions_pos += 1

            prev_interval_pos = pos

        # Final update for remaining positions
        self._apply_value_and_interest(long_idx, prev_interval_pos, None)
        self._apply_theoretical_interest(long_idx, prev_interest_pos, -1, dtype_th)
        self.positions_timeline.loc[
            self.positions_timeline.index[-1], (long_idx, "th_interest_adj")
        ] = (self.transactions.loc[long_idx, "th_interest_adj"].astype(dtype_rl).values)

        if short == []:
            return pd.DataFrame()
        else:
            return pd.concat(short)
