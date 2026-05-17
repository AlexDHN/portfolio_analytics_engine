# core/investments/listed.py

import pandas as pd
import numpy as np

import yfinance as yf
from core.investments.base import Investment
from core.market_data.manager_v3 import MarketDataManager
from factories.inflation_factory import get_inflation_service
from utils.fees import fees
from collections import deque
from typing import Optional
from utils.logger import get_logger
from services.cache import load_cache, save_cache

CACHE_MAX_AGE = 60  # 3600  # 60 minutes


class ListedInvestment(Investment):
    """
    ListedInvestment
    ================

    Represents a listed investment instrument such as a stock or ETF.

    Overview
    --------
    This class manages the full lifecycle of a listed position:

    - Transaction ingestion and adjustment (splits, fees, taxes)
    - Historical market data retrieval via a shared MarketDataManager
    - Corporate action processing (e.g., stock splits)
    - Timeline construction from actual and simulated transactions
    - Inflation-adjusted return computation via injected rates

    It is designed for integration into multi-asset portfolio engines,
    supporting parallel instantiation across large ticker universes.

    Timelines
    ---------
    Two timelines may be constructed depending on position history:

    - ``timeline``
        Built from actual transactions. Represents the real investment
        state over time, including closed positions.

    - ``simulated_timeline``
        Built from simulated transactions derived from closed positions.
        Treats sold positions as if they had remained held. Useful for:
            - "what if" analysis on early exits
            - fractional share allocation in simulated scenarios
            - performance attribution vs. a hold strategy

    Design principles
    -----------------
    - Market data is centralized in a shared MarketDataManager, avoiding
      redundant downloads across investments in the same portfolio.
    - Inflation rates are injected at construction time (resolved once
      upstream by ListedAccount) and never fetched internally.
    - Thread-safe by design: no shared mutable state beyond injected
      read-only dependencies.

    Parameters
    ----------
    name : str
        Ticker symbol of the listed instrument (e.g., "AAPL", "AI.PA").
    transactions : pd.DataFrame
        Transaction-level DataFrame containing adjusted fields:
        'count_adj', 'value_adj', 'fees_adj', 'tax_adj', 'fs_adj',
        'indice', 'empty_date'.
    fractional_share_data : pd.DataFrame
        Reference data for fractional share (rompus) allocation,
        typically produced by stock split events. May be empty.
    market_data_manager : MarketDataManager
        Shared manager responsible for downloading, caching, and serving
        historical OHLCV and corporate action data for all portfolio tickers.
    inflation_rates : pd.Series
        Preloaded inflation time series indexed by date, injected from
        the upstream ListedAccount. Used for real return computations.

    Attributes
    ----------
    name : str
        Ticker symbol of the instrument (e.g., "AAPL", "AI.PA").
    transactions : pd.DataFrame
        Adjusted transaction history (buys, sells, splits, fees, taxes).
    market_data : pd.DataFrame
        Historical OHLCV data enriched with corporate actions
        (e.g., 'Stock Splits') retrieved via MarketDataManager.
    fractional_share_data : pd.DataFrame
        Fractional share reference data used during split processing.
    timeline : pd.DataFrame
        Position and performance timeline derived from actual transactions.
    closed_positions : pd.DataFrame
        Matched short/sale compensations. Empty if no positions were closed.
    simulated_transactions : pd.DataFrame
        Synthetic transaction set derived from closed_positions, used to
        reconstruct a simulated hold scenario. Only populated when
        closed_positions is non-empty.
    simulated_timeline : pd.DataFrame
        Timeline computed in simulate mode. Only populated when
        closed_positions is non-empty.
    logger : logging.Logger
        Logger instance scoped to this investment.
    """

    def __init__(
        self,
        name: str,
        transactions: pd.DataFrame,
        fractional_share_data: pd.DataFrame,
        market_data_manager: MarketDataManager,
        inflation_rates: pd.Series,
    ):
        """
        Initialize a ListedInvestment.

        Parameters
        ----------
        name : str
            Ticker symbol of the listed instrument (e.g., "AAPL", "AI.PA").
        transactions : pd.DataFrame
            Adjusted transaction history for this instrument.
        fractional_share_data : pd.DataFrame
            Fractional share reference data for split adjustments.
            Pass an empty DataFrame if not applicable.
        market_data_manager : MarketDataManager
            Shared market data manager. Must be pre-initialized with all
            relevant tickers before ListedInvestment instantiation.
        inflation_rates : pd.Series
            Inflation time series indexed by date. Must be resolved once
            upstream (e.g., in ListedAccount) and injected here to avoid
            redundant loading across parallel investment instantiations.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing ListedInvestment for %s", name)

        self.name = name
        self.transactions = transactions
        self._inflation_rates = inflation_rates

        # --- Load or download market data ---
        self.market_data = market_data_manager.get_market_data(self.name)

        self.fractional_share_data = fractional_share_data

        # --- Process positions timeline ---
        self.positions_timeline, self.closed_positions = (
            self._process_positions_timeline()
        )

        # --- Initialize metrics dictionary ---
        self.metrics = {}

        # --- Compute all metrics at once ---
        self.compute_all_metrics()

        if self.closed_positions.empty:
            self.simulated_transactions = pd.DataFrame()
            self.simulated_positions_timeline = pd.DataFrame()
        else:
            self.simulated_transactions = pd.DataFrame(
                index=self.closed_positions.reset_index()
                .set_index(["short_date"])
                .index
            )
            self.simulated_transactions[
                [
                    "value_adj",
                    "fees_adj",
                    "count_nm",
                    "count_adj",
                    "fs_adj",
                    "tax_adj",
                    "abond_adj",
                    "indice",
                ]
            ] = self.closed_positions[
                [
                    "value_short",
                    "fees_adj_long",
                    "count_short",
                    "count_short",
                    "fs_adj",
                    "tax_adj",
                    "abond_adj",
                    "indice",
                ]
            ].values
            self.simulated_transactions = self.simulated_transactions.groupby(
                ["short_date"]
            ).agg(
                {
                    "value_adj": "first",
                    "fees_adj": "sum",
                    "count_nm": "sum",
                    "count_adj": "sum",
                    "fs_adj": "sum",
                    "tax_adj": "sum",
                    "abond_adj": "sum",
                    "indice": "first",
                }
            )
            self.simulated_transactions["empty_date"] = pd.NaT

            self.simulated_positions_timeline, _ = self._process_positions_timeline(
                simulate=True
            )

        self.logger.info("Listed investment for %s initialized successfully", self.name)

    def _prepare_empty_positions_timeline(self, simulate: bool = False) -> pd.DataFrame:
        """
        Create an empty timeline DataFrame used to record metric values across valuation dates.

        The timeline DataFrame is a MultiIndex-style column layout where each column corresponds
        to a source transaction date and a metric (('tx_date', 'metric')). The row index is the
        valuation dates taken from `self.market_data` beginning at the first relevant transaction date.

        Parameters
        ----------
        simulate : bool, default False
            If True, the function builds the empty timeline based on `self.simulated_transactions`
            (simulated positions created from short compensations). Otherwise it uses `self.transactions`.

        Returns
        -------
        pd.DataFrame
            Empty DataFrame indexed by valuation dates and with MultiIndex columns
            (transaction_date, metric) for metric in:
            ['count_nm', 'count_adj', 'value_adj', 'fees_adj', 'tax_adj', 'fs_adj'].

        Notes
        -----
        - If there are no positive `count_adj` positions (e.g., empty DataFrame) the resulting
          timeline may have zero columns.
        - The timeline index starts at the minimum transaction date present in the chosen
          transactions DataFrame and continues through the available `self.market_data` dates.
        """
        # Extract columns from transactions with positive adjusted count

        if simulate:
            transactions = self.simulated_transactions
        else:
            transactions = self.transactions

        cols = (
            transactions[transactions["count_adj"] > 0][
                ["count_nm", "count_adj", "value_adj", "fees_adj", "tax_adj", "fs_adj"]
            ]
            .T.unstack()
            .index
        )

        df_positions_timeline = pd.DataFrame(
            columns=cols,
            index=self.market_data.loc[transactions.index.min() :].index,
        )

        self.logger.debug(
            "Prepared empty investment timeline: %d rows, %d columns",
            df_positions_timeline.shape[0],
            df_positions_timeline.shape[1],
        )

        return df_positions_timeline

    @staticmethod
    def _apply_stock_split(
        positions: pd.Series, split_ratio: float
    ) -> tuple[pd.Series, pd.Series]:
        """
        Apply a stock split ratio to a Series of positions and compute fractional shares.

        This method adjusts the number of shares according to the split ratio,
        and returns both the adjusted integer positions and the remaining fractional shares.

        Parameters
        ----------
        positions : pd.Series
            Series representing current positions (must be non-negative).
            Indexed by date or transaction identifier.
        split_ratio : float
            Split multiplier (e.g., 1.1 means a 10% increase in quantity).

        Returns
        -------
        tuple[pd.Series, pd.Series]
            - adjusted_positions (pd.Series): New positions including proportionally assigned fractional shares.
            - fractional_shares (pd.Series): Remaining fractions after correction.

        Raises
        ------
        ValueError
            If any position is negative or the split ratio is not strictly positive.

        Notes
        -----
        - Handles fractional shares proportionally to ensure total quantity is preserved.
        - Useful for adjusting positions after corporate actions like stock splits.

        The adjustment preserves the total number of shares across all positions,
        redistributing fractional residues proportionally to existing positions.

        Example
        -------
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> old_count = pd.Series([3.0, 11.0], index=[datetime(2024,2,5), datetime(2024,6,7)], name="count_adj")
        >>> new_pos, frac = ListedInvestment.apply_stock_split(old_count, 1.1)
        >>> print(new_pos)
        2024-02-05     3.0
        2024-06-07    12.0
        Name: count_adj, dtype: float64
        >>> print(frac )
        2024-02-05    0.3
        2024-06-07    0.1
        Name: count_adj, dtype: float64
        """
        # Validate input
        if (positions < 0).any() or split_ratio <= 0:
            raise ValueError(
                "Positions must be non-negative and ratio must be positive."
            )

        # Apply the split
        adjusted = positions * split_ratio

        # Separate integer and fractional parts
        int_part = adjusted.astype(int)
        frac_part = adjusted - int_part

        # Total fractional residue
        residue = frac_part.sum()

        # If no residue, return immediately
        if np.isclose(residue, 0):
            return int_part, frac_part

        # Distribute fractional residue proportionally
        total_correction = int(residue)
        correction = (frac_part / residue) * total_correction

        # Final adjusted positions including correction
        final_positions = int_part + correction
        final_frac = frac_part - correction

        return final_positions, final_frac

    @staticmethod
    def _match_short_positions(transactions: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Reconcile negative (short) positions by matching them with prior positive positions (FIFO).

        This method simulates a FIFO-style matching where short positions (negative `count_adj`)
        are offset against earlier long positions. Fees, taxes, fractional shares, and other
        quantities are proportionally reallocated.

        Parameters
        ----------
        transactions : pd.DataFrame
            Transaction history of an asset. Must contain at least the columns:
            - 'count_adj' : adjusted quantity
            - 'count_nm' : nominal quantity
            - 'value_adj' : adjusted value of the long position
            - 'value' : raw transaction value
            - 'fees_adj' : fees
            - 'tax_adj' : taxes
            - 'fs_adj' : financial adjustments
            - 'abond_adj' : subsidies
            - 'empty_date' : datetime64[ns] (can be NaT)
            - 'indice' : int, transaction identifier

        ticker : str
            Asset ticker symbol.

        Returns
        -------
        pd.DataFrame
            DataFrame of matched transactions with proportional reallocation,
            indexed by MultiIndex (type, ticker, date), containing:
            - 'value_adj_long', 'fees_adj_long', 'short_date', 'count_short',
              'value_short', 'fees_short', 'fs_adj', 'tax_adj', 'abond_adj', 'indice'

            The input `transactions` DataFrame is also modified in-place.

        Raises
        ------
        ValueError
            If the total long positions are insufficient to offset a short position.

        Notes
        -----
        - Modifies the input `transactions` DataFrame in-place by updating adjusted counts,
          fees, taxes, fractional shares, subsidies, and empty_date.
        - Matching logic strictly follows FIFO (first-in, first-out).
        """
        # Convert relevant columns to NumPy arrays for faster computation
        value_adj = transactions["value_adj"].to_numpy()
        count_adj = transactions["count_adj"].to_numpy()
        count_nm = transactions["count_nm"].to_numpy()
        value = transactions["value"].to_numpy()
        index = transactions.index.to_numpy()
        fees = transactions["fees_adj"].to_numpy()
        taxes = transactions["tax_adj"].to_numpy()
        fs = transactions["fs_adj"].to_numpy()
        abond = transactions["abond_adj"].to_numpy()
        indices = transactions["indice"].to_numpy()
        empty_dates = transactions["empty_date"].to_numpy()

        remaining_longs = deque()
        results = []
        result_idx = []

        def prop(base, share, total_to_distribute):
            return 0 if base == 0 else (share / base) * total_to_distribute

        for i, qty in enumerate(count_adj):
            if qty > 0:
                remaining_longs.append((i, qty))
            elif qty < 0:
                needed = -qty
                short_idx = i
                available = sum(q for _, q in remaining_longs)
                if needed > available:
                    raise ValueError(
                        f"Not enough long positions to offset short at {index[i]} "
                        f"(need {needed}, have {available})"
                    )

                matched = []
                while needed > 0 and remaining_longs:
                    long_idx, long_qty = remaining_longs.popleft()
                    match_qty = min(long_qty, needed)
                    needed -= match_qty
                    matched.append((long_idx, match_qty))
                    if long_qty > match_qty:
                        remaining_longs.appendleft((long_idx, long_qty - match_qty))
                    else:
                        empty_dates[long_idx] = index[short_idx]

                count_adj[short_idx] = 0

                for long_idx, match_qty in matched:
                    comp = {
                        "value_adj_long": value_adj[long_idx],
                        "fees_adj_long": prop(
                            count_adj[long_idx], match_qty, fees[long_idx]
                        ),
                        "short_date": index[short_idx],
                        "count_short": match_qty,
                        "value_short": value[short_idx],
                        "fees_short": prop(
                            sum(q for _, q in matched), match_qty, fees[short_idx]
                        ),
                        "fs_adj": prop(count_adj[long_idx], match_qty, fs[long_idx]),
                        "tax_adj": prop(
                            count_adj[long_idx], match_qty, taxes[long_idx]
                        ),
                        "abond_adj": prop(
                            count_adj[long_idx], match_qty, abond[long_idx]
                        ),
                        "indice": indices[long_idx],
                    }

                    # print(sum(q for _, q in matched), match_qty, fees[short_idx])

                    count_nm[long_idx] -= prop(
                        count_adj[long_idx], count_nm[long_idx], match_qty
                    )
                    fees[long_idx] -= comp["fees_adj_long"]
                    fs[long_idx] -= comp["fs_adj"]
                    taxes[long_idx] -= comp["tax_adj"]
                    abond[long_idx] -= comp["abond_adj"]
                    count_adj[long_idx] -= match_qty

                    results.append(comp)
                    result_idx.append((ticker, index[long_idx]))

        # Update original DataFrame in-place
        transactions.loc[:, "count_adj"] = count_adj
        transactions.loc[:, "count_nm"] = count_nm
        transactions.loc[:, "fees_adj"] = fees
        transactions.loc[:, "fs_adj"] = fs
        transactions.loc[:, "tax_adj"] = taxes
        transactions.loc[:, "abond_adj"] = abond
        transactions.loc[:, "empty_date"] = empty_dates

        return pd.DataFrame(
            results,
            index=pd.MultiIndex.from_tuples(result_idx, names=["ticker", "date"]),
        )

    def _process_positions_timeline(
        self, simulate: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the investment lifecycle by updating positions over time, handling stock splits,
        fractional shares, and short position reconciliations.

        This function builds a per-date timeline of the chosen transactions (actual or simulated)
        and returns both the timeline and a DataFrame of compensation rows produced when short
        positions were matched against prior long positions.

        Parameters
        ----------
        simulate : bool, default False
            If True the method processes `self.simulated_transactions` (a synthetic set of transactions
            derived from short compensations) and values fractional shares using market close prices.
            This mode **does not** modify `self.transactions` but _may_ modify `self.simulated_transactions_
            in-place_ (because `transactions` is a reference to the underlying object).
            If False the method processes `self.transactions` and uses `self.fractional_share_data` (if present)
            to allocate fractional shares where available.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            - timeline: DataFrame indexed by valuation dates; columns are MultiIndex
              (transaction_date, metric) where metric is one of
              ['count_nm', 'count_adj', 'fs_adj', 'fees_adj', 'tax_adj', 'value_adj'].
            - compensations_df: DataFrame of short-compensation rows (possibly empty). Each row
              describes a matched portion of a former long position and contains columns such as:
              ['value_adj_long', 'fees_adj_long', 'short_date', 'count_short', 'value_short',
               'fees_short', 'fs_adj', 'tax_adj', 'abond_adj', 'indice'] and is indexed by
              a MultiIndex (ticker, date) where date is the original long position date.

        Side effects
        ------------
        - The chosen transactions DataFrame (self.transactions when simulate=False, or
          self.simulated_transactions when simulate=True) **may be updated in-place** during processing
          (e.g., 'count_adj', 'value_adj', 'fs_adj', 'empty_date' fields are adjusted).
        - The function logs split applications, fractional share allocation issues, and short matches.

        Notes
        -----
        - When `simulate=True`, fractional shares produced by `_apply_stock_split` are valued using
          market close prices from `self.market_data` and added to `self.simulated_transactions['fs_adj']`.
        - When `simulate=False`, fractional shares are allocated using `self.fractional_share_data` if
          the appropriate index is available; otherwise a warning is emitted and rompus remain unassigned.
        """

        # Placeholder for timeline and settled positions
        positions_timeline = self._prepare_empty_positions_timeline(
            simulate
        )  # pd.DataFrame()

        if simulate:
            transactions = self.simulated_transactions
        else:
            transactions = self.transactions

        # Indices of all long positions (positive adjusted count)
        long_transaction_dates = transactions[transactions["count_adj"] > 0].index
        all_transaction_dates = transactions.index

        # Detect stock split dates after the first long purchase
        split_dates = self.market_data.index[
            (self.market_data["Stock Splits"] != 0)
            & (self.market_data.index > long_transaction_dates.min())
        ]
        split_ratios = self.market_data["Stock Splits"].loc[split_dates]

        # All relevant event dates: transactions and splits
        event_dates = all_transaction_dates[1:].union(split_dates)

        # Fractional shares reference dates if not simulating
        fractional_share_dates = (
            []
            if simulate or self.fractional_share_data.empty
            else self.fractional_share_data.index.unique()
        )

        short_position_compensations = []
        last_event_date = all_transaction_dates[0]
        applied_long_count = 1
        active_group_count = 1

        for current_date in event_dates:
            # Apply positions from previous long transactions up to current event date
            current_long_dates = long_transaction_dates[:applied_long_count]
            for tx_date in current_long_dates:
                period_slice = slice(
                    last_event_date, current_date - pd.Timedelta(days=1)
                )
                positions_timeline.loc[
                    period_slice,
                    (
                        tx_date,
                        [
                            "count_nm",
                            "count_adj",
                            "fs_adj",
                            "fees_adj",
                            "tax_adj",
                            "value_adj",
                        ],
                    ),
                ] = transactions.loc[
                    tx_date,
                    [
                        "count_adj",
                        "count_adj",
                        "fs_adj",
                        "fees_adj",
                        "tax_adj",
                        "value_adj",
                    ],
                ].values
            last_event_date = current_date

            # --- Handle Stock Splits ---
            if current_date in split_dates:
                affected_transactions = transactions.index[:applied_long_count]
                new_counts, fractional_shares = self._apply_stock_split(
                    transactions.loc[affected_transactions, "count_adj"],
                    split_ratios[current_date],
                )

                self.logger.debug(
                    "Applying stock split %.2f on %d positions for %s (fractional shares: %.2f)",
                    split_ratios[current_date],
                    len(affected_transactions),
                    self.name,
                    fractional_shares.sum(),
                )

                # Adjust transaction values proportionally
                transactions.loc[affected_transactions, "value_adj"] *= (
                    transactions.loc[affected_transactions, "count_adj"] / new_counts
                )
                transactions.loc[affected_transactions, "count_adj"] = new_counts

                # Update timeline count_nm
                for tx_date in current_long_dates:
                    period_slice = slice(None, current_date - pd.Timedelta(days=1))
                    positions_timeline.loc[
                        period_slice, (tx_date, "count_nm")
                    ] *= split_ratios[current_date]

                # Assign fractional shares
                if simulate:
                    transactions.loc[affected_transactions, "fs_adj"] += (
                        fractional_shares * self.market_data.loc[split_dates, "Close"]
                    )
                else:
                    if current_date in fractional_share_dates:
                        transactions.loc[affected_transactions, "fs_adj"] += (
                            fractional_shares / fractional_shares.sum()
                        ) * self.fractional_share_data.loc[last_event_date].values
                    elif fractional_shares.sum() > 0:
                        self.logger.warning(
                            "No fractional shares assigned at split date %s despite %.2f shares to allocate for %s.",
                            current_date,
                            fractional_shares.sum(),
                            self.name,
                        )

            # --- Handle Sales / Short Positions ---
            else:
                if transactions.iloc[applied_long_count]["count_adj"] < 0:
                    df_comp = self._match_short_positions(
                        transactions.iloc[: active_group_count + 1], self.name
                    )
                    short_position_compensations.append(df_comp)

                    # Remove fully sold transactions from long index
                    sold_indices = transactions[transactions["count_adj"] == 0].index
                    long_transaction_dates = long_transaction_dates.difference(
                        sold_indices
                    )
                    applied_long_count -= applied_long_count - len(
                        long_transaction_dates
                    )

                applied_long_count += 1
                active_group_count += 1

        # Apply final positions after last event
        for tx_date in long_transaction_dates:
            period_slice = slice(last_event_date, None)
            positions_timeline.loc[
                period_slice,
                (
                    tx_date,
                    [
                        "count_nm",
                        "count_adj",
                        "fs_adj",
                        "fees_adj",
                        "tax_adj",
                        "value_adj",
                    ],
                ),
            ] = transactions.loc[
                tx_date,
                [
                    "count_adj",
                    "count_adj",
                    "fs_adj",
                    "fees_adj",
                    "tax_adj",
                    "value_adj",
                ],
            ].values

        self.logger.debug(
            "Investment timeline processed: %d events handled (%d splits, %d sales, %d additional purchases)",
            len(event_dates),
            len(split_dates),
            len(short_position_compensations),
            len(long_transaction_dates),
        )

        return (
            positions_timeline,
            (
                pd.concat(short_position_compensations)
                if short_position_compensations
                else pd.DataFrame()
            ),
        )

    def compute_all_metrics(
        self, fees_func=fees, benchmark_returns: Optional[pd.Series] = None
    ):
        """
        Compute all metrics for this listed investment and populate `self.metrics`.

        This function orchestrates the calculation of:
        1. Position-level and timeline metrics
        2. Performance metrics
        3. Dividend metrics
        4. Risk metrics
        5. Risk-adjusted performance metrics

        Parameters
        ----------
        fees_func : callable, optional
            Function to compute fees on market value. Signature: `fees_func(value: float) -> float`.
            If None, fees are assumed included in the transactions.
        benchmark_returns : pd.Series, optional
            Benchmark returns indexed by date, used for alpha, beta, and information ratio.

        Returns
        -------
        dict
            Dictionary of metrics structured as:
            self.metrics = {
                "performance": {...},
                "risk_adjusted_performance": {...},
                "risk": {...},
                "positions": {...},
            }
        """
        # Initialize metrics dictionary
        self.metrics = {
            "performance": {},
            "risk_adjusted_performance": {},
            "risk": {},
            "positions": {},
        }

        # 1. Positions and timeline metrics
        self._compute_positions_metrics()

        # 2. Performance metrics
        self._compute_performance_metrics(fees_func=fees_func)

        # 3. Risk metrics
        self._compute_risk_metrics()

        # 4. Risk-adjusted performance metrics
        self._compute_risk_adjusted_metrics(benchmark_returns=benchmark_returns)

        return self.metrics

    def _compute_positions_metrics(self):
        """
        Compute position-level metrics for the listed investment, including:
        - Nominal shares, adjusted shares, fractional shares
        - Transaction-level values: long value, fees, taxes, subsidies

        Populates:
            self.metrics["positions"] with:
            - "nominal_shares" : pd.DataFrame of nominal shares held per position
            - "adjusted_shares" : pd.DataFrame of adjusted shares per position
            - "long_value" : pd.DataFrame of transaction-level long values
            - "fees" : pd.DataFrame of actual fees paid
            - "taxes" : pd.DataFrame of taxes paid
            - "subsidies" : pd.DataFrame of subsidies or similar adjustments
        """
        idx = pd.IndexSlice
        df_nominal = self.positions_timeline.loc[:, idx[:, "count_nm"]].droplevel(
            1, axis=1
        )
        df_adjusted = self.positions_timeline.loc[:, idx[:, "count_adj"]].droplevel(
            1, axis=1
        )
        df_long_value = self.positions_timeline.loc[:, idx[:, "value_adj"]].droplevel(
            1, axis=1
        )
        df_fees = self.positions_timeline.loc[:, idx[:, "fees_adj"]].droplevel(
            1, axis=1
        )
        df_taxes = self.positions_timeline.loc[:, idx[:, "tax_adj"]].droplevel(
            1, axis=1
        )

        df_subsidies = (
            self.positions_timeline.loc[:, idx[:, "abond_adj"]].droplevel(1, axis=1)
            if self.transactions["abond_adj"].sum() != 0
            else pd.DataFrame(0.0, index=df_nominal.index, columns=df_nominal.columns)
        )

        self.metrics["positions"] = {
            "nominal_shares": df_nominal,
            "adjusted_shares": df_adjusted,
            "long_value": df_long_value,
            "fees": df_fees,
            "taxes": df_taxes,
            "subsidies": df_subsidies,
        }

    def _compute_performance_metrics(self, fees_func=None):
        """
        Compute position-level performance metrics, including the effect of ETF/annual fees.

        Metrics computed per position and date:
        - Total return (cumulative)
        - Annualized return
        - Market value and equity invested
        - Benefit (market value + dividends + fractional shares - fees - equity)
        - Dividend yield and annualized dividend yield
        - Fractional shares

        ETF/annual fees are incorporated if available in self.transactions['annual_fee_rate'].

        Populates:
            self.metrics["performance"] with:
            - "total_returns", "annualized_return", "market_value", "equity_invested",
            "benefit", "dividend_yield", "annualized_dividend_yield", "dividends_received",
            "fractional_shares"
        """
        # Extract positions
        df = self.metrics["positions"]
        df_nominal = df["nominal_shares"]
        df_adjusted = df["adjusted_shares"]
        df_long_value = df["long_value"]
        df_fees = df["fees"]
        df_taxes = df["taxes"]

        # Fractional shares
        idx = pd.IndexSlice
        df_fractional = self.positions_timeline.loc[:, idx[:, "fs_adj"]].droplevel(
            1, axis=1
        )

        # Market prices and dividends
        df_prices = self.market_data["Close"].reindex(df_nominal.index).ffill()
        df_divs = (
            self.market_data.get("Dividends", pd.Series(0, index=df_nominal.index))
            .reindex(df_nominal.index)
            .fillna(0.0)
        )

        # Market value and equity
        df_market_value = df_nominal.mul(df_prices, axis=0)
        df_equity = df_long_value.mul(df_adjusted, axis=0) + df_fees + df_taxes

        # Include ETF/annual fees if available
        if (
            "annual_fee_rate" in self.transactions.columns
            and not self.transactions["annual_fee_rate"].isna().all()
        ):
            annual_fee_rate = self.transactions["annual_fee_rate"].ffill().iloc[0]
            df_daily_etf_fees = df_market_value * (annual_fee_rate / 252)
            df_cumulative_etf_fees = df_daily_etf_fees.cumsum()
        else:
            df_cumulative_etf_fees = pd.DataFrame(
                0.0, index=df_nominal.index, columns=df_nominal.columns
            )

        # Cumulative dividends
        df_cumulative_divs = df_nominal.mul(df_divs, axis=0).cumsum()

        # Dividend yield per position and date
        df_dividend_yield = df_cumulative_divs / df_equity

        # Fees function (other fees applied on market value)
        if fees_func is None:
            fees_func = lambda x: 0.0

        # Total benefit
        df_benefit = (
            df_market_value
            + df_cumulative_divs
            + df_fractional
            - df_market_value.map(fees_func)
            - df_equity
        )

        # Total cumulative return per position
        df_total_returns = df_benefit.div(df_equity.replace(0, np.nan))

        # Vectorized computation of annualized returns
        purchase_dates = pd.to_datetime(df_total_returns.columns)
        valuation_dates = df_total_returns.index

        # days_held: number of days each position has been held for each valuation date
        days_held = (
            (valuation_dates.to_numpy()[:, None] - purchase_dates.to_numpy())
            .astype("timedelta64[D]")
            .astype(float)
        )
        days_held[days_held <= 0] = np.nan

        # Annualized return and dividend yield (vectorized)
        df_annualized_returns = (1 + df_total_returns.values) ** (
            365.25 / days_held
        ) - 1
        df_annualized_dividend_yield = (1 + df_dividend_yield.values) ** (
            365.25 / days_held
        ) - 1

        # Convert back to DataFrame
        df_annualized_returns = pd.DataFrame(
            df_annualized_returns,
            index=df_total_returns.index,
            columns=df_total_returns.columns,
        )
        df_annualized_dividend_yield = pd.DataFrame(
            df_annualized_dividend_yield,
            index=df_dividend_yield.index,
            columns=df_dividend_yield.columns,
        )

        self.metrics["performance"] = {
            "total_returns": df_total_returns,
            "annualized_return": df_annualized_returns,
            "market_value": df_market_value,
            "equity_invested": df_equity,
            "benefit": df_benefit,
            "dividends_received": df_cumulative_divs,
            "dividend_yield": df_dividend_yield,
            "annualized_dividend_yield": df_annualized_dividend_yield,
            "fractional_shares": df_fractional,
            "estimated_etf_fees": df_cumulative_etf_fees,
        }

    def _compute_risk_metrics(self, confidence_level: float = 0.95):
        """
        Compute standard risk metrics for listed investments.

        Metrics computed:
        - Volatility (standard deviation of returns)
        - Annualized volatility
        - Downside deviation (volatility of negative returns)
        - Maximum drawdown
        - Value at Risk (VaR)
        - Conditional Value at Risk (CVaR / Expected Shortfall)

        Parameters
        ----------
        confidence_level : float, default=0.95
            Confidence level for VaR and CVaR.

        Populates
        ----------
        self.metrics["risk"]
        """

        # Daily returns
        df_returns = (self.market_data["Adj Close"]).pct_change()

        # Volatility (standard deviation of daily returns)
        volatility = df_returns.std()

        # Annualized volatility
        annualized_volatility = volatility * np.sqrt(252)

        # Downside deviation (std of negative returns)
        downside_deviation = df_returns[df_returns < 0].std()

        # Maximum drawdown
        # Compute cumulative returns
        cumulative_returns = (1 + df_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Value at Risk (VaR) at confidence level
        VaR = df_returns.quantile(1 - confidence_level)

        # Conditional Value at Risk (Expected Shortfall)
        CVaR = df_returns[df_returns <= VaR].mean()

        # Populate risk metrics
        self.metrics["risk"] = {
            "volatility": volatility,
            "annualized_volatility": annualized_volatility,
            "downside_deviation": downside_deviation,
            "max_drawdown": max_drawdown,
            "VaR": VaR,
            "CVaR": CVaR,
        }

    def _compute_risk_adjusted_metrics(
        self, benchmark_returns: Optional[pd.Series] = None
    ):
        """
        Compute risk-adjusted performance metrics:
        - Sharpe ratio
        - Sortino ratio
        - Alpha, Beta, Information Ratio (requires benchmark)

        Populates:
            self.metrics["risk_adjusted_performance"]
        """
        df_returns = self.metrics["performance"]["benefit"].div(
            self.metrics["performance"]["equity_invested"].replace(0, np.nan)
        )
        self.metrics["risk_adjusted_performance"] = {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "alpha": None,
            "beta": None,
            "information_ratio": None,
        }
