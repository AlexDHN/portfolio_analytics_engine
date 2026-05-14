import numpy as np
import pandas as pd


def smart_rounding_banking(row: pd.Series, decimals: int = 2) -> pd.Series:
    """
    Bank-compliant smart rounding:
    - Round the total sum of the series down to the nearest cent.
    - Round each individual value to the nearest cent using standard rounding.
    - Adjust individual values so their sum matches the floored total.

    Parameters
    ----------
    row : pd.Series
        Numeric row to be rounded.
    decimals : int, default=2
        Decimal precision (typically 2 for euro cents).

    Returns
    -------
    pd.Series
        Rounded values adjusted to match the floored total sum.
    """

    step = 10 ** (-decimals)

    # Remove NaNs
    non_nan = row.dropna()
    original = non_nan.astype(float)

    # Step 1: Round total sum down to nearest cent
    floored_total = np.floor(original.sum() / step) * step

    # Step 2: Standard rounding of each individual value
    rounded = original.round(decimals)

    # Step 3: Adjust so that the total matches the floored total
    diff = round(floored_total - rounded.sum(), decimals)

    if np.isclose(diff, 0.0):
        return rounded.reindex(row.index)

    steps_needed = int(round(abs(diff) / step))
    residuals = original - rounded

    # Determine which values to adjust
    if diff < 0:
        # We need to subtract step from values with largest positive residuals
        sorted_idx = residuals.argsort()[::-1]
        adjustment = -step
    else:
        # We need to add step to values with largest negative residuals
        sorted_idx = residuals.argsort()
        adjustment = step

    # Distribute the steps_needed
    repeat = steps_needed // len(sorted_idx)
    remainder = steps_needed % len(sorted_idx)
    full_idx = np.concatenate([np.tile(sorted_idx, repeat), sorted_idx[:remainder]])
    adj_counts = np.bincount(full_idx, minlength=len(original))

    adjustment_series = pd.Series(adj_counts * adjustment, index=original.index)

    corrected = rounded.add(adjustment_series, fill_value=0.0)

    # Restore original shape (fill back NaNs if any)
    return corrected.reindex(row.index)
