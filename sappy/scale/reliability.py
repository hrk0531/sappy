# Standard Library
from typing import Any

# Third Party Library
import numpy as np


def cronbach_alpha(data: np.ndarray[Any, Any]) -> np.float64:
    """
    Calculate tht Cronbach's alpha coefficient for the given dataset.

    Args:
        data (np.ndarray): Dataset where rows are subjects and columns are variables.

    Returns:
        np.float64: The Cronbach's alpha coefficient.
    """
    x = data.copy()

    n_cols = x.shape[1]

    if n_cols < 2:
        raise ValueError("The input dataset must have at least two items.")

    # Calculate the variance of each item
    item_var = np.var(x, axis=0, ddof=1)

    # Calculate the variance of total score of all items
    total_score = np.sum(x, axis=1)
    total_var = np.var(total_score, ddof=1)

    alpha = (n_cols / (n_cols - 1) ) * (1 - np.sum(item_var) / total_var)

    return alpha
