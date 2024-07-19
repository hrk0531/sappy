# Standard Library
from typing import Any

# Third Party Library
import numpy as np


def cronbach_alpha(x: np.ndarray[Any, Any]) -> np.float64:
    """
    Calculate the Cronbach's alpha coefficient for the input dataset.

    Args:
        x (np.ndarray): The dataset where rows are subjects and columns are items.

    Returns:
        np.float64: The Cronbach's alpha coefficient.
    """
    x = x.copy()
    n_cols = x.shape[1]

    if n_cols < 2:
        raise ValueError("The input dataset must have at least two items.")

    # Calculate the variance of each item
    item_var = np.var(x, axis=0, ddof=1)

    # Calculate the variance of the total score of all items
    total_var = np.var(np.sum(x, axis=1), ddof=1)

    alpha = (n_cols / (n_cols - 1)) * (1 - np.sum(item_var) / total_var)

    return alpha
