# Standard Library
from typing import Any

#Third Party Library
import numpy as np


def corr(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Calculate the correlation matrix of the input dataset.

    Args:
        x (np.ndarray): Input dataset where rows are subjects and columns are variables.

    Returns:
        np.ndarray: The correlation matrix of the input dataset.
    """
    x = x.copy()

    # Standardie the data
    x_s = (x - np.mean(x)) / np.std(x, ddof=1)

    r = np.corrcoef(x_s, rowvar=False)

    return r
