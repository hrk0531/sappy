# Standard Library
from typing import Any, Optional, Tuple, Union

# Third Party Library
import numpy as np


def mean(
        data: np.ndarray[Any, Any],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        weight: Optional[np.ndarray[Any, Any]] = None
) -> Any:
    """
    Calculate the weighted mean along the specified axis.

    Args:
        data (np.ndarray): The input dataset.
        axis (int, tuple): The axis or axes along which to calculate the mean.
        weight (np.ndarray): The weights to apply to the data.
    """
    x = data.copy()

    mean = np.average(x, axis=axis, weights=weight)

    return mean


def corr(data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Calculate the correlation matrix of the input dataset.
    """
    x = data.copy()

    # Standardize the data
    x_s = (x - mean(x))/ np.std(x, ddof=1)

    # Calculate the correlation matrix
    corr = np.corrcoef(x_s, rowvar=False)

    return corr
