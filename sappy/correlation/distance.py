# Standard Library
from typing import Any

# Third Party Library
import numpy as np


def distance(data: np.ndarray[Any, Any], method: str = "euclidean") -> np.ndarray[Any, Any]:
    """
    Compute distance matrix for given method.

    Args:
        data (np.ndarray): Dataset where rows are subjects and columns are variables.

    Returns:
        np.ndarray: Distance matrix.
    """
    x = data.copy()
    n_rows = x.shape[0]

    if method == "euclidean":
        dist_mtx = np.zeros((n_rows, n_rows))
        for i in range(n_rows):
            for j in range(n_rows):
                dist_mtx[i, j] = np.linalg.norm(x[i] - x[j])

    return dist_mtx
