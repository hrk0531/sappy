# Standard Library
from typing import Any

# Third Party Library
import numpy as np


def distance(data: np.ndarray[Any, Any], method: str = "euclidean") -> np.ndarray[Any, Any]:
    """
    Compute distance matrix for given method.
    """
    x = data.copy()
    n_rows = x.shape[0]

    if method == "euclidean":
        dist_mtx = np.zeros((n_rows, n_rows))
        for i in range(n_rows):
            for j in range(n_rows):
                dist_mtx[i, j] = np.linalg.norm(x[i] - x[j])

    return dist_mtx

def mds(dist_mtx: np.ndarray[Any, Any], dimension: int, dissimilarity="precomputed"):
    """
    """
    # The size of distance matrix
    n_rows = dist_mtx.shape[0]
