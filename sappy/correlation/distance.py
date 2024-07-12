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


def mds(dist_mtx: np.ndarray[Any, Any], n_components: int) -> np.ndarray[Any, Any]:
    """
    """
    d = dist_mtx.copy()

    # The size of distance matrix
    n_rows, n_cols = dist_mtx.shape

    # H is the centering matrix
    h = np.eye(n_rows) - np.ones((n_rows, n_rows)) / n_cols

    # Initial karnel matrix
    k = -0.5 * h.dot(d**2).dot(h)

    eigvals, eigvecs = np.linalg.eigh(k)
    idx = eigvals.argsort()[::-1]
    eigvals_idx = eigvals[idx]
    eigvecs_idx = eigvecs[:, idx]

    x = eigvecs_idx[:, :n_components].T @ np.diag(np.sqrt(eigvals_idx[:n_components]))

    return x
