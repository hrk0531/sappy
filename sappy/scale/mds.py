# Standard Library
from typing import Any

# Third Party Library
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# First Party Library
from sappy.correlation.distance import distance


def mds(x: np.ndarray[Any, Any], n_dimension: int = 2) -> np.ndarray[Any, Any]:
    """
    Perform Multidimensional Scaling (MDS) on the input dataset.

    Args:
        x (np.ndarray): Dataset where rows are subjects and columns are variables.
        n_dimension (int): Number of dimensions to reduce to. Defaults to 2.

    Returns:
        np.ndarray: Dimensionally reduced dataset.
    """
    dist_mtx = distance(x)
    d = dist_mtx.copy()

    # The size of distance matrix
    n_rows, n_cols = dist_mtx.shape

    # h is the centering matrix
    h = np.eye(n_rows) - np.ones((n_rows, n_rows)) / n_cols

    # Initial karnel matrix
    k = -0.5 * h.dot(d**2).dot(h)

    eigvals, eigvecs = np.linalg.eigh(k)
    idx = eigvals.argsort()[::-1]
    eigvals_idx = eigvals[idx]
    eigvecs_idx = eigvecs[:, idx]

    mds = eigvecs_idx[:, :n_dimension] @ np.diag(np.sqrt(eigvals_idx[:n_dimension]))

    return mds


def plot_mds(x: np.ndarray[Any, Any], title: str) -> None:
    """
    Plot Multidimensional Scaling (MDS) result.

    Args:
        x (np.ndarray): Dataset where rows are subjects and columns are variables.
        title (str): Title of the plot
    """
    result = pl.DataFrame(mds(x), schema=['Dimension1', 'Dimension2'])

    plt.figure(figsize=(10, 10), dpi=300)

    # Plot each point in black
    plt.scatter(result[:, 0], result[:, 1], color='black')

    # Column numbers are displayed in the lower left corner of each point
    for i in range(result.shape[0]):
        plt.text(result[i, 0] + 0.01, result[i, 1] - 0.10, str(i), fontsize=12, ha='right', va='top')

    # Remove axis scale
    plt.xticks([])
    plt.yticks([])

    # Add axis labels and titles
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
