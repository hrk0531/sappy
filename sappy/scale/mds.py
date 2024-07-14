# Standard Library
from typing import Any

# Third Party Library
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# First Party Library
from sappy.correlation.distance import distance


def mds(data: np.ndarray[Any, Any], n_components: int = 2) -> np.ndarray[Any, Any]:
    """
    """
    dist_mtx = distance(data)
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

    x = eigvecs_idx[:, :n_components] @ np.diag(np.sqrt(eigvals_idx[:n_components]))

    return x


def plot_mds(data:np.ndarray[Any, Any], title: str) -> None:
    """
    """
    result = pl.DataFrame(mds(data), schema=['Dimension1', 'Dimension2'])

    # プロット
    plt.figure(figsize=(10, 10), dpi=300)

    # 各点を黒でプロット
    plt.scatter(result[:, 0], result[:, 1], color='black')

    # 各点の左下に列番号を表示
    for i in range(result.shape[0]):
        plt.text(result[i, 0] + 0.01, result[i, 1] - 0.10, str(i), fontsize=12, ha='right', va='top')

    # 軸のメモリを削除
    plt.xticks([])
    plt.yticks([])

    # ラベルとタイトルを追加
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)

    plt.show()
