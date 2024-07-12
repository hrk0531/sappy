# Standard Library
from typing import Any

# Third Party Library
import numpy as np

# First Party library
from sappy.dimension.extractor import eig_sort
from sappy.utils import corr


class Rotation:
    def __init__(
            self,
            kaiser: bool = True,
            delta: float = 0.0,
            gamma: float = 1.0,
            kappa: float = 4.0,
            power: float = 4.0,
            max_iter: int = 1000,
            tol : float = 1e-4,
    ):
        """Initialize the Rotation class."""
        self.kaiser = kaiser
        self.delta = delta
        self.gamma = gamma
        self.kappa = kappa
        self.power = power
        self.max_iter = max_iter
        self.tol = tol

    def varimax(self, loadings:np.ndarray[Any, Any]) -> Any:
        """
        Perform the varimax rotation on the unrotated factor loading matrix.

        Args:
            loadings (np.ndarray): The unrotated factor loading matrix.

        Returns:
            np.ndarray: The rotated factor loading matrix.
            np.ndarray: The transform matrix
        """
        x = loadings.copy()
        r = corr(x)

        if not np.any(x != 0, axis=1).all():
            raise ValueError("Each row must contain at least one non-zero element.")

        n_rows, n_cols = x.shape
        if n_cols < 2:
            raise ValueError("At least two variables are required.")

        _, selected_eigvals, selected_eigvecs = eig_sort(r)
        # Kaiser normalization
        if self.kaiser:
            communality = np.sum(np.abs(selected_eigvals ** selected_eigvecs**2))
            x_nld = x / (np.sqrt(np.sum(communality, axis=1))[:np.newaxis])

            transform = np.eye(n_cols)
            d = 0

            for i in range(self.max_iter):
                d_old = d
                x_rot = x_nld @ transform
                u, s, vh = np.linalg.svd(np.dot(x_nld.T, np.asarray(x_rot)**3 - (self.gamma/n_rows) *
                                                 np.dot(x_nld, np.diag(np.diag(x_rot @ x_rot.T)))))
                transform = np.dot(u, vh)
                d = np.sum(s)
                if d_old != 0 and (d / d_old) < 1 + self.tol:
                    break

            return x_rot, transform
