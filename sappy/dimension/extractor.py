# Standard Library
from typing import Any, Tuple

# Third Party Library
import numpy as np

# First Party Library
from sappy.utils import corr, cov


def eig_sort(data: np.ndarray[Any, Any]) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Sort the eigenvalues and eigenvectors of the input dataset.

    Args:
        data (np.ndarray): Dataset where rows are subjects and columns are variables.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The indices of the sorted eigenvalues, and the sorted eigenvectors.
            - select: The indices of the sorted eigenvalues.
            - selected_eigvals: The sorted eigenvalues.
            - selected_eigvecs: The eigen vectors corresponding to the sorted eigenvalues.
    """
    r = corr(data)

    eigvals, eigvecs = np.linalg.eig(r)

    select = np.where(eigvals >= 1)[0]
    selected_eigvals = eigvals[select]
    selected_eigvecs = eigvecs[:, select]

    return select, selected_eigvals, selected_eigvecs


def n_factors(data: np.ndarray[Any, Any]) -> int:
    """
    Calculate the number of factors with eigenvalues grater than or equal 1,
    based on the correlation matrix of the input dataset.

    Args:
        data (np.ndarray): Dataset where rows are subjects and columns are variables.

    Returns:
        int: Number of factors with eigenvalues grater than or equal to 1.
    """
    r = corr(data)

    select, _, _ = eig_sort(r)

    n_factors = len(select)

    return n_factors


def unique_variances(data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    """
    c = cov(data)

    eigvals, _ = np.linalg.eig(c)

    psi_sq = np.diag(eigvals)

    return psi_sq


class Extraction:
    def __init__(self) -> None:
        pass

    def principal(self, data:np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        """
        x = data.copy()
        r = corr(x)

        _, selected_eigvals, selected_eigvecs = eig_sort(r)

        # Calculate the matrix of factor loadings
        loadings = selected_eigvecs @ np.sqrt(np.diag(np.abs(selected_eigvals)))

        return loadings

    def compute_f(self, data: np.ndarray[Any, Any]) -> Tuple[float, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        """
        m = n_factors(data)
        r = corr(data)
        psi_sq = unique_variances(data)
        eigvals, eigvecs = np.linalg.eig(psi_sq @ np.linalg.inv(r) @ psi_sq)

        f_psi = np.sum(np.log(eigvals[m+1:]) + 1/eigvals[m+1:] - 1)

        return f_psi, eigvals, eigvecs

    def grad_hessian(self, data: np.ndarray[Any, Any]) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Computes the gradient and Hessian of the objective function f(psi^2).

        Args:
            data (np.ndarray): Dataset where rows are subjects and columns are variables.
        """
        f_psi, eigvals, eigvecs = self.compute_f(data)
        r = corr(data)
        n_rows = r.shape[0]
        m = n_factors(data)

        grad = np.zeros(n_rows)
        hessian = np.zeros((n_rows, n_rows))

        for i in range(n_rows):
            grad[i] = np.sum(1 - 1/eigvals[m+1:]) * eigvecs[i, m+1:]**2

            for j in range(n_rows):
                if i == j:
                    hessian[i, j] = -grad[i] + np.sum(eigvecs[i, m+1:]**2 * (1 - 1/eigvals[m+1:]))
                else:
                    hessian[i, j] = np.sum(eigvals[i, m+1:] * eigvals[j, m+1:]) * np.sum(
                                    (eigvals[m:] + eigvals[:m] - 2) / (eigvals[m:] - eigvals[:m])[:, None] *
                                    eigvecs[i, :m] * eigvecs[j, :m], axis=0)

        return grad, hessian

    def newton_raphson(self, data: np.ndarray[Any, Any]) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Performs a Newton-Raphson step to update the variable x.
        """
        x = data.copy()
        grad, hessian = self.grad_hessian(data)

        d = np.linalg.solve(grad, hessian)
        x_neq = x - d

        return x_neq, d

    def ml(self, data:np.ndarray[Any, Any]):
        """
        """
        m = n_factors(data)
        r = corr(data)
        n_rows = r.shape[0]
        x = np.log((1 - m/(2*n_rows)) / np.diag(np.linalg.inv(r)))

        for iter in range(1000):
            grad, hessian = self.grad_hessian(data)
            x_new, d = self.newton_raphson(data)

            if np.max(np.abs(d)) < 1e-4:
                break
