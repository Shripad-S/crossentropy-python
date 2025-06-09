"""
crossentropy.scaled_gaussian
----------------------------

ScaledGaussian class: Multivariate normal distribution with linear transformation applied
to standardize the covariance matrix for better stability.
"""

import numpy as np
from scipy.stats import multivariate_normal


class ScaledGaussian:
    """Multivariate Gaussian distribution with a linear scaling transformation.

    Attributes:
        mean (np.ndarray): Mean in original space.
        cov (np.ndarray): Covariance in original space.
        mean_T (np.ndarray): Mean in transformed space.
        cov_T (np.ndarray): Covariance in transformed space.
        T_inv (np.ndarray): Inverse transform matrix (maps from transformed to original space).
        T (np.ndarray): Forward transform matrix (maps from original to transformed space).
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray, T_inv: np.ndarray = None, T: np.ndarray = None):
        """
        Initialize the ScaledGaussian.

        Args:
            mean (np.ndarray): Mean vector.
            cov (np.ndarray): Covariance matrix.
            T_inv (np.ndarray, optional): Precomputed inverse transform matrix.
            T (np.ndarray, optional): Precomputed transform matrix.
        """
        if T_inv is None or T is None:
            d = cov.shape[0]
            T_inv = np.eye(d)
            T = np.eye(d)
            for i in range(d):
                T_inv[i, i] = np.sqrt(cov[i, i])
                T[i, i] = 1.0 / T_inv[i, i]

            mean_T = (T @ mean.T).T
            cov_T = T @ cov @ T.T
        else:
            mean_T = mean
            cov_T = cov
            mean = (T_inv @ mean.T).T
            cov = T_inv @ cov @ T_inv.T

        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.mean_T = np.asarray(mean_T, dtype=float)
        self.cov_T = np.asarray(cov_T, dtype=float)
        self.T_inv = np.asarray(T_inv, dtype=float)
        self.T = np.asarray(T, dtype=float)

    def sample(self, n: int) -> np.ndarray:
        """Generate n samples from the distribution.

        Args:
            n (int): Number of samples.

        Returns:
            np.ndarray: Samples of shape (n, d).
        """
        samples_T = np.random.multivariate_normal(self.mean_T, self.cov_T, size=n)
        return (self.T_inv @ samples_T.T).T

    def pdf(self, points: np.ndarray) -> np.ndarray:
        """Compute the pdf at the given points.

        Args:
            points (np.ndarray): Points of shape (n, d).

        Returns:
            np.ndarray: PDF values of shape (n,).
        """
        transformed = (self.T @ points.T).T
        return multivariate_normal.pdf(transformed, mean=self.mean_T, cov=self.cov_T)

    def logpdf(self, points: np.ndarray) -> np.ndarray:
        """Compute the log-density at given points.

        Args:
            points: Array of shape (n, d). Each row is a sample point.

        Returns:
            log_densities: 1D array of length n with log-pdf values.

        Raises:
            ValueError: If the covariance matrix is singular.
        """
        points = np.atleast_2d(points)
        transformed = (self.T @ points.T).T
        return multivariate_normal.logpdf(transformed, mean=self.mean_T, cov=self.cov_T)

    def weighted_mle(self, samples: np.ndarray, weights: np.ndarray) -> "ScaledGaussian":
        """Perform weighted maximum likelihood estimation to fit a ScaledGaussian.

        Args:
            samples (np.ndarray): Input samples of shape (n, d).
            weights (np.ndarray): Weights of shape (n,).

        Returns:
            ScaledGaussian: Fitted distribution.
        """
        transformed = (self.T @ samples.T).T
        weight_sum = np.sum(weights)
        weighted_mean = np.sum(transformed * weights[:, None], axis=0) / weight_sum
        centered = transformed - weighted_mean
        weighted_cov = (centered.T @ (centered * weights[:, None])) / weight_sum

        return ScaledGaussian(weighted_mean, weighted_cov, self.T_inv, self.T)
