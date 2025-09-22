"""
crossentropy.gaussian_mixture
-------------------------
Gaussian Mixture Model (GMM) with EM fitting (unweighted and weighted),
backed by ScaledGaussian components that keep a fixed linear scaling (T, T_inv)
per component across EM updates for numerical stability.
"""

import numpy as np
from sklearn.cluster import KMeans
from .scaledgaussian import ScaledGaussian


class GaussianMixture:
    """Gaussian Mixture Model (GMM) using ScaledGaussian components.

    Attributes:
        components (list[ScaledGaussian]): List of Gaussian components, length K.
        weights (np.ndarray): Mixing proportions, 1D array of shape (K,).
    """

    def __init__(self, components: list[ScaledGaussian], weights: np.ndarray):
        """
        Args:
            components: List of Gaussian objects (length K).
            weights: 1D array of length K, must sum to 1.
        """
        self.components = components
        self.weights = np.asarray(weights, dtype=float).ravel()
        if len(self.components) != self.weights.shape[0]:
            raise ValueError("Number of components and weights must match.")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("Mixing weights must sum to 1.")

    def sample(self, n: int) -> np.ndarray:
        """Draw n samples from the mixture.

        Args:
            n: Number of samples.

        Returns:
            samples: Array of shape (n, d).
        """
        K = len(self.components)
        comp_idxs = np.random.choice(K, size=n, p=self.weights)
        d = self.components[0].mean.shape[0]
        samples = np.zeros((n, d))

        for k in range(K):
            idx_k = np.where(comp_idxs == k)[0]
            if idx_k.size > 0:
                samples[idx_k] = self.components[k].sample(idx_k.size)
        return samples

    def pdf(self, points: np.ndarray) -> np.ndarray:
        """Compute the mixture pdf at each point.

        Args:
            points: Array of shape (n, d).

        Returns:
            densities: 1D array of length n.
        """
        points = np.atleast_2d(points)
        n = points.shape[0]
        K = len(self.components)
        densities = np.zeros((n, K))

        for k, comp in enumerate(self.components):
            densities[:, k] = self.weights[k] * comp.pdf(points)
        return densities.sum(axis=1)

    @staticmethod
    def _empirical_mean_cov(samples: np.ndarray, weights: np.ndarray = None):
        """Compute (weighted) empirical mean and covariance in original space."""
        X = np.atleast_2d(samples)
        n, d = X.shape
        if weights is None:
            # Unweighted
            mu = X.mean(axis=0)
            C = np.cov(X.T, bias=True)  # ML estimate (divide by n)
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.shape[0] != n:
                raise ValueError("Length of weights must match number of samples.")
            wsum = w.sum()
            if wsum == 0:
                raise ValueError("Sum of weights must be positive.")
            mu = (w[:, None] * X).sum(axis=0) / wsum
            XC = X - mu
            C = (XC.T @ (XC * (w[:, None] / wsum)))
        return mu, C

    @staticmethod
    def _init_components_with_kmeans(samples: np.ndarray, labels: np.ndarray, K: int):
        """Initialize ScaledGaussian components from k-means clusters.

        Each component is initialized ONCE; its constructor computes a diagonal scaling
        so that the transformed covariance has ones on the diagonal. This scaling is
        then FIXED and carried through EM via ScaledGaussian.weighted_mle(...).
        """
        components = []
        for k in range(K):
            cluster_samples = samples[labels == k]
            if cluster_samples.shape[0] == 0:
                # Fallback: if an empty cluster somehow occurs, seed from global stats
                mu0, C0 = GaussianMixture._empirical_mean_cov(samples)
            else:
                mu0, C0 = GaussianMixture._empirical_mean_cov(cluster_samples)
            comp = ScaledGaussian(mu0, C0)  # sets T, T_inv once for this component
            components.append(comp)
        return components

    @staticmethod
    def mle(samples: np.ndarray, num_components: int = 1) -> "GaussianMixture":
        """Fit a GMM by unweighted EM with k-means initialization.

        Procedure:
          1. Run k-means with K = num_components on 'samples'.
          2. Initialize each component as a ScaledGaussian (fixing its T/T_inv once).
          3. Iterate EM exactly 1000 times:
               - E-step: compute responsibilities r[i,k]
               - M-step: for each component k, call comp.weighted_mle(samples, r[:,k])
                 (this updates mean/cov in ORIGINAL space while preserving T/T_inv),
                 and update mixing coefficients as sum(r[:,k]) / n.

        Args:
            samples: Array of shape (n, d).
            num_components: Number of mixture components, K (default=1).

        Returns:
            GaussianMixture: Fitted model.
        """
        samples = np.atleast_2d(samples)
        n, _ = samples.shape

        # Unweighted EM → all-sample weights are ones (used only for mixing update)
        weights_vec = np.ones(n, dtype=float)

        # K-means initialization
        kmeans = KMeans(n_clusters=num_components, init="k-means++", n_init=1)
        labels = kmeans.fit_predict(samples)

        # Build initial components with FIXED scaling per component
        components = GaussianMixture._init_components_with_kmeans(samples, labels, num_components)
        mixing = np.full(num_components, 1.0 / num_components, dtype=float)

        # EM loop for exactly 1000 iterations
        for _ in range(1000):
            resp = GaussianMixture._compute_responsibilities(components, samples, mixing)
            components, mixing = GaussianMixture._m_step(samples, weights_vec, resp, components)

        return GaussianMixture(components, mixing)

    @staticmethod
    def weighted_mle(
        samples: np.ndarray,
        weights: np.ndarray,
        num_components: int = 1
    ) -> "GaussianMixture":
        """Fit a GMM by weighted EM with k-means initialization.

        Procedure:
          1. Run k-means with K = num_components on 'samples' (clustering ignores weights).
          2. Initialize each component as a ScaledGaussian (fixing its T/T_inv once).
          3. Iterate EM exactly 1000 times:
               - E-step: compute responsibilities r[i,k]
               - M-step: for each k, call comp.weighted_mle(samples, w[i]*r[i,k]),
                 and set mixing[k] = sum(w[i]*r[i,k]) / sum(w).

        Args:
            samples: Array of shape (n, d).
            weights: 1D array of length n (nonnegative).
            num_components: Number of mixture components, K (default=1).

        Returns:
            GaussianMixture: Fitted model.
        """
        samples = np.atleast_2d(samples)
        weights = np.asarray(weights, dtype=float).ravel()
        n, _ = samples.shape
        if weights.shape[0] != n:
            raise ValueError("Length of weights must match number of samples.")

        # K-means init (on locations only)
        kmeans = KMeans(n_clusters=num_components, init="k-means++", n_init=1)
        labels = kmeans.fit_predict(samples)

        # Initial components with FIXED scaling per component
        components = GaussianMixture._init_components_with_kmeans(samples, labels, num_components)
        mixing = np.full(num_components, 1.0 / num_components, dtype=float)

        # EM loop for exactly 1000 iterations
        for _ in range(1000):
            resp = GaussianMixture._compute_responsibilities(components, samples, mixing)
            components, mixing = GaussianMixture._m_step(samples, weights, resp, components)

        return GaussianMixture(components, mixing)

    @staticmethod
    def _compute_responsibilities(
        components,
        samples: np.ndarray,
        mixing: np.ndarray
    ) -> np.ndarray:
        """Compute posterior responsibilities r[i,k] ∝ mixing[k] * p_k(x_i).

        Args:
            components: List of ScaledGaussian components (length K).
            samples: Array of shape (n, d).
            mixing: 1D array of mixing weights (length K).

        Returns:
            resp: Array of shape (n, K) where each row sums to 1.
        """
        n, _ = samples.shape
        K = len(components)
        log_probs = np.zeros((n, K))

        for k in range(K):
            try:
                log_probs[:, k] = np.log(mixing[k]) + components[k].logpdf(samples)
            except ValueError:
                log_probs[:, k] = -np.inf

        # Numerically stable normalization via log-sum-exp
        max_log = np.max(log_probs, axis=1, keepdims=True)
        shifted = np.exp(log_probs - max_log)
        sum_shifted = shifted.sum(axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(sum_shifted)

        log_resp = log_probs - log_sum_exp
        return np.exp(log_resp)

    @staticmethod
    def _m_step(
        samples: np.ndarray,
        weights: np.ndarray,
        responsibilities: np.ndarray,
        components_current: list
    ):
        """M-step: update each ScaledGaussian and mixing weights (preserving T/T_inv).

        For each component k:
          w_total[k] = sum_i [ weights[i] * responsibilities[i,k] ]
          new_mixing[k] = w_total[k] / sum(weights)
          new_comp[k] = components_current[k].weighted_mle(samples, weights * responsibilities[:,k])

        Returns:
            (components, new_mixing)
        """
        n, _ = samples.shape
        K = responsibilities.shape[1]
        total_weights = weights[:, None] * responsibilities  # (n, K)
        weight_sum = weights.sum()

        new_components = []
        new_mixing = np.zeros(K, dtype=float)

        for k in range(K):
            w_k = total_weights[:, k]
            # IMPORTANT: call the INSTANCE method so scaling (T, T_inv) is reused
            comp_updated = components_current[k].weighted_mle(samples, w_k)
            new_components.append(comp_updated)
            new_mixing[k] = w_k.sum() / weight_sum

        return new_components, new_mixing
