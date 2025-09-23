"""
crossentropy.gaussian_mixture
-------------------------
Gaussian Mixture Model (GMM) with EM fitting (unweighted and weighted).

Design:
- Build ONE global linear scaling (T, T_inv) from the initial distribution (mean0, cov0).
- Transform all samples once into the scaled space: X_T = T @ X.
- Run k-means and EM in the scaled space (unit variances on the diagonal).
- Represent each component as ScaledGaussian(mean, cov, T_inv, T) so all share the SAME scaling.
- No per-component or per-iteration recomputation of T/T_inv.
"""

import numpy as np
from sklearn.cluster import KMeans
from .scaledgaussian import ScaledGaussian
from typing import Optional, List, Tuple

class GaussianMixture:
    """Gaussian Mixture Model (GMM) using a SINGLE global scaling (T, T_inv).

    Attributes:
        components (list[ScaledGaussian]): Components; all share the same T/T_inv.
        weights (np.ndarray): Mixing proportions, shape (K,).
        T (np.ndarray): Global forward transform (original -> scaled).
        T_inv (np.ndarray): Global inverse transform (scaled -> original).
    """

    def __init__(self, components, weights: np.ndarray, T_inv: np.ndarray, T: np.ndarray):
        self.components = list(components)
        self.weights = np.asarray(weights, dtype=float).ravel()
        self.T_inv = np.asarray(T_inv, dtype=float)
        self.T = np.asarray(T, dtype=float)

        if len(self.components) != self.weights.shape[0]:
            raise ValueError("Number of components and weights must match.")
        if not np.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("Mixing weights must be nonnegative and finite.")
        s = self.weights.sum()
        if not np.isclose(s, 1.0):
            if s <= 0:
                raise ValueError("Mixing weights sum must be positive.")
            self.weights = self.weights / s  # normalize defensively

    # ---------- Public API ----------

    def sample(self, n: int) -> np.ndarray:
        """Draw n samples from the mixture (original space)."""
        K = len(self.components)
        idx = np.random.choice(K, size=n, p=self.weights)
        d = self.components[0].mean.shape[0]
        out = np.zeros((n, d))
        for k in range(K):
            sel = np.where(idx == k)[0]
            if sel.size:
                out[sel] = self.components[k].sample(sel.size)  # ScaledGaussian returns original-space samples
        return out

    def pdf(self, points: np.ndarray) -> np.ndarray:
        """Mixture pdf at original-space points."""
        pts = np.atleast_2d(points)
        n = pts.shape[0]
        K = len(self.components)
        vals = np.zeros((n, K))
        for k, comp in enumerate(self.components):
            vals[:, k] = self.weights[k] * comp.pdf(pts)
        return vals.sum(axis=1)

    # ---------- Fit routines (global scaling once) ----------

    @staticmethod
    def mle(
        samples: np.ndarray,
        num_components: int = 1,
        mean0: Optional[np.ndarray] = None,
        cov0: Optional[np.ndarray] = None,
        n_init_kmeans: int = 1,
        em_iters: int = 1000,
    ) -> "GaussianMixture":
        """
        Unweighted EM with k-means init, using ONE global scaling from (mean0, cov0).
        If mean0/cov0 not provided, they are computed from all samples (unweighted).
        """
        X = np.atleast_2d(samples)
        n, d = X.shape

        # 1) Build ONE global scaler from (mean0, cov0)
        if mean0 is None or cov0 is None:
            mean0, cov0 = _empirical_mean_cov(X, None)
        global_scaler = ScaledGaussian(mean0, cov0)  # computes T, T_inv once
        T, T_inv = global_scaler.T, global_scaler.T_inv

        # 2) Transform ALL samples once into scaled space
        X_T = (T @ X.T).T

        # 3) K-means in scaled space
        kmeans = KMeans(n_clusters=num_components, init="k-means++", n_init=n_init_kmeans)
        labels = kmeans.fit_predict(X_T)

        # 4) Initialize components in scaled space, then convert to original; share T/T_inv
        components, mixing = _init_components_in_T_space(X_T, labels, num_components, T_inv, T)

        # 5) EM loop in scaled space (using shared T/T_inv)
        weights_vec = np.ones(n, dtype=float)
        for _ in range(em_iters):
            resp = _compute_responsibilities_T(components, X, mixing)  # uses comp.logpdf(X) with shared T
            components, mixing = _m_step_T(X_T, weights_vec, resp, T_inv, T)

        return GaussianMixture(components, mixing, T_inv, T)

    @staticmethod
    def weighted_mle(
        samples: np.ndarray,
        weights: np.ndarray,
        num_components: int = 1,
        mean0: Optional[np.ndarray] = None,
        cov0: Optional[np.ndarray] = None,
        n_init_kmeans: int = 1,
        em_iters: int = 1000,
    ) -> "GaussianMixture":
        """
        Weighted EM with k-means init, using ONE global scaling from (mean0, cov0).
        If mean0/cov0 not provided, they are computed from all samples (weighted).
        """
        X = np.atleast_2d(samples)
        w = np.asarray(weights, dtype=float).ravel()
        n, d = X.shape
        if w.shape[0] != n:
            raise ValueError("Length of weights must match number of samples.")
        if w.sum() <= 0:
            raise ValueError("Sum of weights must be positive.")

        # 1) Build ONE global scaler from (mean0, cov0)
        if mean0 is None or cov0 is None:
            mean0, cov0 = _empirical_mean_cov(X, w)
        global_scaler = ScaledGaussian(mean0, cov0)
        T, T_inv = global_scaler.T, global_scaler.T_inv

        # 2) Transform ALL samples once into scaled space
        X_T = (T @ X.T).T

        # 3) K-means in scaled space (clustering ignores weights)
        kmeans = KMeans(n_clusters=num_components, init="k-means++", n_init=n_init_kmeans)
        labels = kmeans.fit_predict(X_T)

        # 4) Initialize components in scaled space, then convert to original; share T/T_inv
        components, mixing = _init_components_in_T_space(X_T, labels, num_components, T_inv, T)

        # 5) EM loop in scaled space (using shared T/T_inv)
        for _ in range(em_iters):
            resp = _compute_responsibilities_T(components, X, mixing)  # uses shared T via ScaledGaussian.logpdf
            components, mixing = _m_step_T(X_T, w, resp, T_inv, T)

        return GaussianMixture(components, mixing, T_inv, T)


# ---------- Helpers (operate in scaled space) ----------

def _empirical_mean_cov(samples: np.ndarray, weights: Optional[np.ndarray])-> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, cov) as ML estimates; weights=None => unweighted."""
    X = np.atleast_2d(samples)
    n, d = X.shape
    if weights is None:
        mu = X.mean(axis=0)
        XC = X - mu
        cov = (XC.T @ XC) / n
        return mu, cov
    w = np.asarray(weights, dtype=float).ravel()
    wsum = w.sum()
    if wsum <= 0:
        raise ValueError("Sum of weights must be positive.")
    mu = (w[:, None] * X).sum(axis=0) / wsum
    XC = X - mu
    cov = (XC.T @ (XC * (w[:, None] / wsum)))
    return mu, cov


def _init_components_in_T_space(X_T, labels, K, T_inv, T):
    """Compute (mean_T, cov_T) per k-means cluster; convert to original; wrap as ScaledGaussian with SAME T/T_inv."""
    components = []
    mixing = np.full(K, 1.0 / K, dtype=float)
    n = X_T.shape[0]

    for k in range(K):
        sel = (labels == k)
        Xk_T = X_T[sel]
        if Xk_T.shape[0] == 0:
            # Fallback to global stats if empty cluster
            mu_T = X_T.mean(axis=0)
            XC_T = X_T - mu_T
            cov_T = (XC_T.T @ XC_T) / max(X_T.shape[0], 1)
            nk = 1e-12
        else:
            nk = Xk_T.shape[0]
            mu_T = Xk_T.mean(axis=0)
            XC_T = Xk_T - mu_T
            cov_T = (XC_T.T @ XC_T) / nk

        # Convert back to original to instantiate ScaledGaussian with shared T/T_inv
        mu = (T_inv @ mu_T.T).T
        cov = T_inv @ cov_T @ T_inv.T
        comp = ScaledGaussian(mu, cov, T_inv=T_inv, T=T)  # IMPORTANT: pass T/T_inv to REUSE scaling
        components.append(comp)
        mixing[k] = nk / n

    # Normalize mixing in case of degenerate cluster
    s = mixing.sum()
    if s > 0:
        mixing /= s
    return components, mixing


def _compute_responsibilities_T(components, X_orig, mixing):
    """
    Responsibilities using component logpdf in transformed space.
    ScaledGaussian.logpdf internally transforms X with the shared T.
    """
    X = np.atleast_2d(X_orig)
    n = X.shape[0]
    K = len(components)
    log_probs = np.zeros((n, K))

    for k, comp in enumerate(components):
        log_probs[:, k] = np.log(mixing[k]) + comp.logpdf(X)

    # log-sum-exp normalization
    max_log = np.max(log_probs, axis=1, keepdims=True)
    shifted = np.exp(log_probs - max_log)
    sum_shifted = shifted.sum(axis=1, keepdims=True)
    return np.exp(log_probs - (max_log + np.log(sum_shifted)))


def _m_step_T(X_T, sample_weights, resp, T_inv, T):
    """
    M-step in scaled space.
      - For each k, compute weighted mean_T and cov_T from X_T with weights w_i * r_ik.
      - Convert (mean_T, cov_T) back to original and build ScaledGaussian(mean, cov, T_inv, T).
      - Update mixing = sum(w_i r_ik) / sum(w_i), renormalize at end.
    """
    X_T = np.atleast_2d(X_T)
    w = np.asarray(sample_weights, dtype=float).ravel()
    n, d = X_T.shape
    K = resp.shape[1]

    total_w = w[:, None] * resp  # (n, K)
    wsum = w.sum()

    new_comps = []
    new_mix = np.zeros(K, dtype=float)

    for k in range(K):
        wk = total_w[:, k]
        wk_sum = wk.sum()
        if wk_sum <= 0:
            # Degeneracy guard: use global stats
            mu_T = X_T.mean(axis=0)
            XC_T = X_T - mu_T
            cov_T = (XC_T.T @ XC_T) / max(n, 1)
            new_mix[k] = 1e-12
        else:
            mu_T = (wk[:, None] * X_T).sum(axis=0) / wk_sum
            XC_T = X_T - mu_T
            cov_T = (XC_T.T @ (XC_T * (wk[:, None] / wk_sum)))
            new_mix[k] = wk_sum / wsum

        mu = (T_inv @ mu_T.T).T
        cov = T_inv @ cov_T @ T_inv.T
        new_comps.append(ScaledGaussian(mu, cov, T_inv=T_inv, T=T))

    # Renormalize mixing to 1
    s = new_mix.sum()
    if s > 0:
        new_mix /= s
    else:
        new_mix[:] = 1.0 / K

    return new_comps, new_mix
