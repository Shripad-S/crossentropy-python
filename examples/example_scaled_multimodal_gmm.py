#!/usr/bin/env python3
"""
example_gmm_multitarget_scaled.py

Cross‐Entropy + Importance Sampling with a Gaussian Mixture proposal targeting
multiple rare-event regions.
Find the probability that a Gaussian random variable lies in any of three small circles.

Targets are circular regions centered at (5, 2.5), (−5, 2), and (5, −2) with radius 0.5.

This version uses:
  - crossentropy.scaledgaussian.ScaledGaussian  (for the base distribution p)
  - crossentropy.scaledGaussianMixture.GaussianMixture (your ScaledGaussian-based mixture)
"""

import numpy as np
import matplotlib.pyplot as plt

from crossentropy.scaledgaussian import ScaledGaussian          
from crossentropy.scaledGaussianMixture import GaussianMixture  
from crossentropy.cross_entropy import cross_entropy
from crossentropy.importance_sampling import importance_sampling


def main():
    # Base distribution p: ScaledGaussian centered at origin
    mean = np.array([0.0, 0.0])
    cov = np.array([[2.0, 0.0],
                    [0.0, 1.0]])
    p = ScaledGaussian(mean, cov) 

    # Define multiple rare-event centers
    centers = [
        np.array([5.0, 2.5]),
        np.array([-5.0, 2.0]),
        np.array([5.0, -2.0])
    ]
    radius = 0.5

    def score_fn(x: np.ndarray) -> float:
        """Minimum distance to any target center."""
        return min(np.linalg.norm(x - c) for c in centers)

    # Cross‐Entropy parameters (kept identical)
    num_samples = 100000
    quantile = 0.25
    max_iters = 10
    parallel = False
    extras = 0
    num_components = 3

    # Update function: weighted MLE of GMM (using your ScaledGaussian-based mixture)
    def update_fn(samples: np.ndarray, weights: np.ndarray) -> GaussianMixture:
        return GaussianMixture.weighted_mle(samples, weights, num_components)

    # Run Cross‐Entropy
    q_dist, n_iters, completed = cross_entropy(
        p=p,
        update_fn=update_fn,
        score_fn=score_fn,
        threshold=radius,
        num=num_samples,
        quantile=quantile,
        max_iters=max_iters,
        parallel=parallel,
        extras=extras
    )

    print(f"Multi-target CE completed in {n_iters} iterations. Success = {completed}")

    # Run Importance Sampling
    mean_est, var_est, RE, interval, samples, scores, hit_mask = importance_sampling(
        p=p,
        q=q_dist,
        score_fn=score_fn,
        threshold=radius,
        num=num_samples,
        parallel=parallel
    )

    print(f"Estimated P(min dist < {radius}): {mean_est}")
    print(f"Variance: {var_est}")
    print(f"Relative error: {RE}")
    print(f"95% confidence interval: {interval}")

    # Plot results (unchanged)
    hit_samples = samples[hit_mask]
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.25, label="All samples")
    plt.scatter(hit_samples[:, 0], hit_samples[:, 1], color="red", s=5, label="Hits")

    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Multi-Target IS with ScaledGaussian Mixture Proposal")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
