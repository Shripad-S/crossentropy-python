"""
crossentropy.cross_entropy
----------------------
Cross‐Entropy method for optimizing importance-sampling distributions.

Given an initial distribution p, a scoring function s(x), and a parametric
update function d(samples, weights), this routine iteratively finds a better
importance-sampler q so that P[s(X) < threshold] can be efficiently estimated.
"""

import numpy as np
from multiprocessing import Pool

cpu_cores = 24
# ——————————————————————————————————————————— Parallel Scoring ——————————————————————————————————————————— #
def parallel_score(samples, score_fn, num_workers=cpu_cores):
    """Evaluates scores in parallel using multiprocessing."""
    with Pool(processes=num_workers) as pool:
        scores = pool.map(score_fn, samples)
    return np.array(scores)


def cross_entropy(
    p,
    update_fn,
    score_fn,
    threshold: float,
    num: int = 10000,
    quantile: float = 0.1,
    max_iters: int = 10,
    parallel: bool = False,
    extras: int = 0
):
    """Run the Cross‐Entropy method.

    Args:
        p: Initial (target) distribution object with methods:
            - p.sample(n): returns array of shape (n, d)
            - p.pdf(points): returns array of shape (n,)
        update_fn: Callable(samples, weights) → new distribution object (weighted MLE fitter)
        score_fn: Callable(x) → scalar score for 1D array x
        threshold: Scalar cutoff for “elite” samples
        num: Number of samples per iteration (default=10000)
        quantile: Fraction in (0, 1) for top‐quantile selection (default=0.1)
        max_iters: Max number of iterations (default=10)
        parallel: If True, attempt parallel scoring (TODO: implement this. For now falls back to serial)
        extras: Number of extra refinement iterations once threshold is met. Defaults to 0.

    Returns:
        q: Final distribution object
        iters: Number of iterations performed (including extras)
        completed: True if threshold was met (i.e., at least quantile*num samples were found meeting threshold), False otherwise
    """
    def get_elites(dist):
        """Draws samples, computes scores, and selects elite subset."""
        samples = dist.sample(num)
        n = samples.shape[0]

        if parallel:
            scores = parallel_score(samples, score_fn)
        else:
            scores = np.zeros(n, dtype=float)
            for i in range(n):
                scores[i] = score_fn(samples[i])

        sorted_scores = np.sort(scores)
        cutoff = sorted_scores[int(np.ceil(quantile * n)) - 1]

        if cutoff > threshold:
            elite_mask = scores < cutoff
            done = False
        else:
            elite_mask = scores < threshold
            done = True

        elites = samples[elite_mask]
        return elites, done

    # First iteration using initial p
    elites, done = get_elites(p)
    weights = np.ones(elites.shape[0], dtype=float)
    q = update_fn(elites, weights)

    if done:
        return q, 1, True

    # Subsequent iterations
    for it in range(2, max_iters + 1):
        elites, done = get_elites(q)
        w_elites = p.pdf(elites) / q.pdf(elites)
        q = update_fn(elites, w_elites)

        if done:
            performed = it
            for _ in range(extras):
                performed += 1
                elites, _ = get_elites(q)
                w_elites = p.pdf(elites) / q.pdf(elites)
                q = update_fn(elites, w_elites)
            return q, performed, True

    # Failed to converge within max_iters
    return q, max_iters, False