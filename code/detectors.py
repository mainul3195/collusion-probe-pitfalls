"""Run-level detectors over per-agent probe scores s (1-D array, one value per agent).

Each returns (score, coalition_estimate as a frozenset of agent indices).
"""
from itertools import combinations

import numpy as np
from scipy.stats import norm


def peak(s, **_):
    i = int(np.argmax(s))
    return float(s[i]), frozenset([i])


def deception_split(s, k=2, **_):
    """NARCBench Deception Split with a FIXED assumed coalition size k."""
    order = np.argsort(s)[::-1]
    return float(s[order[:k]].mean() - s[order[k:]].mean()), frozenset(order[:k].tolist())


def subsets(n):
    for size in range(1, n):
        for c in combinations(range(n), size):
            yield c


def ca_t(s, sigma0=1.0, size_penalty=0.0, **_):
    """Coalition-agnostic two-sample statistic, maximised over all 2^n - 2 coalitions.

    S(C) = (mean_C - mean_rest) / (sigma0 * sqrt(1/|C| + 1/|rest|)) - size_penalty * |C|
    The sqrt term is the size calibration (phi(|C|) in the proposal): small coalitions
    are not favoured just because a single max is noisier.
    """
    n = len(s); best, arg = -np.inf, None
    for c in subsets(n):
        m = np.zeros(n, bool); m[list(c)] = True
        v = (s[m].mean() - s[~m].mean()) / (sigma0 * np.sqrt(1 / m.sum() + 1 / (~m).sum()))
        v -= size_penalty * m.sum()
        if v > best:
            best, arg = v, frozenset(c)
    return float(best), arg


def ca_glr(s, mu0=0.0, sd0=1.0, mu1=1.0, sd1=1.0, **_):
    """Generalised likelihood ratio: max over C (1<=|C|<=n-1) of sum_{i in C} LLR_i.

    Under H1 colluders draw from N(mu1, sd1), honest from N(mu0, sd0).
    The maximiser includes every agent with positive LLR, clipped to [1, n-1].
    """
    llr = norm.logpdf(s, mu1, sd1) - norm.logpdf(s, mu0, sd0)
    order = np.argsort(llr)[::-1]
    k = int(np.clip((llr > 0).sum(), 1, len(s) - 1))
    return float(llr[order[:k]].sum()), frozenset(order[:k].tolist())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)
