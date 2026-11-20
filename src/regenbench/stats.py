"""Statistics used by the reliability analysis.

Everything here is written from NumPy primitives, with SciPy used only for
distribution functions. The tests check each function against values worked
out by hand or against the SciPy reference implementation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _dist


# ---------------------------------------------------------------- proportions


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(_dist.norm.ppf(0.5 + confidence / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def binomial_p_greater(k: int, n: int, p0: float) -> float:
    """Exact one-sided p-value: probability of k or more successes when the true rate is p0."""
    return float(_dist.binom.sf(k - 1, n, p0))


def binomial_p_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided p-value, summing every outcome no more likely than the observed one."""
    pmf = _dist.binom.pmf(np.arange(n + 1), n, p0)
    return float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-9)].sum()))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar test for paired right/wrong outcomes.

    ``b`` is the number of items only the first model got right and ``c`` the number only
    the second got right. Items both got right or both got wrong carry no information.
    """
    if b + c == 0:
        return 1.0
    return binomial_p_two_sided(b, b + c, 0.5)


def holm(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def homogeneity_chi2(successes, totals) -> tuple[float, int, float]:
    """Pearson chi-square test that several binomial samples share one success rate.

    Returns (statistic, degrees of freedom, p-value). With a pooled rate of 0 or 1 there
    is nothing to test and the p-value is NaN.
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    pooled = k.sum() / n.sum()
    df = len(k) - 1
    if pooled <= 0.0 or pooled >= 1.0:
        return 0.0, df, float("nan")
    stat = float((((k - n * pooled) ** 2) / (n * pooled * (1 - pooled))).sum())
    return stat, df, float(_dist.chi2.sf(stat, df))


# ---------------------------------------------------------------------- ranks


def average_ranks(values) -> np.ndarray:
    """Ranks starting at 1 for the smallest value; tied values share their mean rank."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def rank_models(scores) -> np.ndarray:
    """Rank 1 for the highest score; ties share their mean rank."""
    return average_ranks(-np.asarray(scores, dtype=float))


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    return float((xc * yc).sum() / denom) if denom > 0 else float("nan")


def spearman(x, y) -> float:
    """Spearman rank correlation, with average ranks for ties."""
    return pearson(average_ranks(x), average_ranks(y))


def kendall_tau(x, y) -> float:
    """Kendall's tau-b, which corrects for ties in either variable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i, j = np.triu_indices(len(x), k=1)
    sx = np.sign(x[i] - x[j])
    sy = np.sign(y[i] - y[j])
    denom = np.sqrt((sx != 0).sum() * (sy != 0).sum())
    return float((sx * sy).sum() / denom) if denom > 0 else float("nan")


# ------------------------------------------------------------ item statistics


def kr20(responses) -> float:
    """Kuder-Richardson formula 20 for a (respondents, items) matrix of 0/1 responses.

    The reliability of the total score: 1 means differences between respondents are
    all signal, 0 means they are what independent coin flips would give.
    """
    x = np.asarray(responses, dtype=float)
    k = x.shape[1]
    p = x.mean(axis=0)
    total_var = x.sum(axis=1).var()
    if k < 2 or total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - (p * (1 - p)).sum() / total_var))


def item_rest_correlation(responses, ability=None) -> np.ndarray:
    """Correlation of each item with ability, across respondents.

    By default ability is the rest score: the respondent's total on all other items,
    so an item is never correlated with itself. Items that every respondent answers
    the same way have no variance and get NaN.
    """
    x = np.asarray(responses, dtype=float)
    total = x.sum(axis=1)
    out = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        col = x[:, j]
        if col.min() == col.max():
            continue
        out[j] = pearson(col, total - col if ability is None else ability)
    return out


def spearman_brown(r: float, factor: float = 2.0) -> float:
    """Reliability of a test ``factor`` times as long, from the reliability ``r`` of the current one."""
    return float(factor * r / (1 + (factor - 1) * r))


# ------------------------------------------------------------------ bootstrap


def stratified_bootstrap_means(
    correct, strata, n_boot: int, rng: np.random.Generator, size: int | None = None
) -> tuple[np.ndarray, list]:
    """Bootstrap the per-stratum accuracy of several models over the same items.

    ``correct`` is (models, items) of 0/1 and ``strata`` gives the stratum (task) of each
    item. Items are resampled with replacement inside each stratum, and the same resample
    is applied to every model, which keeps the comparison paired. ``size`` sets how many
    items are drawn per stratum (default: as many as the stratum holds), which simulates a
    suite of another size. Returns an array of shape (n_boot, models, strata) and the list
    of stratum labels.
    """
    x = np.asarray(correct, dtype=float)
    strata = np.asarray(strata)
    labels = list(dict.fromkeys(strata.tolist()))
    out = np.empty((n_boot, x.shape[0], len(labels)))
    for s, label in enumerate(labels):
        block = x[:, strata == label]
        n = block.shape[1]
        drawn = size or n
        counts = rng.multinomial(drawn, np.full(n, 1.0 / n), size=n_boot)  # how often each item is drawn
        out[:, :, s] = counts @ block.T / drawn
    return out, labels


def percentile_interval(samples, confidence: float = 0.95, axis: int = 0):
    lo = (1 - confidence) / 2 * 100
    return np.percentile(samples, lo, axis=axis), np.percentile(samples, 100 - lo, axis=axis)


def subsample_means(correct, strata, n_per_stratum: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Per-stratum accuracy on random subsets of ``n_per_stratum`` items, drawn without replacement.

    Returns an array of shape (n_draws, models, strata).
    """
    x = np.asarray(correct, dtype=float)
    strata = np.asarray(strata)
    labels = list(dict.fromkeys(strata.tolist()))
    out = np.empty((n_draws, x.shape[0], len(labels)))
    for s, label in enumerate(labels):
        block = x[:, strata == label]
        n = block.shape[1]
        if n_per_stratum > n:
            raise ValueError(f"asked for {n_per_stratum} items but stratum {label!r} has {n}")
        picks = np.argsort(rng.random((n_draws, n)), axis=1)[:, :n_per_stratum]
        out[:, :, s] = block[:, picks].mean(axis=2).T
    return out
