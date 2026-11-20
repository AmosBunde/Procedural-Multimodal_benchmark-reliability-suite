"""Statistics checked against hand-worked values and SciPy reference implementations."""

import numpy as np
import pytest
from scipy import stats as sps

from regenbench import stats


def test_wilson_interval_known_values():
    lo, hi = stats.wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=5e-4) and hi == pytest.approx(0.9433, abs=5e-4)
    lo, hi = stats.wilson_interval(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1611, abs=5e-4)
    lo, hi = stats.wilson_interval(50, 100)
    assert (lo + hi) / 2 == pytest.approx(0.5) and hi - lo == pytest.approx(0.1926, abs=5e-4)


def test_wilson_matches_scipy():
    for k, n in [(3, 17), (120, 600), (599, 600)]:
        ref = sps.binomtest(k, n).proportion_ci(method="wilson")
        assert stats.wilson_interval(k, n) == pytest.approx((ref.low, ref.high), abs=1e-9)


def test_binomial_tests_known_values():
    assert stats.binomial_p_greater(8, 10, 0.5) == pytest.approx(56 / 1024)
    assert stats.binomial_p_greater(0, 10, 0.5) == pytest.approx(1.0)
    assert stats.binomial_p_two_sided(8, 10, 0.5) == pytest.approx(112 / 1024)
    for k, n, p in [(37, 120, 0.25), (70, 120, 0.5), (2, 9, 0.5)]:
        assert stats.binomial_p_two_sided(k, n, p) == pytest.approx(sps.binomtest(k, n, p).pvalue, rel=1e-9)
        assert stats.binomial_p_greater(k, n, p) == pytest.approx(sps.binomtest(k, n, p, alternative="greater").pvalue, rel=1e-9)


def test_mcnemar_exact():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert stats.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert stats.mcnemar_exact(9, 3) == pytest.approx(sps.binomtest(9, 12, 0.5).pvalue)


def test_holm_known_values():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_homogeneity_chi2_matches_contingency_test():
    k, n = np.array([30, 45, 38, 41]), np.array([120, 120, 120, 120])
    stat, df, p = stats.homogeneity_chi2(k, n)
    ref = sps.chi2_contingency(np.stack([k, n - k]), correction=False)
    assert stat == pytest.approx(ref.statistic) and df == 3 and p == pytest.approx(ref.pvalue)
    stat, df, p = stats.homogeneity_chi2([10, 10], [10, 10])
    assert stat == 0.0 and np.isnan(p)


def test_average_ranks_and_model_ranks():
    assert stats.average_ranks([10, 20, 20, 5]).tolist() == [2.0, 3.5, 3.5, 1.0]
    assert stats.rank_models([0.9, 0.5, 0.7]).tolist() == [1.0, 3.0, 2.0]
    assert stats.rank_models([0.5, 0.5, 0.2]).tolist() == [1.5, 1.5, 3.0]
    rng = np.random.default_rng(0)
    v = rng.integers(0, 6, 40)
    assert np.allclose(stats.average_ranks(v), sps.rankdata(v))


def test_kendall_tau_known_values_and_scipy():
    assert stats.kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert stats.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert stats.kendall_tau([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(4 / 6)  # five concordant pairs, one discordant
    assert np.isnan(stats.kendall_tau([1, 1, 1], [1, 2, 3]))
    rng = np.random.default_rng(1)
    for _ in range(20):
        x, y = rng.integers(0, 5, 12), rng.integers(0, 5, 12)  # with ties
        assert stats.kendall_tau(x, y) == pytest.approx(sps.kendalltau(x, y).statistic, abs=1e-12)


def test_spearman_known_values_and_scipy():
    assert stats.spearman([1, 2, 3], [10, 20, 35]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)  # 1 - 6*4/(5*24)
    rng = np.random.default_rng(2)
    for _ in range(20):
        x, y = rng.integers(0, 6, 15), rng.normal(size=15)
        assert stats.spearman(x, y) == pytest.approx(sps.spearmanr(x, y).statistic, abs=1e-12)


def test_kr20_known_value():
    # Four respondents, three items. Item p = .75, .5, .25, so sum of pq = .625;
    # totals 3, 2, 1, 0 have variance 1.25; KR-20 = 3/2 * (1 - .625/1.25) = .75.
    x = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]])
    assert stats.kr20(x) == pytest.approx(0.75)
    # Items that are copies of each other are perfectly reliable.
    assert stats.kr20(np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]])) == pytest.approx(1.0)
    assert np.isnan(stats.kr20(np.ones((3, 4))))


def test_kr20_is_near_zero_for_coin_flips_and_high_for_real_ability():
    rng = np.random.default_rng(3)
    noise = rng.random((200, 60)) < 0.5
    assert abs(stats.kr20(noise)) < 0.25
    ability = rng.normal(size=(200, 1))
    signal = rng.random((200, 60)) < 1 / (1 + np.exp(-2 * ability))
    assert stats.kr20(signal) > 0.9


def test_item_rest_correlation():
    x = np.array([[1, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    r = stats.item_rest_correlation(x)
    # Item 0 against the rest score (1, 0, 1, 1) of the other three items.
    assert r[0] == pytest.approx(np.corrcoef([1, 1, 0, 0], [2, 1, 2, 2])[0, 1])
    assert r[2] == pytest.approx(np.corrcoef([0, 0, 1, 1], [3, 2, 1, 1])[0, 1]) and r[2] < 0
    assert np.isnan(r[3]), "an item everyone gets right has no variance"
    ability = np.array([4.0, 3.0, 2.0, 1.0])
    assert stats.item_rest_correlation(x, ability=ability)[0] == pytest.approx(sps.pointbiserialr([1, 1, 0, 0], ability).statistic)


def test_spearman_brown():
    assert stats.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert stats.spearman_brown(0.6, 3) == pytest.approx(1.8 / 2.2)


def test_stratified_bootstrap_recovers_the_standard_error():
    rng = np.random.default_rng(4)
    correct = (rng.random((2, 400)) < np.array([[0.7], [0.4]])).astype(int)
    strata = np.repeat(["a", "b"], 200)
    boot, labels = stats.stratified_bootstrap_means(correct, strata, 4000, np.random.default_rng(5))
    assert labels == ["a", "b"] and boot.shape == (4000, 2, 2)
    for m in range(2):
        for s in range(2):
            p = correct[m, strata == labels[s]].mean()
            assert boot[:, m, s].mean() == pytest.approx(p, abs=0.003)
            assert boot[:, m, s].std() == pytest.approx(np.sqrt(p * (1 - p) / 200), rel=0.06)


def test_bootstrap_can_simulate_a_suite_of_another_size():
    rng = np.random.default_rng(12)
    correct = (rng.random((1, 500)) < 0.3).astype(int)
    p = correct.mean()
    for size in (50, 2000):
        boot, _ = stats.stratified_bootstrap_means(correct, np.zeros(500), 4000, np.random.default_rng(13), size=size)
        assert boot.mean() == pytest.approx(p, abs=0.004)
        assert boot.std() == pytest.approx(np.sqrt(p * (1 - p) / size), rel=0.06)


def test_bootstrap_is_paired_across_models():
    rng = np.random.default_rng(6)
    base = (rng.random(300) < 0.6).astype(int)
    flipped = base.copy()
    flipped[:6] = 1 - flipped[:6]  # the second model differs on six items only
    boot, _ = stats.stratified_bootstrap_means(np.stack([base, flipped]), np.zeros(300), 3000, np.random.default_rng(7))
    diff = boot[:, 0, 0] - boot[:, 1, 0]
    unpaired_sd = np.sqrt(2 * base.mean() * (1 - base.mean()) / 300)
    assert diff.std() < 0.25 * unpaired_sd, "a paired difference of two near-identical models must be tight"


def test_bootstrap_is_reproducible():
    x = (np.random.default_rng(8).random((3, 50)) < 0.5).astype(int)
    a, _ = stats.stratified_bootstrap_means(x, np.zeros(50), 100, np.random.default_rng(9))
    b, _ = stats.stratified_bootstrap_means(x, np.zeros(50), 100, np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_subsample_means():
    x = np.stack([np.ones(40), np.zeros(40), np.tile([1, 0], 20)])
    strata = np.repeat(["a", "b"], 20)
    out = stats.subsample_means(x, strata, 20, 50, np.random.default_rng(10))
    assert out.shape == (50, 3, 2)
    assert np.all(out[:, 0] == 1) and np.all(out[:, 1] == 0) and np.allclose(out[:, 2], 0.5)  # the whole stratum
    part = stats.subsample_means(x, strata, 5, 2000, np.random.default_rng(11))
    assert part[:, 2].mean() == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError):
        stats.subsample_means(x, strata, 21, 5, np.random.default_rng(0))


def test_percentile_interval():
    lo, hi = stats.percentile_interval(np.arange(1001))
    assert lo == pytest.approx(25.0) and hi == pytest.approx(975.0)
