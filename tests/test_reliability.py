"""Reliability analyses on synthetic outcomes where the right answer is known."""

import numpy as np
import pytest

from regenbench import reliability as rel


def simulate(rates, n_per_task, rng, tasks=("a", "b", "c")):
    """Independent right/wrong outcomes: rates[m][t] is model m's true accuracy on task t."""
    rates = np.asarray(rates, dtype=float)
    if rates.ndim == 1:
        rates = np.repeat(rates[:, None], len(tasks), axis=1)
    labels = np.repeat(tasks, n_per_task)
    correct = np.concatenate([rng.random((len(rates), n_per_task)) < rates[:, [t]] for t in range(len(tasks))], axis=1)
    return correct.astype(int), labels


def test_task_accuracy_and_overall_score():
    correct = np.array([[1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0]])
    tasks = np.array(["a", "a", "a", "a", "b", "b"])
    assert rel.task_accuracy(correct, tasks).tolist() == [[0.5, 1.0], [0.0, 0.5]]
    # The overall score weights tasks equally, not items.
    assert rel.overall_score(correct, tasks).tolist() == [0.75, 0.25]


def test_ranking_stability_separates_clear_gaps_and_not_ties():
    rng = np.random.default_rng(0)
    correct, tasks = simulate([0.9, 0.6, 0.6, 0.3], 300, rng)
    out = rel.ranking_stability(correct, tasks, ["top", "mid1", "mid2", "low"], 1000, np.random.default_rng(1))
    assert out["ranking"][0] == "top" and out["ranking"][-1] == "low"
    assert out["models"]["top"]["rank_lo"] == out["models"]["top"]["rank_hi"] == 1
    assert out["models"]["mid1"]["rank_lo"] == 2 and out["models"]["mid1"]["rank_hi"] == 3
    pairs = {(p["a"], p["b"]): p for p in out["pairs"]}
    assert pairs[("top", "low")]["separated"] and pairs[("top", "mid1")]["separated"]
    assert not pairs[("mid1", "mid2")]["separated"]
    assert out["n_pairs"] == 6 and out["n_pairs_separated"] == 5
    assert out["p_same_top1"] == 1.0 and out["p_same_order"] < 1.0
    assert out["models"]["mid1"]["p_rank_unchanged"] == out["p_same_order"]
    assert pairs[("top", "low")]["mcnemar_p_holm"] < 1e-6 and pairs[("mid1", "mid2")]["mcnemar_p"] > 0.05
    for name, row in out["models"].items():
        assert row["score_lo"] <= row["score"] <= row["score_hi"]


def test_ranking_interval_covers_the_truth_at_the_nominal_rate():
    truth, covered = 0.6, 0
    for seed in range(60):
        correct, tasks = simulate([truth, 0.5], 150, np.random.default_rng(100 + seed))
        row = rel.ranking_stability(correct, tasks, ["m", "other"], 400, np.random.default_rng(seed))["models"]["m"]
        covered += row["score_lo"] <= truth <= row["score_hi"]
    assert covered >= 51, f"a 95% interval covered the truth in only {covered} of 60 simulations"


def test_items_needed_grows_with_size():
    rng = np.random.default_rng(2)
    correct, tasks = simulate([0.70, 0.66, 0.62, 0.58], 400, rng)
    out = rel.items_needed(correct, tasks, (10, 50, 200, 400), 300, np.random.default_rng(3), top_k=2)
    ind, pool = out["independent"], out["against_pool"]
    assert set(ind) == {"10", "50", "200"} and set(pool) == {"10", "50", "200", "400"}
    assert ind["10"]["mean_tau"] < ind["50"]["mean_tau"] < ind["200"]["mean_tau"]
    assert ind["10"]["p_same_top1"] < ind["200"]["p_same_top1"]
    assert pool["400"]["mean_tau"] == pytest.approx(1.0) and pool["400"]["p_same_order"] == 1.0
    # Comparing with the pool that contains the subset flatters the benchmark.
    assert pool["200"]["mean_tau"] > ind["200"]["mean_tau"]


def test_items_needed_is_trivial_for_widely_separated_models():
    correct, tasks = simulate([0.95, 0.5, 0.05], 200, np.random.default_rng(4))
    out = rel.items_needed(correct, tasks, (20, 100), 200, np.random.default_rng(5))
    assert out["independent"]["20"]["p_same_order"] == 1.0


def test_seed_reproducibility_accepts_fresh_draws_and_rejects_a_changed_generator():
    rates = [[0.8, 0.5, 0.3], [0.6, 0.5, 0.3]]
    suites = [simulate(rates, 400, np.random.default_rng(10 + s)) for s in range(5)]
    ok = rel.seed_reproducibility([c for c, _ in suites], [t for _, t in suites], ["m1", "m2"], list("vwxyz"))
    assert ok["n_cells"] == 6 and ok["share_cells_rejected"] <= 1 / 6
    assert 0.3 < ok["variance_ratio"] < 2.5
    assert ok["mean_tau"] == 1.0 and ok["n_suites_agreeing_on_top"] == 5
    assert ok["sd"]["m1"] == pytest.approx(ok["sd_expected_from_sampling"]["m1"], rel=0.9)

    # One "regenerated" suite is in fact much harder: the check must notice.
    drifted = suites[:4] + [simulate([[0.5, 0.5, 0.3], [0.3, 0.5, 0.3]], 400, np.random.default_rng(99))]
    bad = rel.seed_reproducibility([c for c, _ in drifted], [t for _, t in drifted], ["m1", "m2"], list("vwxyz"))
    rejected = {(c["model"], c["task"]) for c in bad["homogeneity_cells"] if c["p"] < 0.05}
    assert {("m1", "a"), ("m2", "a")} <= rejected
    assert bad["variance_ratio"] > ok["variance_ratio"]


def test_template_sensitivity_zero_for_identical_wordings():
    correct, tasks = simulate([0.8, 0.5], 100, np.random.default_rng(20))
    same = np.repeat(correct[:, :, None], 3, axis=2)
    out = rel.template_sensitivity(same, tasks, ["m1", "m2"], 200, np.random.default_rng(21))
    assert out["mean_sd"] == 0 and out["mean_range"] == 0 and out["mean_flip_rate"] == 0
    assert out["n_contrasts_nonzero"] == 0 and out["mean_tau"] == 1.0


def test_template_sensitivity_detects_a_wording_that_hurts_one_model():
    rng = np.random.default_rng(22)
    correct, tasks = simulate([0.8, 0.6], 200, rng)
    x = np.repeat(correct[:, :, None], 3, axis=2)
    x[0, :, 2] = rng.random(600) < 0.4  # the third wording breaks the stronger model
    out = rel.template_sensitivity(x, tasks, ["m1", "m2"], 500, np.random.default_rng(23))
    assert out["range"]["m1"] > 0.3 and out["range"]["m2"] == 0
    assert out["rankings"][0] == ["m1", "m2"] and out["rankings"][2] == ["m2", "m1"]
    assert out["min_tau"] == -1.0
    hurt = [c for c in out["contrasts"] if c["nonzero"]]
    assert [(c["model"], c["template"]) for c in hurt] == [("m1", 2)] and hurt[0]["hi"] < 0
    assert out["flip_rate"]["m1"] > 0.3 and out["flip_rate"]["m2"] == 0


def test_null_share_below_is_exact():
    ability = [4.0, 3.0, 2.0, 1.0]
    # One model right: the correlation is negative when it is one of the two weakest models.
    assert rel.null_share_below(ability, 1, -0.01) == pytest.approx(0.5)
    assert rel.null_share_below(ability, 1, -2.0) == 0.0
    assert rel.null_share_below(ability, 2, 2.0) == 1.0


def test_item_analysis_counts_and_flags():
    rng = np.random.default_rng(30)
    ability = np.linspace(0.9, 0.3, 6)
    good, tasks_good = simulate(ability, 300, rng, tasks=("signal",))
    noise, tasks_noise = simulate(np.full(6, 0.5), 300, rng, tasks=("noise",))
    planted = np.array([[0], [0], [0], [1], [1], [1]])  # the three weakest models get it right
    everyone = np.ones((6, 1), dtype=int)
    nobody = np.zeros((6, 1), dtype=int)
    correct = np.concatenate([good, noise, planted, everyone, nobody], axis=1)
    tasks = np.concatenate([tasks_good, tasks_noise, ["signal"] * 3])
    ids = np.array([f"i{k}" for k in range(correct.shape[1])])
    out = rel.item_analysis(correct, tasks, ids, 50, np.random.default_rng(31))
    n_right = correct.sum(axis=0)
    assert out["n_all_right"] == (n_right == 6).sum() >= 1
    assert out["n_none_right"] == (n_right == 0).sum() >= 1
    assert out["n_varied"] == ((n_right > 0) & (n_right < 6)).sum()
    assert out["discrimination"][-3] < -0.7, "the planted item runs against ability"
    assert out["most_anti_correlated"][0]["discrimination"] <= out["discrimination"][-3]
    assert np.isnan(out["discrimination"][-1]) and np.isnan(out["discrimination"][-2])
    assert out["difficulty"][-3:].tolist() == [0.5, 1.0, 0.0]
    # Items that follow ability discriminate; coin flips do not.
    assert out["tasks"]["signal"]["mean_discrimination"] > 0.2
    assert abs(out["tasks"]["noise"]["mean_discrimination"]) < 0.06
    assert out["tasks"]["signal"]["kr20"] > 0.9 and out["tasks"]["noise"]["kr20"] < 0.5
    assert out["tasks"]["signal"]["split_half_reliability"] > 0.9
    # For coin-flip items the number of anti-correlated items is what chance predicts.
    n, expected = out["tasks"]["noise"]["n_anti_correlated"], out["tasks"]["noise"]["expected_anti_correlated_by_chance"]
    assert abs(n - expected) < 4 * np.sqrt(expected)
    assert out["tasks"]["signal"]["n_anti_correlated"] < 0.5 * out["tasks"]["signal"]["expected_anti_correlated_by_chance"]


def test_floor_ceiling_status():
    rng = np.random.default_rng(40)
    rates = [[0.25, 0.99, 0.9, 0.10], [0.26, 0.98, 0.5, 0.25], [0.24, 1.0, 0.3, 0.25]]
    correct, tasks = simulate(rates, 400, rng, tasks=("floor", "ceiling", "useful", "biased"))
    n_choices = np.full(len(tasks), 4)
    out = rel.floor_ceiling(correct, tasks, n_choices, ["m1", "m2", "m3"], 500, np.random.default_rng(41))
    assert [out[t]["status"] for t in ("floor", "ceiling", "useful", "biased")] == ["floor", "ceiling", "informative", "floor"]
    assert out["floor"]["chance"] == 0.25 and out["floor"]["n_models_above_chance"] == 0
    assert out["useful"]["n_pairs_separated"] == 3 and out["floor"]["n_pairs_separated"] <= 1
    assert out["ceiling"]["n_models_at_ceiling"] == 3
    assert out["biased"]["models"]["m1"]["below_chance"] and out["biased"]["n_models_below_chance"] == 1
    assert out["useful"]["models"]["m3"]["above_chance"]  # 30% against a chance level of 25%, n = 400


def test_floor_uses_the_chance_level_of_each_task():
    rng = np.random.default_rng(42)
    correct, tasks = simulate([[0.5, 0.5]], 400, rng, tasks=("two", "four"))
    n_choices = np.where(tasks == "two", 2, 4)
    out = rel.floor_ceiling(correct, tasks, n_choices, ["m"], 200, np.random.default_rng(43))
    assert out["two"]["chance"] == 0.5 and out["two"]["status"] == "floor"
    assert out["four"]["chance"] == 0.25 and out["four"]["status"] == "informative"


def test_accuracy_by_level():
    rng = np.random.default_rng(50)
    tasks = np.repeat(["t"], 600)
    levels = np.repeat([1, 2, 3], 200)
    correct = np.concatenate([rng.random((2, 200)) < p for p in (0.9, 0.6, 0.3)], axis=1)
    out = rel.accuracy_by_level(correct, tasks, levels, ["m1", "m2"])["t"]
    assert out["monotone"] and out["n_per_level"] == 200
    assert out["mean"] == pytest.approx([0.9, 0.6, 0.3], abs=0.06)
    flat = rel.accuracy_by_level(correct[:, ::-1], tasks, levels, ["m1", "m2"])["t"]
    assert not flat["monotone"]


def test_task_redundancy():
    rng = np.random.default_rng(60)
    ability = np.array([0.9, 0.75, 0.6, 0.45, 0.3])
    rates = np.stack([ability, ability, ability[::-1]], axis=1)  # the third task orders models the other way
    correct, tasks = simulate(rates, 500, rng, tasks=("a", "copy", "reversed"))
    out = rel.task_redundancy(correct, tasks, list("vwxyz"), 200, np.random.default_rng(61))
    pairs = {(p["a"], p["b"]): p for p in out["pairs"]}
    assert pairs[("a", "copy")]["spearman"] == pytest.approx(1.0)
    assert pairs[("a", "reversed")]["spearman"] == pytest.approx(-1.0)
    assert pairs[("a", "copy")]["lo"] > 0.5
    assert out["leave_one_out"]["reversed"]["tau_to_full_ranking"] == pytest.approx(1.0)
    assert out["leave_one_out"]["reversed"]["task_vs_rest_spearman"] == pytest.approx(-1.0)
