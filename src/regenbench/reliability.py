"""Benchmark-reliability analyses.

Each function takes item-level right/wrong outcomes for several models and
answers one question about the benchmark, not about the models: can its
ranking be trusted, how many items does it need, does a regenerated copy give
the same answer, how much does wording matter, which items and tasks carry
signal, and which tasks say the same thing twice.

Conventions: ``correct`` is a boolean or 0/1 array of shape (models, items);
``tasks`` gives the task of each item; the overall score of a model is the
unweighted mean of its per-task accuracies.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import stats

CEILING = 0.95
ANTI_CORRELATION = -0.4


def task_list(tasks) -> list[str]:
    return list(dict.fromkeys(np.asarray(tasks).tolist()))


def task_accuracy(correct, tasks) -> np.ndarray:
    """Accuracy with shape (models, tasks)."""
    x = np.asarray(correct, dtype=float)
    tasks = np.asarray(tasks)
    return np.stack([x[:, tasks == t].mean(axis=1) for t in task_list(tasks)], axis=1)


def overall_score(correct, tasks) -> np.ndarray:
    return task_accuracy(correct, tasks).mean(axis=1)


# ------------------------------------------------------- (a) ranking stability


def ranking_stability(correct, tasks, models, n_boot: int, rng: np.random.Generator) -> dict:
    """Bootstrap intervals on scores, ranks and paired differences between models."""
    x = np.asarray(correct, dtype=float)
    point = overall_score(x, tasks)
    ranks = stats.rank_models(point)
    boot, _ = stats.stratified_bootstrap_means(x, tasks, n_boot, rng)
    boot_score = boot.mean(axis=2)  # (n_boot, models)
    boot_rank = np.stack([stats.rank_models(row) for row in boot_score])
    lo, hi = stats.percentile_interval(boot_score)
    rlo, rhi = stats.percentile_interval(boot_rank)
    per_model = {}
    for m, name in enumerate(models):
        per_model[name] = {
            "score": float(point[m]),
            "score_lo": float(lo[m]),
            "score_hi": float(hi[m]),
            "rank": float(ranks[m]),
            "rank_lo": float(rlo[m]),
            "rank_hi": float(rhi[m]),
            "p_rank_unchanged": float((boot_rank[:, m] == ranks[m]).mean()),
        }
    pairs = []
    mcnemar = []
    for a, b in itertools.combinations(range(len(models)), 2):
        diff = boot_score[:, a] - boot_score[:, b]
        dlo, dhi = stats.percentile_interval(diff)
        only_a = int(((x[a] == 1) & (x[b] == 0)).sum())
        only_b = int(((x[a] == 0) & (x[b] == 1)).sum())
        mcnemar.append(stats.mcnemar_exact(only_a, only_b))
        observed = point[a] - point[b]
        pairs.append(
            {
                "a": models[a],
                "b": models[b],
                "difference": float(observed),
                "lo": float(dlo),
                "hi": float(dhi),
                "separated": bool(dlo > 0 or dhi < 0),
                "p_same_sign": float((np.sign(diff) == np.sign(observed)).mean()) if observed != 0 else 0.0,
                "only_a_right": only_a,
                "only_b_right": only_b,
            }
        )
    adjusted = stats.holm(mcnemar)
    for pair, p, q in zip(pairs, mcnemar, adjusted):
        pair["mcnemar_p"] = float(p)
        pair["mcnemar_p_holm"] = float(q)
    order = np.argsort(ranks, kind="mergesort")
    gaps = np.abs(np.diff(point[order]))
    return {
        "n_items": int(x.shape[1]),
        "n_boot": int(n_boot),
        "models": per_model,
        "ranking": [models[i] for i in order],
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_pairs_separated": int(sum(p["separated"] for p in pairs)),
        "n_pairs_mcnemar_holm": int((adjusted < 0.05).sum()),
        "adjacent_gaps": [float(g) for g in gaps],
        "median_adjacent_gap": float(np.median(gaps)),
        "mean_tau_to_point_ranking": float(np.mean([stats.kendall_tau(-row, ranks) for row in boot_score])),
        "p_same_top1": float((boot_rank[:, order[0]] == 1).mean()),
        "p_same_order": float(np.all(np.argsort(boot_rank, axis=1, kind="mergesort") == order, axis=1).mean()),
    }


# --------------------------------------------------------- (b) items needed


def _agreement(score_a: np.ndarray, score_b: np.ndarray, top_k: int) -> dict:
    """Agreement between two sets of rankings, each of shape (draws, models)."""
    tau, top1, topk, exact = [], [], [], []
    for a, b in zip(score_a, score_b):
        tau.append(stats.kendall_tau(a, b))
        oa, ob = np.argsort(-a, kind="mergesort"), np.argsort(-b, kind="mergesort")
        top1.append(oa[0] == ob[0])
        topk.append(set(oa[:top_k].tolist()) == set(ob[:top_k].tolist()))
        exact.append(bool(np.all(oa == ob)))
    return {
        "mean_tau": float(np.nanmean(tau)),
        "p_same_top1": float(np.mean(top1)),
        "p_same_topk": float(np.mean(topk)),
        "p_same_order": float(np.mean(exact)),
    }


def items_needed(correct, tasks, sizes, n_draws: int, rng: np.random.Generator, top_k: int = 3) -> dict:
    """How ranking agreement grows with the number of items per task.

    Two kinds of agreement are reported for each size n:

    * ``independent``: two disjoint random sets of n items per task are drawn and their
      rankings compared. This is what two people would see if each generated their own
      suite of that size. It needs 2n items per task, so n stops at half the pool.
    * ``against_pool``: one random set of n items per task against the ranking from the
      whole pool. This is optimistic, because the subset is part of the pool, and reaches
      1 by construction when n equals the pool size.
    """
    x = np.asarray(correct, dtype=float)
    tasks = np.asarray(tasks)
    labels = task_list(tasks)
    pool = min(int((tasks == t).sum()) for t in labels)
    full = overall_score(x, tasks)
    independent, against_pool = {}, {}
    for n in sizes:
        if n > pool:
            continue
        sub = stats.subsample_means(x, tasks, n, n_draws, rng).mean(axis=2)
        against_pool[str(n)] = _agreement(sub, np.broadcast_to(full, sub.shape), top_k)
        if 2 * n <= pool:
            both = np.empty((2, n_draws, x.shape[0], len(labels)))
            for s, t in enumerate(labels):
                block = x[:, tasks == t]
                picks = np.argsort(rng.random((n_draws, block.shape[1])), axis=1)[:, : 2 * n]
                both[0, :, :, s] = block[:, picks[:, :n]].mean(axis=2).T
                both[1, :, :, s] = block[:, picks[:, n:]].mean(axis=2).T
            independent[str(n)] = _agreement(both[0].mean(axis=2), both[1].mean(axis=2), top_k)
    return {"pool_per_task": pool, "n_draws": int(n_draws), "top_k": top_k, "independent": independent, "against_pool": against_pool}


# ------------------------------------------------- (c) seed reproducibility


def seed_reproducibility(correct_by_suite: list, tasks_by_suite: list, models, suite_ids) -> dict:
    """Do regenerated suites give the same scores and the same ranking?"""
    labels = task_list(tasks_by_suite[0])
    acc = np.stack([task_accuracy(c, t) for c, t in zip(correct_by_suite, tasks_by_suite)])  # (suites, models, tasks)
    n = np.array([[int((np.asarray(t) == lab).sum()) for lab in labels] for t in tasks_by_suite])  # (suites, tasks)
    score = acc.mean(axis=2)  # (suites, models)
    sd = score.std(axis=0, ddof=1)
    # What binomial sampling alone predicts for the standard deviation of the overall score.
    pooled = (acc * n[:, None, :]).sum(axis=0) / n.sum(axis=0)  # (models, tasks)
    expected_var = (pooled * (1 - pooled) / n.mean(axis=0)).sum(axis=1) / len(labels) ** 2
    taus = [stats.kendall_tau(score[a], score[b]) for a, b in itertools.combinations(range(len(score)), 2)]
    tops = [models[int(np.argmax(row))] for row in score]
    cells = []
    for m, name in enumerate(models):
        for s, lab in enumerate(labels):
            successes = np.rint(acc[:, m, s] * n[:, s])
            stat, df, p = stats.homogeneity_chi2(successes, n[:, s])
            cells.append({"model": name, "task": lab, "chi2": stat, "df": df, "p": p})
    p_values = np.array([c["p"] for c in cells])
    testable = ~np.isnan(p_values)  # cells where every seed scored 0% or 100% have nothing to test
    observed_var = acc.var(axis=0, ddof=1)  # (models, tasks)
    binomial_var = pooled * (1 - pooled) / n.mean(axis=0)
    usable = binomial_var > 0
    return {
        "suites": list(suite_ids),
        "scores": {name: [float(v) for v in score[:, m]] for m, name in enumerate(models)},
        "sd": {name: float(sd[m]) for m, name in enumerate(models)},
        "sd_expected_from_sampling": {name: float(np.sqrt(expected_var[m])) for m, name in enumerate(models)},
        "range": {name: float(np.ptp(score[:, m])) for m, name in enumerate(models)},
        "mean_sd": float(sd.mean()),
        "max_sd": float(sd.max()),
        "mean_sd_expected": float(np.sqrt(expected_var).mean()),
        "pairwise_tau": [float(t) for t in taus],
        "mean_tau": float(np.mean(taus)),
        "min_tau": float(np.min(taus)),
        "top_model_per_suite": tops,
        "n_suites_agreeing_on_top": int(max(tops.count(t) for t in set(tops))),
        "rankings": [[models[i] for i in np.argsort(-row, kind="mergesort")] for row in score],
        "homogeneity_cells": cells,
        "n_cells": int(testable.sum()),
        "n_cells_rejected": int((p_values[testable] < 0.05).sum()),
        "share_cells_rejected": float((p_values[testable] < 0.05).mean()) if testable.any() else 0.0,
        "variance_ratio": float(observed_var[usable].sum() / binomial_var[usable].sum()),
    }


# --------------------------------------------------- (d) template sensitivity


def template_sensitivity(correct, tasks, models, n_boot: int, rng: np.random.Generator) -> dict:
    """How much the wording of the captions moves scores and ranks.

    ``correct`` has shape (models, items, templates): the same items under each wording.
    """
    x = np.asarray(correct, dtype=float)
    n_models, _, n_templates = x.shape
    labels = task_list(tasks)
    acc = np.stack([task_accuracy(x[:, :, t], tasks) for t in range(n_templates)])  # (templates, models, tasks)
    score = acc.mean(axis=2)  # (templates, models)
    sd = score.std(axis=0, ddof=1)
    spread = np.ptp(score, axis=0)
    taus = [stats.kendall_tau(score[a], score[b]) for a, b in itertools.combinations(range(n_templates), 2)]
    flips = (x.min(axis=2) != x.max(axis=2)).mean(axis=1)  # share of items whose outcome depends on the wording
    contrasts = []
    for t in range(1, n_templates):
        # Paired bootstrap of the change from the canonical wording, same items in both.
        boot, _ = stats.stratified_bootstrap_means(x[:, :, t] - x[:, :, 0], tasks, n_boot, rng)
        diff = boot.mean(axis=2)
        lo, hi = stats.percentile_interval(diff)
        for m, name in enumerate(models):
            contrasts.append(
                {
                    "model": name,
                    "template": t,
                    "change": float(score[t, m] - score[0, m]),
                    "lo": float(lo[m]),
                    "hi": float(hi[m]),
                    "nonzero": bool(lo[m] > 0 or hi[m] < 0),
                }
            )
    task_spread = np.ptp(acc, axis=0)  # (models, tasks)
    return {
        "n_items": int(x.shape[1]),
        "n_templates": int(n_templates),
        "scores": {name: [float(v) for v in score[:, m]] for m, name in enumerate(models)},
        "sd": {name: float(sd[m]) for m, name in enumerate(models)},
        "range": {name: float(spread[m]) for m, name in enumerate(models)},
        "mean_sd": float(sd.mean()),
        "mean_range": float(spread.mean()),
        "max_range": float(spread.max()),
        "pairwise_tau": [float(t) for t in taus],
        "mean_tau": float(np.mean(taus)),
        "min_tau": float(np.min(taus)),
        "rankings": [[models[i] for i in np.argsort(-row, kind="mergesort")] for row in score],
        "flip_rate": {name: float(flips[m]) for m, name in enumerate(models)},
        "mean_flip_rate": float(flips.mean()),
        "contrasts": contrasts,
        "n_contrasts": len(contrasts),
        "n_contrasts_nonzero": int(sum(c["nonzero"] for c in contrasts)),
        "task_accuracy": {lab: {name: [float(v) for v in acc[:, m, s]] for m, name in enumerate(models)} for s, lab in enumerate(labels)},
        "task_mean_range": {lab: float(task_spread[:, s].mean()) for s, lab in enumerate(labels)},
        "task_max_range": {lab: float(task_spread[:, s].max()) for s, lab in enumerate(labels)},
    }


# ------------------------------------------------------------ (e) item analysis


def null_share_below(ability, k: int, threshold: float) -> float:
    """Exact chance that an item answered correctly by a random k of the models has an
    item-ability correlation at or below ``threshold``. Every choice of k models is listed."""
    ability = np.asarray(ability, dtype=float)
    m = len(ability)
    hits = total = 0
    for subset in itertools.combinations(range(m), k):
        col = np.zeros(m)
        col[list(subset)] = 1
        hits += stats.pearson(col, ability) <= threshold
        total += 1
    return hits / total


def item_analysis(correct, tasks, item_ids, n_split: int, rng: np.random.Generator) -> dict:
    """Classical item statistics, with models in the role of examinees."""
    x = np.asarray(correct, dtype=float)
    tasks = np.asarray(tasks)
    n_models = x.shape[0]
    ability = overall_score(x, tasks)
    difficulty = x.mean(axis=0)
    n_right = x.sum(axis=0).astype(int)
    # Ability is the overall score. One item among thousands moves it by a negligible amount,
    # so the item is not removed from it, which keeps the permutation null below exact.
    disc = stats.item_rest_correlation(x, ability=ability)
    null = {k: null_share_below(ability, k, ANTI_CORRELATION) for k in range(1, n_models)}
    per_task = {}
    for t in task_list(tasks):
        sel = tasks == t
        d = disc[sel]
        varied = ~np.isnan(d)
        halves = []
        block = x[:, sel]
        for _ in range(n_split):
            perm = rng.permutation(block.shape[1])
            a, b = block[:, perm[::2]].mean(axis=1), block[:, perm[1::2]].mean(axis=1)
            halves.append(stats.pearson(a, b))
        half_r = float(np.nanmean(halves)) if np.any(~np.isnan(halves)) else float("nan")
        expected_anti = float(sum(null[k] * int((n_right[sel] == k).sum()) for k in null))
        per_task[t] = {
            "n_items": int(sel.sum()),
            "n_none_right": int((n_right[sel] == 0).sum()),
            "n_all_right": int((n_right[sel] == n_models).sum()),
            "n_varied": int(varied.sum()),
            "mean_difficulty": float(difficulty[sel].mean()),
            "mean_discrimination": float(d[varied].mean()) if varied.any() else float("nan"),
            "n_anti_correlated": int((d[varied] <= ANTI_CORRELATION).sum()),
            "expected_anti_correlated_by_chance": expected_anti,
            "kr20": stats.kr20(block),
            "split_half_r": half_r,
            "split_half_reliability": stats.spearman_brown(half_r) if np.isfinite(half_r) else float("nan"),
        }
    varied = ~np.isnan(disc)
    worst = np.argsort(np.where(varied, disc, np.inf), kind="mergesort")[:10]
    return {
        "n_models": int(n_models),
        "n_items": int(x.shape[1]),
        "anti_correlation_threshold": ANTI_CORRELATION,
        "tasks": per_task,
        "n_none_right": int((n_right == 0).sum()),
        "n_all_right": int((n_right == n_models).sum()),
        "n_varied": int(varied.sum()),
        "n_anti_correlated": int((disc[varied] <= ANTI_CORRELATION).sum()),
        "expected_anti_correlated_by_chance": float(sum(v["expected_anti_correlated_by_chance"] for v in per_task.values())),
        "kr20_all_items": stats.kr20(x),
        "most_anti_correlated": [
            {"item": str(item_ids[i]), "discrimination": float(disc[i]), "n_right": int(n_right[i])} for i in worst if varied[i]
        ],
        "difficulty": difficulty,
        "discrimination": disc,
    }


# ------------------------------------------------- (f) floor and ceiling checks


def floor_ceiling(correct, tasks, n_choices, models, n_boot: int, rng: np.random.Generator) -> dict:
    """Per task: is any model above chance, are all at ceiling, and can the task separate models?"""
    x = np.asarray(correct, dtype=float)
    tasks = np.asarray(tasks)
    n_choices = np.asarray(n_choices)
    out = {}
    for t in task_list(tasks):
        sel = tasks == t
        block = x[:, sel]
        n = int(sel.sum())
        chance = float((1.0 / n_choices[sel]).mean())
        k = block.sum(axis=1).astype(int)
        p_above = stats.holm([stats.binomial_p_greater(int(v), n, chance) for v in k])
        rows = {}
        for m, name in enumerate(models):
            lo, hi = stats.wilson_interval(int(k[m]), n)
            rows[name] = {
                "accuracy": float(k[m] / n),
                "lo": lo,
                "hi": hi,
                "p_above_chance_holm": float(p_above[m]),
                "above_chance": bool(p_above[m] < 0.05),
                "below_chance": bool(hi < chance),
                "at_ceiling": bool(k[m] / n >= CEILING),
            }
        boot, _ = stats.stratified_bootstrap_means(block, np.zeros(n, dtype=int), n_boot, rng)
        acc_boot = boot[:, :, 0]
        separated = 0
        for a, b in itertools.combinations(range(len(models)), 2):
            lo, hi = stats.percentile_interval(acc_boot[:, a] - acc_boot[:, b])
            separated += bool(lo > 0 or hi < 0)
        acc = k / n
        at_floor = not any(r["above_chance"] for r in rows.values())
        at_ceiling = all(r["at_ceiling"] for r in rows.values())
        out[t] = {
            "n_items": n,
            "chance": chance,
            "models": rows,
            "min_accuracy": float(acc.min()),
            "max_accuracy": float(acc.max()),
            "mean_accuracy": float(acc.mean()),
            "spread": float(acc.max() - acc.min()),
            "n_models_above_chance": int(sum(r["above_chance"] for r in rows.values())),
            "n_models_below_chance": int(sum(r["below_chance"] for r in rows.values())),
            "n_models_at_ceiling": int(sum(r["at_ceiling"] for r in rows.values())),
            "at_floor": bool(at_floor),
            "at_ceiling": bool(at_ceiling),
            "n_pairs_separated": int(separated),
            "n_pairs": len(models) * (len(models) - 1) // 2,
            "status": "floor" if at_floor else ("ceiling" if at_ceiling else "informative"),
        }
    return out


def accuracy_by_level(correct, tasks, levels, models) -> dict:
    """Accuracy per task and difficulty level, per model and averaged over models."""
    x = np.asarray(correct, dtype=float)
    tasks, levels = np.asarray(tasks), np.asarray(levels)
    out = {}
    for t in task_list(tasks):
        per_level = np.stack([x[:, (tasks == t) & (levels == lv)].mean(axis=1) for lv in (1, 2, 3)], axis=1)  # (models, 3)
        mean = per_level.mean(axis=0)
        out[t] = {
            "n_per_level": int(((tasks == t) & (levels == 1)).sum()),
            "models": {name: [float(v) for v in per_level[m]] for m, name in enumerate(models)},
            "mean": [float(v) for v in mean],
            "monotone": bool(mean[0] > mean[1] > mean[2]),
            "min_per_level": [float(v) for v in per_level.min(axis=0)],
            "max_per_level": [float(v) for v in per_level.max(axis=0)],
        }
    return out


# ------------------------------------------------------- (g) task redundancy


def task_redundancy(correct, tasks, models, n_boot: int, rng: np.random.Generator) -> dict:
    """Do tasks order the models the same way, and does the overall ranking need each task?"""
    x = np.asarray(correct, dtype=float)
    labels = task_list(tasks)
    acc = task_accuracy(x, tasks)
    boot, _ = stats.stratified_bootstrap_means(x, tasks, n_boot, rng)
    pairs = []
    for a, b in itertools.combinations(range(len(labels)), 2):
        rho = stats.spearman(acc[:, a], acc[:, b])
        draws = np.array([stats.spearman(row[:, a], row[:, b]) for row in boot])
        draws = draws[~np.isnan(draws)]
        lo, hi = stats.percentile_interval(draws) if len(draws) else (np.nan, np.nan)
        pairs.append({"a": labels[a], "b": labels[b], "spearman": float(rho), "lo": float(lo), "hi": float(hi)})
    full = acc.mean(axis=1)
    leave_out = {}
    for s, lab in enumerate(labels):
        rest = np.delete(acc, s, axis=1).mean(axis=1)
        leave_out[lab] = {
            "tau_to_full_ranking": stats.kendall_tau(full, rest),
            "same_top1": bool(np.argmax(full) == np.argmax(rest)),
            "task_vs_rest_spearman": stats.spearman(acc[:, s], rest),
        }
    return {"tasks": labels, "pairs": pairs, "leave_one_out": leave_out}
