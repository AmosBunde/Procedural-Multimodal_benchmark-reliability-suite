"""The hypotheses of PLAN.md, scored from the summary by fixed rules.

The thresholds were written into the plan before the evaluation matrix ran and
are applied here mechanically, so an outcome cannot be reworded after the fact.
A hypothesis with several parts is "supported" only if every part holds,
"failed" if none does, and "partly supported" otherwise.
"""

from __future__ import annotations

from typing import Any


def _outcome(parts: dict[str, bool]) -> str:
    held = sum(bool(v) for v in parts.values())
    if held == len(parts):
        return "supported"
    return "failed" if held == 0 else "partly supported"


def _statuses(block: dict) -> dict[str, str]:
    return {task: info["status"] for task, info in block.items()}


def evaluate_hypotheses(summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    v = summary.get("vision")
    if v is not None:
        n_models = len(v["design"]["models"])
        n_suites = len(v["design"]["suites"])
        status = _statuses(v["floor_ceiling_pooled"])
        floor = sorted(t for t, s in status.items() if s == "floor")
        not_floor = [t for t in v["design"]["tasks"] if status[t] != "floor"]
        informative = [t for t in v["design"]["tasks"] if status[t] == "informative"]

        seeds = v["seeds"]
        parts = {
            "mean_tau_at_least_0.70": seeds["mean_tau"] >= 0.70,
            "same_top_model_in_all_but_one_suite": seeds["n_suites_agreeing_on_top"] >= n_suites - 1,
            "at_most_10_percent_of_cells_inconsistent": seeds["share_cells_rejected"] <= 0.10,
        }
        out["H1"] = {
            "outcome": _outcome(parts),
            "parts": parts,
            "mean_tau": seeds["mean_tau"],
            "n_suites_agreeing_on_top": seeds["n_suites_agreeing_on_top"],
            "share_cells_rejected": seeds["share_cells_rejected"],
        }

        tmpl = v["templates"]
        larger = [m for m in v["design"]["models"] if tmpl["sd"][m] > seeds["sd"][m]]
        need = -(-2 * n_models // 3)  # two thirds of the models, rounded up
        out["H2"] = {
            "outcome": "supported" if len(larger) >= need else "failed",
            "n_models_template_sd_larger": len(larger),
            "n_models": n_models,
            "n_models_required": need,
            "mean_template_sd": tmpl["mean_sd"],
            "mean_seed_sd": seeds["mean_sd"],
        }

        gap = v["ranking_pooled"]["median_adjacent_gap"]
        parts = {
            "mean_template_range_exceeds_median_adjacent_gap": tmpl["mean_range"] > gap,
            "some_template_pair_reorders_models": tmpl["min_tau"] < 1.0,
        }
        out["H3"] = {
            "outcome": _outcome(parts),
            "parts": parts,
            "mean_template_range": tmpl["mean_range"],
            "median_adjacent_gap": gap,
            "min_template_tau": tmpl["min_tau"],
        }

        parts = {"at_least_two_tasks_at_floor": len(floor) >= 2, "spatial_at_floor": "spatial" in floor, "binding_at_floor": "binding" in floor}
        out["H4"] = {"outcome": _outcome(parts), "parts": parts, "tasks_at_floor": floor}

        level1 = v["floor_ceiling_by_level"]["1"]["object"]
        out["H5"] = {
            "outcome": "supported" if level1["at_ceiling"] else "failed",
            "min_accuracy_level_1": level1["min_accuracy"],
            "n_models_at_ceiling_level_1": level1["n_models_at_ceiling"],
            "n_models": n_models,
        }

        ind = v["items_needed"]["independent"]
        per_suite = str(v["design"]["n_items_per_task_per_suite"])
        largest = str(max(int(k) for k in ind))
        parts = {
            "top1_agreement_at_least_0.90": ind[per_suite]["p_same_top1"] >= 0.90,
            "top3_agreement_at_least_0.80": ind[per_suite]["p_same_topk"] >= 0.80,
            "full_order_agreement_below_0.50_at_largest_size": ind[largest]["p_same_order"] < 0.50,
        }
        out["H6"] = {
            "outcome": _outcome(parts),
            "parts": parts,
            "n_per_task": int(per_suite),
            "p_same_top1": ind[per_suite]["p_same_top1"],
            "p_same_top3": ind[per_suite]["p_same_topk"],
            "largest_n_per_task": int(largest),
            "p_same_order_at_largest": ind[largest]["p_same_order"],
        }

        disc = {t: v["items"]["tasks"][t]["mean_discrimination"] for t in v["design"]["tasks"]}
        if floor and informative:
            parts = {
                "floor_tasks_within_0.05_of_zero": all(disc[t] is not None and abs(disc[t]) <= 0.05 for t in floor),
                "informative_tasks_above_0.10": all(disc[t] is not None and disc[t] > 0.10 for t in informative),
            }
            outcome = _outcome(parts)
        else:
            parts, outcome = {}, "not testable"
        out["H7"] = {"outcome": outcome, "parts": parts, "mean_discrimination": disc, "floor_tasks": floor, "informative_tasks": informative}

        mono = {t: v["levels"][t]["monotone"] for t in not_floor}
        out["H8"] = {
            "outcome": "supported" if all(mono.values()) else ("failed" if not any(mono.values()) else "partly supported"),
            "monotone": mono,
            "n_tasks_monotone": sum(mono.values()),
            "n_tasks_tested": len(mono),
        }

    a = summary.get("audio")
    if a is not None:
        status = _statuses(a["floor_ceiling_pooled"])
        floor = sorted(t for t, s in status.items() if s == "floor")
        parts = {"at_least_two_tasks_at_floor": len(floor) >= 2, "order_at_floor": "order" in floor, "tempo_at_floor": "tempo" in floor}
        out["H9"] = {"outcome": _outcome(parts), "parts": parts, "tasks_at_floor": floor}
    return out
