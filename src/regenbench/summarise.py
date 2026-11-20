"""Turn item-level result files into ``results/summary.json``.

Every number quoted in the README, the report and the HTML health report is
read from that file. The analysis is deterministic: the random generators used
for bootstrapping and subsampling are seeded from fixed constants.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import reliability, study
from .evaluate import load_result
from .models import MODELS
from .names import validate_name

ANALYSIS_SEED = 20261120
N_BOOT = 2000
N_DRAWS = 1000
N_SPLIT = 200
MODALITY_OF = {"vision": "image", "audio": "audio"}


@dataclass
class RunData:
    modality: str
    models: list[str]
    suites: list[str]
    seeds: list[int]
    results: dict[str, dict[str, dict]]  # suite -> model -> loaded result file

    def correct(self, suite: str) -> np.ndarray:
        """(models, items, templates)"""
        return np.stack([self.results[suite][m]["correct"] for m in self.models])

    def field(self, suite: str, name: str) -> np.ndarray:
        return self.results[suite][self.models[0]][name]

    @property
    def main(self) -> str:
        return self.suites[self.seeds.index(study.MAIN_SEED)] if study.MAIN_SEED in self.seeds else self.suites[0]

    def pooled(self, name: str) -> np.ndarray:
        return np.concatenate([self.field(s, name) for s in self.suites])

    def pooled_correct(self) -> np.ndarray:
        """(models, all items), canonical template."""
        return np.concatenate([self.correct(s)[:, :, 0] for s in self.suites], axis=1)


def load_run(results_root: Path, run_id: str) -> dict[str, RunData]:
    validate_name(run_id, "run id")
    directory = Path(results_root) / "runs" / run_id
    by_modality: dict[str, dict[str, dict[str, dict]]] = {}
    for path in sorted(directory.glob("*.npz")):
        res = load_result(path)
        info = res["info"]
        by_modality.setdefault(info["modality"], {}).setdefault(info["suite_id"], {})[info["model_key"]] = res
    out = {}
    for modality, suites in by_modality.items():
        ordered = sorted(suites, key=lambda s: (next(iter(suites[s].values()))["info"]["suite_seed"], s))
        common = set.intersection(*(set(suites[s]) for s in ordered))
        preferred = [m for m in study.models_for(modality) if m in common]
        models = preferred + sorted(common - set(preferred))
        for s in ordered:  # every model must have seen exactly the same items
            shas = {suites[s][m]["info"]["suite_sha256"] for m in models}
            if len(shas) != 1:
                raise ValueError(f"suite {s}: models were evaluated on different versions of the suite")
        seeds = [int(suites[s][models[0]]["info"]["suite_seed"]) for s in ordered]
        out[modality] = RunData(modality, models, ordered, seeds, suites)
    return out


def _clean(obj: Any) -> Any:
    """Make a nested structure JSON-safe: plain types, NaN as null, floats rounded."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(obj) else round(float(obj), 6)
    return obj


def _rng(*key: int) -> np.random.Generator:
    return np.random.default_rng([ANALYSIS_SEED, *key])


def analyse(data: RunData) -> dict:
    m = data.models
    tasks_pool, levels_pool, choices_pool = data.pooled("task"), data.pooled("level"), data.pooled("n_choices")
    pool = data.pooled_correct()
    main = data.correct(data.main)
    tasks_main = data.field(data.main, "task")
    item_ids = np.concatenate([[f"{s}/{i}" for i in data.field(s, "item_id")] for s in data.suites])
    k = 0 if data.modality == "vision" else 100
    items = reliability.item_analysis(pool, tasks_pool, item_ids, N_SPLIT, _rng(k, 5))
    item_arrays = {"difficulty": items.pop("difficulty"), "discrimination": items.pop("discrimination")}
    seconds = {mk: sum(data.results[s][mk]["info"]["seconds"] for s in data.suites) for mk in m}
    out = {
        "design": {
            "models": m,
            "suites": data.suites,
            "seeds": data.seeds,
            "main_suite": data.main,
            "tasks": reliability.task_list(tasks_pool),
            "n_items_per_suite": int(len(tasks_main)),
            "n_items_per_task_per_suite": int((tasks_main == tasks_main[0]).sum()),
            "n_items_pooled": int(pool.shape[1]),
            "n_items_per_task_pooled": int((tasks_pool == tasks_pool[0]).sum()),
            "n_templates_main": int(main.shape[2]),
            "suite_sha256": {s: data.results[s][m[0]]["info"]["suite_sha256"] for s in data.suites},
            "eval_seconds": seconds,
            "eval_seconds_total": float(sum(seconds.values())),
        },
        "task_accuracy_pooled": {
            name: dict(zip(reliability.task_list(tasks_pool), row.tolist()))
            for name, row in zip(m, reliability.task_accuracy(pool, tasks_pool))
        },
        "ranking_pooled": reliability.ranking_stability(pool, tasks_pool, m, N_BOOT, _rng(k, 1)),
        "ranking_main": reliability.ranking_stability(main[:, :, 0], tasks_main, m, N_BOOT, _rng(k, 2)),
        "items_needed": reliability.items_needed(
            pool,
            tasks_pool,
            study.SIZES[data.modality],
            N_DRAWS,
            _rng(k, 3),
            top_k=min(3, len(m) - 1),  # "top three" of three models would always agree
            resampled_sizes=study.RESAMPLED_SIZES[data.modality],
        ),
        "seeds": reliability.seed_reproducibility(
            [data.correct(s)[:, :, 0] for s in data.suites], [data.field(s, "task") for s in data.suites], m, data.suites
        ),
        "templates": reliability.template_sensitivity(main, tasks_main, m, N_BOOT, _rng(k, 4)),
        "items": items,
        "floor_ceiling_pooled": reliability.floor_ceiling(pool, tasks_pool, choices_pool, m, N_BOOT, _rng(k, 6)),
        "floor_ceiling_main": reliability.floor_ceiling(main[:, :, 0], tasks_main, data.field(data.main, "n_choices"), m, N_BOOT, _rng(k, 7)),
        "levels": reliability.accuracy_by_level(pool, tasks_pool, levels_pool, m),
        "redundancy": reliability.task_redundancy(pool, tasks_pool, m, N_BOOT, _rng(k, 8)),
    }
    # The floor and ceiling checks again at each difficulty level.
    out["floor_ceiling_by_level"] = {
        str(lv): reliability.floor_ceiling(
            pool[:, levels_pool == lv], tasks_pool[levels_pool == lv], choices_pool[levels_pool == lv], m, N_BOOT, _rng(k, 10 + lv)
        )
        for lv in (1, 2, 3)
    }
    return out, item_arrays, item_ids


def tidy_item_results(data: RunData) -> pd.DataFrame:
    """One row per (suite, item, template, model): right or wrong, and the margin of the true caption."""
    frames = []
    for suite, seed in zip(data.suites, data.seeds):
        for model in data.models:
            res = data.results[suite][model]
            scores, answer = res["scores"].astype(float), res["answer"].astype(int)
            n, n_templates, _ = scores.shape
            true = scores[np.arange(n), :, answer]
            others = scores.copy()
            others[np.arange(n), :, answer] = np.nan
            margin = true - np.nanmax(others, axis=2)
            for t in range(n_templates):
                frames.append(
                    pd.DataFrame(
                        {
                            "modality": data.modality,
                            "suite": suite,
                            "seed": np.int16(seed),
                            "item_id": res["item_id"],
                            "task": res["task"],
                            "level": res["level"].astype(np.int8),
                            "n_choices": res["n_choices"].astype(np.int8),
                            "template": np.int8(t),
                            "model": model,
                            "correct": res["correct"][:, t],
                            "margin": margin[:, t].astype(np.float32),
                        }
                    )
                )
    frame = pd.concat(frames, ignore_index=True)
    for col in ("modality", "suite", "task", "model"):
        frame[col] = frame[col].astype("category")
    return frame


def summarise(results_root: Path, run_id: str = study.RUN_ID) -> dict:
    results_root = Path(results_root)
    runs = load_run(results_root, run_id)
    if not runs:
        raise FileNotFoundError(f"no result files under {results_root / 'runs' / run_id}")
    summary: dict[str, Any] = {
        "run_id": run_id,
        "analysis": {"seed": ANALYSIS_SEED, "n_boot": N_BOOT, "n_draws": N_DRAWS, "n_split": N_SPLIT, "ceiling": reliability.CEILING},
        "models": {},
    }
    tidy, item_frames = [], []
    for modality in ("vision", "audio"):
        if modality not in runs:
            continue
        data = runs[modality]
        analysis, arrays, item_ids = analyse(data)
        summary[modality] = analysis
        for key in data.models:
            spec = MODELS.get(key)
            summary["models"][key] = {
                "modality": modality,
                "identifier": data.results[data.suites[0]][key]["info"]["model_identifier"],
                "label": spec.label if spec else key,
                "architecture": spec.architecture if spec else "",
                "training_data": spec.training_data if spec else "",
            }
        tidy.append(tidy_item_results(data))
        item_frames.append(
            pd.DataFrame(
                {
                    "modality": modality,
                    "item": item_ids,
                    "task": data.pooled("task"),
                    "level": data.pooled("level").astype(np.int8),
                    "share_of_models_right": arrays["difficulty"].astype(np.float32),
                    "item_ability_correlation": arrays["discrimination"].astype(np.float32),
                }
            )
        )
    from .hypotheses import evaluate_hypotheses

    summary["hypotheses"] = evaluate_hypotheses(summary)
    summary = _clean(summary)
    with open(results_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
        f.write("\n")
    pd.concat(tidy, ignore_index=True).to_parquet(results_root / "item_results.parquet", index=False, compression="zstd")
    pd.concat(item_frames, ignore_index=True).to_parquet(results_root / "item_statistics.parquet", index=False, compression="zstd")
    return summary
