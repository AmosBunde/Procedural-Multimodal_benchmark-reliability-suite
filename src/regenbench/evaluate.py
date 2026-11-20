"""The evaluation pipeline: one model, one suite, item-level scores.

Items are processed in fixed chunks. Each finished chunk is written to a cache
directory at once, so an interrupted run loses at most one chunk and a restart
skips everything already done. When all chunks exist they are joined into one
result file per (suite, model), which is what the analysis reads.

A result file stores the raw score of every candidate caption under every
template, not just right or wrong. Accuracy, margins and any later scoring rule
can be recomputed without running a model again.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .adapters.base import Adapter, check_adapter
from .items import Item, read_suite
from .names import validate_name
from .render import render

RESULT_VERSION = 1
MODALITY_OF_SUITE = {"vision": "image", "audio": "audio"}


def result_path(results_root: Path, run_id: str, suite_id: str, model_key: str) -> Path:
    for value, what in ((run_id, "run id"), (suite_id, "suite id"), (model_key, "model key")):
        validate_name(value, what)
    return results_root / "runs" / run_id / f"{suite_id}__{model_key}.npz"


def _fingerprint(suite_sha: str, adapter: Adapter, chunk_size: int, n_items: int) -> str:
    text = json.dumps([RESULT_VERSION, suite_sha, adapter.name, adapter.identifier, chunk_size, n_items])
    return hashlib.sha256(text.encode()).hexdigest()


def _atomic_savez(path: Path, **arrays: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def score_items(adapter: Adapter, items: list[Item], width: int | None = None) -> np.ndarray:
    """Scores with shape (items, templates, width), padded with NaN where an item has fewer choices."""
    n_templates = items[0].n_templates
    width = width or max(it.n_choices for it in items)
    batch = [(render(it), [c for caps in it.captions for c in caps]) for it in items]
    raw = adapter.score_batch(batch)
    out = np.full((len(items), n_templates, width), np.nan, dtype=np.float32)
    for i, (it, scores) in enumerate(zip(items, raw)):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (it.n_templates * it.n_choices,):
            raise ValueError(
                f"{adapter.name} returned {scores.shape} scores for {it.item_id}, "
                f"expected {(it.n_templates * it.n_choices,)}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{adapter.name} returned a non-finite score for {it.item_id}")
        out[i, :, : it.n_choices] = scores.reshape(it.n_templates, it.n_choices)
    return out


def evaluate_suite(
    adapter: Adapter,
    suite_dir: Path,
    results_root: Path,
    run_id: str,
    cache_root: Path | None = None,
    chunk_size: int = 32,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Evaluate one model on one suite and return the path of the result file."""
    check_adapter(adapter)
    validate_name(run_id, "run id")
    meta, items = read_suite(Path(suite_dir))
    suite_id = validate_name(meta["suite_id"], "suite id")
    if {it.modality for it in items} != {adapter.modality}:
        raise ValueError(f"{adapter.name} takes {adapter.modality} input but suite {suite_id} holds something else")
    if len({it.n_templates for it in items}) != 1:
        raise ValueError(f"suite {suite_id}: items differ in their number of templates")

    out = result_path(Path(results_root), run_id, suite_id, adapter.name)
    cache_root = Path(cache_root) if cache_root is not None else Path(results_root).parent / ".cache"
    cache = cache_root / run_id / suite_id / adapter.name
    fingerprint = _fingerprint(meta["sha256"], adapter, chunk_size, len(items))
    marker = cache / "fingerprint.txt"
    if cache.exists() and (not marker.exists() or marker.read_text().strip() != fingerprint):
        shutil.rmtree(cache)  # cached chunks belong to a different suite, model or chunking
    cache.mkdir(parents=True, exist_ok=True)
    marker.write_text(fingerprint + "\n")

    width = max(it.n_choices for it in items)
    starts = list(range(0, len(items), chunk_size))
    for k, start in enumerate(starts):
        chunk_file = cache / f"chunk-{k:05d}.npz"
        if chunk_file.exists():
            continue
        t0 = time.perf_counter()
        scores = score_items(adapter, items[start : start + chunk_size], width)
        _atomic_savez(chunk_file, scores=scores, seconds=np.float64(time.perf_counter() - t0))
        if progress is not None:
            progress(f"{adapter.name} on {suite_id}: {min(start + chunk_size, len(items))}/{len(items)} items")

    parts, seconds = [], 0.0
    for k in range(len(starts)):
        with np.load(cache / f"chunk-{k:05d}.npz") as z:
            parts.append(z["scores"])
            seconds += float(z["seconds"])
    info = {
        "result_version": RESULT_VERSION,
        "run_id": run_id,
        "suite_id": suite_id,
        "suite_sha256": meta["sha256"],
        "suite_seed": meta.get("seed"),
        "modality": meta.get("modality"),
        "model_key": adapter.name,
        "model_identifier": adapter.identifier,
        "n_items": len(items),
        "seconds": round(seconds, 3),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        out,
        scores=np.concatenate(parts, axis=0),
        item_id=np.array([it.item_id for it in items]),
        task=np.array([it.task for it in items]),
        level=np.array([it.level for it in items], dtype=np.int8),
        answer=np.array([it.answer for it in items], dtype=np.int8),
        n_choices=np.array([it.n_choices for it in items], dtype=np.int8),
        info=np.array(json.dumps(info, sort_keys=True)),
    )
    return out


def correctness(scores: np.ndarray, answer: np.ndarray) -> np.ndarray:
    """Boolean array (items, templates): the true caption scored strictly above every other one."""
    n = scores.shape[0]
    true = scores[np.arange(n), :, answer]  # (items, templates)
    others = scores.copy()
    others[np.arange(n), :, answer] = np.nan
    best_other = np.nanmax(others, axis=2)
    return true > best_other


def load_result(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    info = json.loads(str(data.pop("info")))
    for key in ("run_id", "suite_id", "model_key"):
        validate_name(info.get(key), key.replace("_", " "))
    data["info"] = info
    data["correct"] = correctness(data["scores"], data["answer"].astype(int))
    return data
