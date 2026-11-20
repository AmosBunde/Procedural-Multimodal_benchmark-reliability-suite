"""End to end: generate small suites, evaluate fake models, summarise, and check the summary."""

import json

import pandas as pd
import pytest

from regenbench.adapters import HashAdapter
from regenbench.evaluate import evaluate_suite
from regenbench.generate import write_generated_suite
from regenbench.summarise import load_run, summarise


def test_summary_file_is_valid_json_without_nan(run):
    tmp, summary = run
    text = (tmp / "results" / "summary.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == summary


def test_design_block(run):
    _, summary = run
    design = summary["vision"]["design"]
    assert design["suites"] == ["vision-seed0", "vision-seed1", "vision-seed2"] and design["main_suite"] == "vision-seed0"
    assert design["n_items_per_task_per_suite"] == 36 and design["n_items_per_task_pooled"] == 108
    assert design["n_items_pooled"] == 648 and design["n_templates_main"] == 3
    assert sorted(design["models"]) == ["fragile", "guesser", "middling", "strong"]


def test_known_structure_is_recovered(run):
    _, summary = run
    v = summary["vision"]
    assert v["ranking_pooled"]["ranking"] == ["strong", "fragile", "middling", "guesser"]
    assert v["ranking_pooled"]["models"]["strong"]["score"] == pytest.approx(0.9, abs=0.04)
    # The guesser sits near chance on every task, the others are above it on every task.
    # (A 5% test on six tasks will now and then call a guesser "above chance" once.)
    for task, block in v["floor_ceiling_pooled"].items():
        assert abs(block["models"]["guesser"]["accuracy"] - block["chance"]) < 0.15, task
        assert block["models"]["strong"]["above_chance"], task
        assert block["status"] == "informative"
    assert sum(b["models"]["guesser"]["above_chance"] for b in v["floor_ceiling_pooled"].values()) <= 1
    # Only the fragile model depends on the wording, and only under the third template.
    t = v["templates"]
    assert t["range"]["strong"] == 0 and t["range"]["middling"] == 0 and t["range"]["fragile"] > 0.7
    assert {(c["model"], c["template"]) for c in t["contrasts"] if c["nonzero"]} >= {("fragile", 2)}
    assert v["seeds"]["mean_tau"] > 0.6
    assert set(summary["hypotheses"]) == {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"}
    assert summary["hypotheses"]["H4"]["outcome"] == "failed" and summary["hypotheses"]["H4"]["tasks_at_floor"] == []


def test_summarise_is_deterministic(run):
    tmp, summary = run
    assert summarise(tmp / "results", "t") == summary


def test_item_level_tables(run):
    tmp, summary = run
    tidy = pd.read_parquet(tmp / "results" / "item_results.parquet")
    # Seed 0 has three templates, seeds 1 and 2 have one; four models; 216 items per suite.
    assert len(tidy) == 4 * 216 * (3 + 1 + 1)
    canon = tidy[(tidy.template == 0) & (tidy.model == "strong")]
    assert canon.correct.mean() == pytest.approx(summary["vision"]["ranking_pooled"]["models"]["strong"]["score"], abs=1e-6)
    assert ((tidy.margin > 0) == tidy.correct).all()
    stats = pd.read_parquet(tmp / "results" / "item_statistics.parquet")
    assert len(stats) == 648 and stats.share_of_models_right.between(0, 1).all()
    varied = stats.item_ability_correlation.notna()
    assert varied.sum() == summary["vision"]["items"]["n_varied"]


def test_load_run_refuses_mixed_suite_versions(run, tmp_path):
    tmp, _ = run
    import shutil

    shutil.copytree(tmp / "results", tmp_path / "results")
    write_generated_suite(tmp_path / "data", "vision", seed=7, n_per_task=36, suite_id="vision-seed1", n_templates=1)
    evaluate_suite(HashAdapter(name="guesser"), tmp_path / "data" / "vision-seed1", tmp_path / "results", "t", cache_root=tmp_path / "c")
    with pytest.raises(ValueError, match="different versions"):
        load_run(tmp_path / "results", "t")


def test_summarise_rejects_unsafe_run_id(tmp_path):
    with pytest.raises(ValueError, match="invalid run id"):
        summarise(tmp_path, "../other")


def test_figures_are_drawn_from_the_summary(run, tmp_path):
    from regenbench.figures import make_figures

    tmp, _ = run
    made = make_figures(tmp / "results", tmp_path / "img")
    names = {p.name for p in made}
    assert {"gallery_vision.png", "gallery_audio.png", "headline.png", "tasks_vision.png", "items_needed.png"} <= names
    assert all(p.exists() and p.stat().st_size > 5_000 for p in made)
