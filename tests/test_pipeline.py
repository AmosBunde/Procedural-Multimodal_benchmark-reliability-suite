"""The adapter contract, the item-level cache and resuming, tested with fake models."""

import json

import numpy as np
import pytest
from fakes import Oracle

from regenbench.adapters import Adapter, HashAdapter
from regenbench.evaluate import correctness, evaluate_suite, load_result, result_path
from regenbench.generate import write_generated_suite
from regenbench.items import read_suite


@pytest.fixture()
def suite(tmp_path):
    write_generated_suite(tmp_path / "data", "vision", seed=11, n_per_task=6)
    directory = tmp_path / "data" / "vision-seed11"
    return directory, read_suite(directory)[1], tmp_path


def test_result_file_shape_and_padding(suite):
    directory, items, tmp = suite
    out = evaluate_suite(HashAdapter(), directory, tmp / "results", "run1", cache_root=tmp / "cache", chunk_size=7)
    assert out == tmp / "results" / "runs" / "run1" / "vision-seed11__hash-baseline.npz"
    res = load_result(out)
    assert res["scores"].shape == (36, 3, 4)
    for it, row in zip(items, res["scores"]):
        assert np.isfinite(row[:, : it.n_choices]).all() and np.isnan(row[:, it.n_choices :]).all()
    assert list(res["item_id"]) == [it.item_id for it in items]
    assert list(res["answer"]) == [it.answer for it in items]
    assert res["info"]["model_key"] == "hash-baseline" and res["info"]["n_items"] == 36


def test_oracle_scores_everything_right_and_its_inverse_everything_wrong(suite):
    directory, items, tmp = suite
    right = load_result(evaluate_suite(Oracle(items), directory, tmp / "results", "r", cache_root=tmp / "cache"))
    wrong = load_result(
        evaluate_suite(Oracle(items, name="anti", invert=True), directory, tmp / "results", "r", cache_root=tmp / "cache")
    )
    assert right["correct"].all() and right["correct"].shape == (36, 3)
    assert not wrong["correct"].any()


def test_ties_count_as_wrong():
    scores = np.array([[[0.5, 0.5, np.nan, np.nan]], [[0.2, 0.9, 0.1, 0.3]], [[0.9, 0.2, 0.1, 0.3]]], dtype=np.float32)
    assert correctness(scores, np.array([0, 1, 1])).tolist() == [[False], [True], [False]]


def test_outputs_are_deterministic(suite):
    directory, _, tmp = suite
    a = load_result(evaluate_suite(HashAdapter(), directory, tmp / "a", "r", cache_root=tmp / "cache-a"))
    b = load_result(evaluate_suite(HashAdapter(), directory, tmp / "b", "r", cache_root=tmp / "cache-b", chunk_size=5))
    assert np.array_equal(a["scores"], b["scores"], equal_nan=True)


def test_interrupted_run_resumes_from_the_cache(suite):
    directory, items, tmp = suite
    kwargs = dict(results_root=tmp / "results", run_id="r", cache_root=tmp / "cache", chunk_size=8)
    with pytest.raises(RuntimeError, match="simulated crash"):
        evaluate_suite(Oracle(items, fail_after=20), directory, **kwargs)
    chunks = sorted((tmp / "cache" / "r" / "vision-seed11" / "oracle").glob("chunk-*.npz"))
    assert len(chunks) == 2, "two full chunks of eight were finished before the crash"
    assert not result_path(tmp / "results", "r", "vision-seed11", "oracle").exists()

    second = Oracle(items)
    res = load_result(evaluate_suite(second, directory, **kwargs))
    assert second.calls == 36 - 16, "only the items missing from the cache are scored again"
    assert res["correct"].all()

    third = Oracle(items)
    evaluate_suite(third, directory, **kwargs)
    assert third.calls == 0, "a finished run is rebuilt from the cache without calling the model"


def test_cache_is_dropped_when_the_suite_changes(suite):
    directory, items, tmp = suite
    kwargs = dict(results_root=tmp / "results", run_id="r", cache_root=tmp / "cache")
    evaluate_suite(HashAdapter(), directory, **kwargs)
    write_generated_suite(tmp / "data", "vision", seed=12, n_per_task=6, suite_id="vision-seed11")
    model = HashAdapter()
    res = load_result(evaluate_suite(model, directory, **kwargs))
    assert model.calls == 36, "a cache built for other items must not be reused"
    assert res["info"]["suite_sha256"] == read_suite(directory)[0]["sha256"]


def test_modality_mismatch_is_refused(suite):
    directory, _, tmp = suite
    with pytest.raises(ValueError, match="takes audio input"):
        evaluate_suite(HashAdapter(modality="audio"), directory, tmp / "results", "r", cache_root=tmp / "cache")


def test_malformed_model_output_is_refused(suite):
    directory, _, tmp = suite

    class TooFew(HashAdapter):
        def score(self, media, candidate_texts):
            return super().score(media, candidate_texts)[:-1]

    class NotFinite(HashAdapter):
        def score(self, media, candidate_texts):
            return np.full(len(candidate_texts), np.nan)

    with pytest.raises(ValueError, match="expected"):
        evaluate_suite(TooFew(), directory, tmp / "results", "r", cache_root=tmp / "c1")
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_suite(NotFinite(), directory, tmp / "results", "r", cache_root=tmp / "c2")


@pytest.mark.parametrize("field", ["run_id", "suite_id", "model_key"])
def test_names_are_validated_before_building_paths(tmp_path, field):
    good = {"run_id": "r", "suite_id": "s", "model_key": "m"}
    with pytest.raises(ValueError):
        result_path(tmp_path, **{**good, field: "../escape"})


def test_names_inside_result_files_are_validated(suite):
    directory, _, tmp = suite
    out = evaluate_suite(HashAdapter(), directory, tmp / "results", "r", cache_root=tmp / "cache")
    with np.load(out) as z:
        data = {k: z[k] for k in z.files}
    info = json.loads(str(data["info"]))
    info["suite_id"] = "../../etc"
    data["info"] = np.array(json.dumps(info))
    np.savez_compressed(out, **data)
    with pytest.raises(ValueError, match="invalid suite id"):
        load_result(out)


def test_model_key_must_be_a_safe_name(suite):
    directory, _, tmp = suite
    with pytest.raises(ValueError, match="invalid model key"):
        evaluate_suite(HashAdapter(name="bad/name"), directory, tmp / "results", "r", cache_root=tmp / "cache")


def test_hash_baseline_scores_at_chance(tmp_path):
    write_generated_suite(tmp_path / "data", "vision", seed=3, n_per_task=90, n_templates=1)
    res = load_result(evaluate_suite(HashAdapter(), tmp_path / "data" / "vision-seed3", tmp_path / "results", "r", cache_root=tmp_path / "c"))
    expected = (1 / res["n_choices"]).mean()
    n = len(res["answer"])
    sd = np.sqrt(((1 / res["n_choices"]) * (1 - 1 / res["n_choices"])).sum()) / n
    assert abs(res["correct"][:, 0].mean() - expected) < 4 * sd


def test_audio_suite_runs_through_the_same_pipeline(tmp_path):
    write_generated_suite(tmp_path / "data", "audio", seed=3, n_per_task=3)
    res = load_result(
        evaluate_suite(HashAdapter(modality="audio"), tmp_path / "data" / "audio-seed3", tmp_path / "results", "r", cache_root=tmp_path / "c")
    )
    assert res["scores"].shape == (12, 3, 4) and res["info"]["modality"] == "audio"
