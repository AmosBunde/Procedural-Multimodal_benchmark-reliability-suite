"""The model registry and the study design. No weights are loaded."""

import pytest

from regenbench import study
from regenbench.models import MODELS, get_model, models_for
from regenbench.names import validate_name


def test_registry_keys_are_safe_names_and_identifiers_are_repositories():
    for key, spec in MODELS.items():
        assert validate_name(key) == key == spec.key
        assert spec.modality in {"image", "audio"} and spec.loader in {"open_clip", "clap"}
        owner, _, repo = spec.identifier.partition("/")
        assert owner and repo and " " not in spec.identifier


def test_study_models_exist_and_match_their_modality():
    assert len(study.VISION_MODELS) == 6 and len(study.AUDIO_MODELS) == 3
    for key in study.VISION_MODELS:
        assert get_model(key).modality == "image"
    for key in study.AUDIO_MODELS:
        assert get_model(key).modality == "audio"
    assert {s.key for s in models_for("image")} >= set(study.VISION_MODELS)


def test_unknown_or_unsafe_model_keys_are_refused():
    with pytest.raises(KeyError):
        get_model("no-such-model")
    with pytest.raises(ValueError):
        get_model("../weights")


def test_study_design():
    suites = study.suites()
    vision = [s for s in suites if s["modality"] == "vision"]
    audio = [s for s in suites if s["modality"] == "audio"]
    assert [s["seed"] for s in vision] == [0, 1, 2, 3, 4] and [s["seed"] for s in audio] == [0, 1, 2]
    assert [s["n_templates"] for s in vision] == [3, 1, 1, 1, 1], "only the main suite carries the paraphrases"
    assert all(s["n_per_task"] % 3 == 0 for s in suites)
    # The suite size actually used must be one of the sizes at which agreement is measured.
    assert study.VISION_N_PER_TASK in study.SIZES["vision"] and study.AUDIO_N_PER_TASK in study.SIZES["audio"]
