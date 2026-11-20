import json

import pytest

from regenbench.generate import generate_suite, write_generated_suite
from regenbench.items import MANIFEST_NAME, META_NAME, Item, check_item, manifest_sha256, read_suite


def test_manifest_round_trip(tmp_path):
    meta = write_generated_suite(tmp_path, "vision", seed=5, n_per_task=3)
    read_meta, items = read_suite(tmp_path / "vision-seed5")
    assert read_meta == meta
    assert meta["n_items"] == len(items) == 18
    assert manifest_sha256(items) == meta["sha256"]
    assert items == generate_suite("vision", 5, 3)


def test_tampered_manifest_is_detected(tmp_path):
    write_generated_suite(tmp_path, "vision", seed=5, n_per_task=3)
    path = tmp_path / "vision-seed5" / MANIFEST_NAME
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["answer"] = (first["answer"] + 1) % len(first["choices"])
    path.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        read_suite(tmp_path / "vision-seed5")


def test_suite_id_from_file_is_validated(tmp_path):
    write_generated_suite(tmp_path, "vision", seed=5, n_per_task=3)
    meta_path = tmp_path / "vision-seed5" / META_NAME
    meta = json.loads(meta_path.read_text())
    meta["suite_id"] = "../../escape"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="invalid suite id"):
        read_suite(tmp_path / "vision-seed5")


def test_check_item_rejects_broken_items():
    good = generate_suite("vision", 0, 3)[0]
    check_item(good)
    fields = good.__dict__
    with pytest.raises(ValueError):
        check_item(Item(**{**fields, "answer": 9}))
    with pytest.raises(ValueError):
        check_item(Item(**{**fields, "captions": [good.captions[0][:1] * good.n_choices]}))
    with pytest.raises(ValueError):
        check_item(Item(**{**fields, "item_id": "../x"}))


def test_question_answer_rendering():
    for item in generate_suite("vision", 2, 3) + generate_suite("audio", 2, 3):
        prompt = item.qa_prompt()
        lines = prompt.splitlines()
        assert lines[0] == item.question
        assert len(lines) == item.n_choices + 2
        letter = item.qa_answer()
        assert f"{letter}. {item.choices[item.answer]}" in lines
