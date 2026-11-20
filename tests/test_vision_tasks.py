"""Ground truth of the image tasks, checked from the scene and from the pixels."""

import itertools
import math

import numpy as np
import pytest
from scipy import ndimage

from regenbench.generate import generate_suite, task_names
from regenbench.render import COLOURS, render, render_image
from regenbench.tasks_vision import INVERSE, NUMBER_WORDS, RELATION_PHRASE, SIZE

SEED = 314
ITEMS = generate_suite("vision", SEED, 60)
BY_TASK = {t: [i for i in ITEMS if i.task == t] for t in task_names("vision")}


def test_all_tasks_and_levels_present():
    assert set(BY_TASK) == {"object", "count", "binding", "spatial", "size", "digits"}
    for items in BY_TASK.values():
        assert [sum(i.level == lv for i in items) for lv in (1, 2, 3)] == [20, 20, 20]


def test_generation_is_deterministic():
    again = generate_suite("vision", SEED, 60)
    assert again == ITEMS
    other = generate_suite("vision", SEED + 1, 60)
    assert all(a.scene != b.scene for a, b in zip(ITEMS, other))


def test_smaller_suite_is_a_prefix_of_a_larger_one():
    small = {i.item_id: i for i in generate_suite("vision", SEED, 6)}
    large = {i.item_id: i for i in ITEMS}
    assert set(small) <= set(large)
    assert all(large[k] == v for k, v in small.items())


def test_template_count_changes_captions_only():
    one = generate_suite("vision", SEED, 6, n_templates=1)
    three = generate_suite("vision", SEED, 6)
    for a, b in zip(one, three):
        assert a.n_templates == 1 and b.n_templates == 3
        assert a.scene == b.scene and a.answer == b.answer and a.captions[0] == b.captions[0]


def test_objects_do_not_overlap_and_stay_inside_the_canvas():
    for item in ITEMS:
        objs = item.scene["objects"]
        for o in objs:
            assert o["r"] <= o["cx"] <= SIZE - o["r"] and o["r"] <= o["cy"] <= SIZE - o["r"], item.item_id
        for a, b in itertools.combinations(objs, 2):
            gap = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) - a["r"] - b["r"]
            assert gap >= 5.9, f"{item.item_id}: objects closer than the minimum gap"


def test_answer_position_is_not_biased():
    for n in (2, 4):
        answers = np.array([i.answer for i in ITEMS if i.n_choices == n])
        share = np.bincount(answers, minlength=n) / len(answers)
        assert np.all(np.abs(share - 1 / n) < 0.1), share


def test_captions_use_the_right_article():
    for item in ITEMS:
        for caption in itertools.chain.from_iterable(item.captions):
            assert " a orange" not in f" {caption}", caption


def _count_blobs(item, colour):
    """Count connected regions of one colour in the rendered image."""
    img = np.asarray(render(item), dtype=float)
    mask = np.linalg.norm(img - np.array(COLOURS[colour]), axis=2) < 60
    labels, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return int((sizes >= 30).sum())  # ignore specks left by pixel noise


def test_count_truth_matches_scene_and_pixels():
    for item in BY_TASK["count"]:
        n = item.truth["n"]
        targets = [o for o in item.scene["objects"] if o["role"] == "target"]
        assert len(targets) == n
        assert all(o["colour"] == item.truth["colour"] and o["shape"] == item.truth["shape"] for o in targets)
        # Distractors share neither the colour nor the shape, so they cannot be counted by mistake.
        for o in item.scene["objects"]:
            if o["role"] == "distractor":
                assert o["colour"] != item.truth["colour"] and o["shape"] != item.truth["shape"]
        assert _count_blobs(item, item.truth["colour"]) == n, item.item_id
        assert item.choices[item.answer] == NUMBER_WORDS[n]
        assert f" {NUMBER_WORDS[n]} " in item.captions[0][item.answer]
        assert f" {n} " in item.captions[2][item.answer]
        offered = sorted(NUMBER_WORDS.index(c) for c in item.choices)
        assert offered == list(range(offered[0], offered[0] + 4)) and offered[0] >= 1


def test_count_levels_control_the_range():
    for item in BY_TASK["count"]:
        lo, hi = {1: (1, 4), 2: (2, 7), 3: (3, 9)}[item.level]
        assert lo <= item.truth["n"] <= hi
        assert (item.scene["noise"] > 0) == (item.level == 3)
        assert any(o["role"] == "distractor" for o in item.scene["objects"]) == (item.level == 3)


def test_object_truth():
    for item in BY_TASK["object"]:
        (obj,) = item.scene["objects"]
        assert item.choices[item.answer].endswith(f"{obj['colour']} {obj['shape']}")
        assert _count_blobs(item, obj["colour"]) == 1
        for k, caption in enumerate(item.captions[0]):
            assert caption.endswith(f"{obj['colour']} {obj['shape']}") == (k == item.answer)


def _pairs(scene):
    return {(o["colour"], o["shape"]) for o in scene["objects"]}


def test_binding_true_caption_holds_and_foil_does_not():
    for item in BY_TASK["binding"]:
        present = _pairs(item.scene)
        first, second = (o for o in item.scene["objects"] if o["role"] in {"first", "second"})
        assert first["colour"] != second["colour"] and first["shape"] != second["shape"]
        true = f"a photo of {_a(first)} and {_a(second)}"
        foil = f"a photo of {_a({**first, 'colour': second['colour']})} and {_a({**second, 'colour': first['colour']})}"
        assert item.captions[0][item.answer] == true
        assert item.captions[0][1 - item.answer] == foil
        # Neither swapped pairing exists anywhere in the scene, distractors included.
        assert (second["colour"], first["shape"]) not in present
        assert (first["colour"], second["shape"]) not in present
        # Both captions use exactly the same words, so word overlap cannot separate them.
        assert sorted(true.split()) == sorted(foil.split())


def _a(o):
    return f"{'an' if o['colour'] == 'orange' else 'a'} {o['colour']} {o['shape']}"


def test_spatial_truth_matches_coordinates():
    for item in BY_TASK["spatial"]:
        first, second = (o for o in item.scene["objects"] if o["role"] in {"first", "second"})
        dx, dy = second["cx"] - first["cx"], second["cy"] - first["cy"]
        if abs(dx) > abs(dy):
            relation = "left" if dx > 0 else "right"
            assert abs(dx) >= 3 * abs(dy), "the pair must be clearly side by side"
        else:
            relation = "above" if dy > 0 else "below"  # image y grows downwards
            assert abs(dy) >= 3 * abs(dx), "the pair must be clearly one above the other"
        assert relation == item.truth["relation"]
        true, foil = item.captions[0][item.answer], item.captions[0][1 - item.answer]
        assert true == f"a photo of {_a(first)} {RELATION_PHRASE[relation]} {_a(second)}"
        assert foil == f"a photo of {_a(first)} {RELATION_PHRASE[INVERSE[relation]]} {_a(second)}"
        # The third template states the same fact from the other object's side.
        assert item.captions[2][item.answer] == f"a picture showing {_a(second)} {RELATION_PHRASE[INVERSE[relation]]} {_a(first)}"
        # No distractor repeats either tested object.
        tested = {(first["colour"], first["shape"]), (second["colour"], second["shape"])}
        others = [o for o in item.scene["objects"] if o["role"] == "distractor"]
        assert all(o["colour"] not in {c for c, _ in tested} and o["shape"] not in {s for _, s in tested} for o in others)


def test_size_truth_matches_radii():
    for item in BY_TASK["size"]:
        first, second = item.scene["objects"]
        assert first["shape"] == second["shape"] and first["colour"] != second["colour"]
        big, small = (first, second) if first["r"] > second["r"] else (second, first)
        ratio = big["r"] / small["r"]
        lo, hi = {1: (2.0, 2.6), 2: (1.5, 1.8), 3: (1.25, 1.4)}[item.level]
        assert lo - 0.01 <= ratio <= hi + 0.01
        assert item.truth["larger"] == big["colour"]
        assert item.choices[item.answer] == f"the {big['colour']} {big['shape']}"
        assert f"large {big['colour']}" in item.captions[0][item.answer]
        assert f"small {big['colour']}" in item.captions[0][1 - item.answer]
        # The rendered areas agree with the stated order.
        area = {o["colour"]: (np.linalg.norm(np.asarray(render(item), float) - np.array(COLOURS[o["colour"]]), axis=2) < 60).sum() for o in (big, small)}
        assert area[big["colour"]] > 1.3 * area[small["colour"]]


def test_digits_truth_and_foils():
    for item in BY_TASK["digits"]:
        truth = item.scene["text"]["string"]
        assert item.choices[item.answer] == truth == item.truth["string"]
        assert len(truth) == {1: 2, 2: 3, 3: 4}[item.level] and truth[0] != "0"
        for k, choice in enumerate(item.choices):
            assert len(choice) == len(truth) and choice[0] != "0"
            if k != item.answer and item.level > 1:
                assert sum(a != b for a, b in zip(choice, truth)) == 1
        assert item.captions[0][item.answer] == f"a photo of the number {truth}"


def test_each_digit_foil_would_render_a_different_image():
    for item in BY_TASK["digits"][::7]:
        base = np.asarray(render(item), dtype=int)
        for k, choice in enumerate(item.choices):
            if k == item.answer:
                continue
            scene = {**item.scene, "text": {**item.scene["text"], "string": choice}}
            assert np.abs(np.asarray(render_image(scene), dtype=int) - base).max() > 50


def test_rendering_is_deterministic_and_well_formed():
    for item in ITEMS[::9]:
        a, b = render(item), render(item)
        assert a.size == (SIZE, SIZE) and a.mode == "RGB"
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("bad", [0, 4, 7, -3])
def test_items_per_task_must_split_over_three_levels(bad):
    with pytest.raises(ValueError):
        generate_suite("vision", 0, bad)
