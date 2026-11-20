"""Build a suite of items from a seed.

Each item draws from its own random stream, keyed by (suite seed, task, level,
index). An item therefore does not depend on how many other items were asked
for: a suite of 60 items per task is a prefix of a suite of 240 with the same
seed, and one task can be regenerated without touching the others.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np

from . import tasks_audio, tasks_vision
from .items import Item, check_item, write_suite
from .names import validate_name
from .render import render

MODALITIES = {
    "vision": ("image", tasks_vision),
    "audio": ("audio", tasks_audio),
}
LEVELS = (1, 2, 3)
GENERATOR_VERSION = 1
N_TEMPLATES = 3


def task_names(modality: str) -> list[str]:
    return list(MODALITIES[modality][1].GENERATORS)


def chance_level(n_choices: int) -> float:
    return 1.0 / n_choices


def make_item(modality: str, task: str, level: int, index: int, seed: int, n_templates: int | None = None) -> Item:
    kind, module = MODALITIES[modality]
    stream = np.random.SeedSequence([int(seed), zlib.crc32(task.encode()), int(level), int(index), GENERATOR_VERSION])
    rng = np.random.default_rng(stream)
    scene, options, answer, question, truth = module.GENERATORS[task](rng, level)
    item = Item(
        item_id=f"{task}-L{level}-{index:04d}",
        task=task,
        modality=kind,
        level=level,
        scene=scene,
        captions=[[template(o) for o in options] for template in module.TEMPLATES[task][:n_templates]],
        answer=answer,
        question=question,
        choices=[module.SHORT(task, o) for o in options],
        truth=truth,
    )
    check_item(item)
    return item


def generate_suite(
    modality: str, seed: int, n_per_task: int, tasks: list[str] | None = None, n_templates: int | None = None
) -> list[Item]:
    """Generate ``n_per_task`` items for each task, split evenly over three levels.

    ``n_templates`` keeps only the first few caption templates (1 keeps the canonical
    wording alone). The scenes and the answers do not depend on it.
    """
    if n_templates is not None and not 1 <= n_templates <= N_TEMPLATES:
        raise ValueError(f"n_templates must be between 1 and {N_TEMPLATES}")
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    if n_per_task <= 0 or n_per_task % len(LEVELS) != 0:
        raise ValueError("n_per_task must be a positive multiple of 3 (three difficulty levels)")
    names = tasks or task_names(modality)
    items = []
    for task in names:
        if task not in MODALITIES[modality][1].GENERATORS:
            raise ValueError(f"unknown {modality} task {task!r}")
        for level in LEVELS:
            for index in range(n_per_task // len(LEVELS)):
                items.append(make_item(modality, task, level, index, seed, n_templates))
    return items


def default_suite_id(modality: str, seed: int) -> str:
    return f"{modality}-seed{seed}"


def write_generated_suite(
    out_root: Path,
    modality: str,
    seed: int,
    n_per_task: int,
    suite_id: str | None = None,
    write_media: bool = False,
    n_templates: int | None = None,
) -> dict[str, Any]:
    suite_id = validate_name(suite_id or default_suite_id(modality, seed), "suite id")
    items = generate_suite(modality, seed, n_per_task, n_templates=n_templates)
    directory = out_root / suite_id
    meta = write_suite(
        directory,
        suite_id,
        items,
        {
            "modality": modality,
            "seed": int(seed),
            "n_per_task": int(n_per_task),
            "n_templates": items[0].n_templates,
            "generator_version": GENERATOR_VERSION,
        },
    )
    if write_media:
        write_media_files(directory / "media", items)
    return meta


def write_media_files(directory: Path, items: list[Item]) -> None:
    """Write PNG or WAV files for inspection. Evaluation does not need them."""
    import wave

    directory.mkdir(parents=True, exist_ok=True)
    for item in items:
        media = render(item)
        name = validate_name(item.item_id, "item id")
        if item.modality == "image":
            media.save(directory / f"{name}.png")
        else:
            pcm = (np.clip(media.samples, -1, 1) * 32767).astype("<i2")
            with wave.open(str(directory / f"{name}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(media.rate)
                w.writeframes(pcm.tobytes())
