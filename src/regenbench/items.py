"""The item format and the manifest that holds a generated suite.

An item is a complete description of one test question: the scene that the
media is rendered from, the candidate captions under every caption template,
the index of the correct candidate, and a plain question/answer rendering for
generative models. The media is a pure function of ``scene``, so a manifest is
all that has to be stored or shared.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .names import validate_name

MANIFEST_NAME = "items.jsonl"
META_NAME = "suite.json"


@dataclass(frozen=True)
class Item:
    item_id: str
    task: str
    modality: str  # "image" or "audio"
    level: int  # 1 (easy) to 3 (hard)
    scene: dict[str, Any]
    captions: list[list[str]]  # captions[template][choice]
    answer: int  # index of the correct choice
    question: str
    choices: list[str]  # short answers for the question/answer rendering
    truth: dict[str, Any] = field(default_factory=dict)  # the fact being tested

    @property
    def n_choices(self) -> int:
        return len(self.choices)

    @property
    def n_templates(self) -> int:
        return len(self.captions)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_json(line: str) -> "Item":
        return Item(**json.loads(line))

    def qa_prompt(self) -> str:
        """Plain multiple-choice rendering for a generative model."""
        letters = "ABCDEFGH"
        lines = [self.question]
        lines += [f"{letters[i]}. {c}" for i, c in enumerate(self.choices)]
        lines.append("Answer with a single letter.")
        return "\n".join(lines)

    def qa_answer(self) -> str:
        return "ABCDEFGH"[self.answer]


def check_item(item: Item) -> None:
    """Structural checks that every generator must satisfy."""
    validate_name(item.item_id, "item id")
    validate_name(item.task, "task name")
    if item.modality not in {"image", "audio"}:
        raise ValueError(f"{item.item_id}: unknown modality {item.modality!r}")
    if item.level not in {1, 2, 3}:
        raise ValueError(f"{item.item_id}: level must be 1, 2 or 3")
    n = item.n_choices
    if n < 2:
        raise ValueError(f"{item.item_id}: needs at least two choices")
    if not 0 <= item.answer < n:
        raise ValueError(f"{item.item_id}: answer index out of range")
    if len(set(item.choices)) != n:
        raise ValueError(f"{item.item_id}: duplicate choices")
    for t, caps in enumerate(item.captions):
        if len(caps) != n:
            raise ValueError(f"{item.item_id}: template {t} has {len(caps)} captions, expected {n}")
        if len(set(caps)) != n:
            raise ValueError(f"{item.item_id}: template {t} has duplicate captions")


def manifest_sha256(items: list[Item]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it.to_json().encode())
        h.update(b"\n")
    return h.hexdigest()


def write_suite(directory: Path, suite_id: str, items: list[Item], meta: dict[str, Any]) -> dict[str, Any]:
    validate_name(suite_id, "suite id")
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / MANIFEST_NAME, "w") as f:
        for it in items:
            f.write(it.to_json() + "\n")
    meta = dict(meta, suite_id=suite_id, n_items=len(items), sha256=manifest_sha256(items))
    with open(directory / META_NAME, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return meta


def read_suite(directory: Path) -> tuple[dict[str, Any], list[Item]]:
    with open(directory / META_NAME) as f:
        meta = json.load(f)
    validate_name(meta.get("suite_id"), "suite id")
    with open(directory / MANIFEST_NAME) as f:
        items = [Item.from_json(line) for line in f if line.strip()]
    for it in items:
        check_item(it)
    if manifest_sha256(items) != meta.get("sha256"):
        raise ValueError(f"{directory}: manifest does not match the checksum in {META_NAME}")
    return meta, items
