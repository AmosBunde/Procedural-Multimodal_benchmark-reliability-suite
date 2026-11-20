"""The model adapter interface.

A model takes part in the benchmark by answering one question: given one
image or one audio clip and a list of candidate texts, how well does each text
match? Higher is better. The scale is free, because only the order of the
scores within one call is used.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from ..names import validate_name


class Adapter:
    """Base class for models. Subclasses set ``name`` and ``modality`` and implement ``score``."""

    name: str = "unnamed"
    modality: str = "image"  # "image" or "audio"
    identifier: str = ""  # exact checkpoint, recorded with the results

    def score(self, media, candidate_texts: Sequence[str]) -> np.ndarray:
        """Return one score per candidate text for a single image or clip."""
        raise NotImplementedError

    def score_batch(self, batch: Sequence[tuple[object, Sequence[str]]]) -> list[np.ndarray]:
        """Score several (media, candidate_texts) pairs. Override to batch the forward pass."""
        return [np.asarray(self.score(media, texts), dtype=np.float64) for media, texts in batch]

    def close(self) -> None:
        """Release the model."""


def check_adapter(adapter: Adapter) -> None:
    validate_name(adapter.name, "model key")
    if adapter.modality not in {"image", "audio"}:
        raise ValueError(f"{adapter.name}: modality must be 'image' or 'audio'")


def media_bytes(media) -> bytes:
    if hasattr(media, "samples"):
        return np.ascontiguousarray(media.samples).tobytes()
    return media.tobytes()


class HashAdapter(Adapter):
    """A model that knows nothing: scores are a fixed hash of the media and the text.

    It is deterministic, costs nothing to run and scores at chance, which makes it
    useful for testing the pipeline and for checking the chance-floor analysis.
    """

    def __init__(self, name: str = "hash-baseline", modality: str = "image", salt: str = ""):
        self.name = name
        self.modality = modality
        self.identifier = f"hash:{salt}"
        self._salt = salt.encode()
        self.calls = 0

    def score(self, media, candidate_texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        base = hashlib.sha256(self._salt + media_bytes(media)).digest()
        out = []
        for text in candidate_texts:
            digest = hashlib.sha256(base + text.encode()).digest()
            out.append(int.from_bytes(digest[:8], "big") / 2**64)
        return np.array(out)
