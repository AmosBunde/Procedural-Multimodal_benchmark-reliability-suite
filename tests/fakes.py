"""Fake models for tests. None of them loads any weights."""

import hashlib

import numpy as np

from regenbench.adapters import Adapter
from regenbench.adapters.base import media_bytes
from regenbench.render import render


class Oracle(Adapter):
    """Knows the answer to every item it was built from.

    ``skill`` is the share of items it answers correctly (chosen by a hash, so the
    outcome is fixed per item), ``invert`` makes it always wrong, and ``fail_after``
    makes it raise after that many calls, to simulate a crash.
    """

    def __init__(self, items, name="oracle", fail_after=None, invert=False, skill=1.0, weak_template=None):
        self.name, self.modality, self.identifier = name, items[0].modality, f"test:{skill}"
        self._truth = {}
        for it in items:
            # Two suites can contain the same picture (a centred two-digit number); keep every wording.
            known = self._truth.setdefault(media_bytes(render(it)), [])
            known += [caps[it.answer] for caps in it.captions[len(known) :]]
        self.calls, self._fail_after, self._invert, self._skill = 0, fail_after, invert, skill
        self._weak_template = weak_template

    def score(self, media, candidate_texts):
        if self._fail_after is not None and self.calls >= self._fail_after:
            raise RuntimeError("simulated crash")
        self.calls += 1
        key = media_bytes(media)
        truth = self._truth[key]
        luck = int.from_bytes(hashlib.sha256(self.name.encode() + key).digest()[:8], "big") / 2**64
        knows = luck < self._skill
        hit = np.array([t in truth for t in candidate_texts], dtype=float)
        if self._weak_template is not None and self._weak_template < len(truth):  # this wording is always answered wrongly
            weak = np.array([t == truth[self._weak_template] for t in candidate_texts])
            hit = np.where(weak, 0.0, hit)
        if self._invert or not knows:
            hit = 1.0 - hit
        return hit
