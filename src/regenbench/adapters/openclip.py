"""Adapter for contrastive image-text models loaded through open_clip.

Text encoding is the slow part on a CPU, because the standard CLIP text tower
always processes 77 token positions and the captions here use fewer than 25.
For models whose text tower is causal and pooled at the end-of-text token, the
positions after that token cannot influence the result, so they are skipped.
The shortcut is checked against the full-length encoding when the model is
loaded and is switched off if the two disagree. Other text towers, SigLIP
among them, always run at full length.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Adapter


class OpenClipAdapter(Adapter):
    modality = "image"

    def __init__(self, name: str, identifier: str, threads: int = 6, text_batch: int = 256):
        import open_clip
        import torch

        self.name = name
        self.identifier = identifier
        self._torch = torch
        torch.set_num_threads(threads)
        model, _, preprocess = open_clip.create_model_and_transforms(identifier)
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(identifier)
        self._text_batch = text_batch
        self._text_cache: dict[str, np.ndarray] = {}
        self.short_context = self._short_context_is_exact()

    def _encode_tokens(self, tokens, short: bool):
        """Text embeddings, optionally skipping the padding after the end-of-text token."""
        torch, model = self._torch, self._model
        if not short:
            return model.encode_text(tokens)
        length = int(tokens.argmax(dim=-1).max()) + 1  # end-of-text has the highest token id
        positions, mask = model.positional_embedding, model.attn_mask
        try:
            model.positional_embedding = torch.nn.Parameter(positions[:length], requires_grad=False)
            model.attn_mask = mask[:length, :length]
            return model.encode_text(tokens[:, :length])
        finally:
            model.positional_embedding = positions
            model.attn_mask = mask

    def _short_context_is_exact(self) -> bool:
        import open_clip

        torch, model = self._torch, self._model
        if type(model) is not open_clip.model.CLIP or getattr(model, "text_pool_type", None) != "argmax":
            return False
        if getattr(model, "attn_mask", None) is None:
            return False
        probes = self._tokenizer(
            [
                "a photo of a red square",
                "an image in which an orange triangle is to the right of a purple square",
                "a picture showing 7 blue stars",
                "an image with the digits 8746 written on it",
            ]
        )
        try:
            with torch.no_grad():
                full = self._encode_tokens(probes, short=False)
                short = self._encode_tokens(probes, short=True)
        except Exception:
            return False
        return bool((full - short).abs().max() <= 1e-4 * full.abs().max())

    def _embed_texts(self, texts: Sequence[str]) -> None:
        # Captions repeat a great deal across items, so each distinct string is encoded once.
        new = sorted({t for t in texts if t not in self._text_cache})
        torch = self._torch
        for i in range(0, len(new), self._text_batch):
            chunk = new[i : i + self._text_batch]
            with torch.no_grad():
                emb = self._encode_tokens(self._tokenizer(chunk), short=self.short_context)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            for t, e in zip(chunk, emb.numpy().astype(np.float64)):
                self._text_cache[t] = e

    def score_batch(self, batch):
        torch = self._torch
        self._embed_texts([t for _, texts in batch for t in texts])
        pixels = torch.stack([self._preprocess(media.convert("RGB")) for media, _ in batch])
        with torch.no_grad():
            emb = self._model.encode_image(pixels)
            emb = (emb / emb.norm(dim=-1, keepdim=True)).numpy().astype(np.float64)
        return [np.stack([self._text_cache[t] for t in texts]) @ e for e, (_, texts) in zip(emb, batch)]

    def score(self, media, candidate_texts):
        return self.score_batch([(media, list(candidate_texts))])[0]

    def close(self) -> None:
        self._model = None
        self._text_cache.clear()
