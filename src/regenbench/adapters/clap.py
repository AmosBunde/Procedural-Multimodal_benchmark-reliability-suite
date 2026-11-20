"""Adapter for contrastive audio-text models (CLAP) loaded through transformers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Adapter


class ClapAdapter(Adapter):
    modality = "audio"

    def __init__(self, name: str, identifier: str, threads: int = 6):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.name = name
        self.identifier = identifier
        self._torch = torch
        torch.set_num_threads(threads)
        self._model = ClapModel.from_pretrained(identifier).eval()
        self._processor = ClapProcessor.from_pretrained(identifier)
        self._rate = self._processor.feature_extractor.sampling_rate
        self._text_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _features(output):
        # Newer transformers versions wrap the projected embedding in an output object.
        return output if hasattr(output, "norm") else output.pooler_output

    def _embed_texts(self, texts: Sequence[str]) -> None:
        new = sorted({t for t in texts if t not in self._text_cache})
        torch = self._torch
        for i in range(0, len(new), 64):
            chunk = new[i : i + 64]
            tokens = self._processor.tokenizer(chunk, padding=True, return_tensors="pt")
            with torch.no_grad():
                emb = self._features(self._model.get_text_features(**tokens))
                emb = emb / emb.norm(dim=-1, keepdim=True)
            for t, e in zip(chunk, emb.numpy().astype(np.float64)):
                self._text_cache[t] = e

    def score_batch(self, batch):
        torch = self._torch
        self._embed_texts([t for _, texts in batch for t in texts])
        clips = []
        for media, _ in batch:
            if media.rate != self._rate:
                raise ValueError(f"{self.name} expects {self._rate} Hz audio, got {media.rate} Hz")
            clips.append(np.asarray(media.samples, dtype=np.float32))
        inputs = self._processor.feature_extractor(clips, sampling_rate=self._rate, return_tensors="pt")
        with torch.no_grad():
            emb = self._features(self._model.get_audio_features(**inputs))
            emb = (emb / emb.norm(dim=-1, keepdim=True)).numpy().astype(np.float64)
        return [np.stack([self._text_cache[t] for t in texts]) @ e for e, (_, texts) in zip(emb, batch)]

    def score(self, media, candidate_texts):
        return self.score_batch([(media, list(candidate_texts))])[0]

    def close(self) -> None:
        self._model = None
        self._text_cache.clear()
