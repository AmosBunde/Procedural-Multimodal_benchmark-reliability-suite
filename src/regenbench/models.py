"""The models evaluated in the study, and how to load them.

Identifiers are Hugging Face repository names. Weights are downloaded
anonymously on first use and are never stored in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapters.base import Adapter
from .names import validate_name


@dataclass(frozen=True)
class ModelSpec:
    key: str
    modality: str  # "image" or "audio"
    loader: str  # "open_clip" or "clap"
    identifier: str  # Hugging Face repository
    architecture: str
    training_data: str
    label: str

    def load(self, threads: int = 6) -> Adapter:
        if self.loader == "open_clip":
            from .adapters.openclip import OpenClipAdapter

            return OpenClipAdapter(self.key, f"hf-hub:{self.identifier}", threads=threads)
        if self.loader == "clap":
            from .adapters.clap import ClapAdapter

            return ClapAdapter(self.key, self.identifier, threads=threads)
        raise ValueError(f"unknown loader {self.loader!r}")


_SPECS = [
    ModelSpec("openai-rn50", "image", "open_clip", "timm/resnet50_clip.openai", "ResNet-50", "OpenAI WIT, 400M pairs", "OpenAI RN50"),
    ModelSpec("openai-vit-b32", "image", "open_clip", "timm/vit_base_patch32_clip_224.openai", "ViT-B/32", "OpenAI WIT, 400M pairs", "OpenAI ViT-B/32"),
    ModelSpec("laion2b-vit-b32", "image", "open_clip", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "ViT-B/32", "LAION-2B", "LAION-2B ViT-B/32"),
    ModelSpec("datacomp-vit-b32", "image", "open_clip", "laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K", "ViT-B/32", "DataComp-1B", "DataComp ViT-B/32"),
    ModelSpec("metaclip-vit-b32", "image", "open_clip", "timm/vit_base_patch32_clip_224.metaclip_400m", "ViT-B/32", "MetaCLIP 400M", "MetaCLIP ViT-B/32"),
    ModelSpec("siglip-vit-b16", "image", "open_clip", "timm/ViT-B-16-SigLIP", "ViT-B/16, sigmoid loss", "WebLI", "SigLIP ViT-B/16"),
    ModelSpec("clap-htsat-unfused", "audio", "clap", "laion/clap-htsat-unfused", "HTSAT + RoBERTa", "LAION-Audio-630K", "CLAP HTSAT unfused"),
    ModelSpec("clap-larger-general", "audio", "clap", "laion/larger_clap_general", "HTSAT (larger) + RoBERTa", "general audio, music and speech", "CLAP larger, general"),
    ModelSpec("clap-larger-music-speech", "audio", "clap", "laion/larger_clap_music_and_speech", "HTSAT (larger) + RoBERTa", "music and speech", "CLAP larger, music and speech"),
]
MODELS: dict[str, ModelSpec] = {validate_name(s.key, "model key"): s for s in _SPECS}


def models_for(modality: str) -> list[ModelSpec]:
    return [s for s in _SPECS if s.modality == modality]


def get_model(key: str) -> ModelSpec:
    validate_name(key, "model key")
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; known models: {', '.join(MODELS)}")
    return MODELS[key]
