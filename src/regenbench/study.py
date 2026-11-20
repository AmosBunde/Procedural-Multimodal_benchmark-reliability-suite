"""The design of the study reported in REPORT.md: which suites, which models.

Kept in one place so that the plan, the commands and the analysis agree.
"""

from __future__ import annotations

RUN_ID = "main"

# Five image suites of 72 items per task (24 per difficulty level). Seed 0 is the main suite and carries all
# three caption templates; seeds 1 to 4 are regenerations with the canonical wording.
VISION_SEEDS = (0, 1, 2, 3, 4)
VISION_N_PER_TASK = 72
# Three audio suites of 30 items per task (10 per level), all with three templates.
AUDIO_SEEDS = (0, 1, 2)
AUDIO_N_PER_TASK = 30

MAIN_SEED = 0

# Suite sizes (items per task) at which ranking agreement is measured.
SIZES = {"vision": (9, 18, 36, 72, 108, 144, 180, 270, 360), "audio": (10, 20, 30, 45, 90)}

VISION_MODELS = (
    "openai-rn50",
    "openai-vit-b32",
    "laion2b-vit-b32",
    "datacomp-vit-b32",
    "metaclip-vit-b32",
    "siglip-vit-b16",
)
AUDIO_MODELS = ("clap-htsat-unfused", "clap-larger-general", "clap-larger-music-speech")


def suites() -> list[dict]:
    """Every suite of the study as keyword arguments for ``write_generated_suite``."""
    out = []
    for seed in VISION_SEEDS:
        out.append(
            {
                "modality": "vision",
                "seed": seed,
                "n_per_task": VISION_N_PER_TASK,
                "n_templates": 3 if seed == MAIN_SEED else 1,
            }
        )
    for seed in AUDIO_SEEDS:
        out.append({"modality": "audio", "seed": seed, "n_per_task": AUDIO_N_PER_TASK, "n_templates": 3})
    return out


def models_for(modality: str) -> tuple[str, ...]:
    return VISION_MODELS if modality == "vision" else AUDIO_MODELS
