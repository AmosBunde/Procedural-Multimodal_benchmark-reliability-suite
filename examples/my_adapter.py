"""Add your own model: subclass Adapter, implement score, hand it to the pipeline.

Run from the repository root after generating a suite:

    uv run regenbench generate --seed 2026 --n-per-task 30 --suite-id fresh
    uv run python examples/my_adapter.py data/fresh
"""

import sys
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from regenbench.adapters import Adapter
from regenbench.evaluate import evaluate_suite, load_result


class HFClip(Adapter):
    name = "hf-clip-vit-b32"  # used in result file names
    modality = "image"  # or "audio"
    identifier = "openai/clip-vit-base-patch32"

    def __init__(self):
        self.model = CLIPModel.from_pretrained(self.identifier).eval()
        self.processor = CLIPProcessor.from_pretrained(self.identifier)

    def score(self, media, candidate_texts):
        inputs = self.processor(text=list(candidate_texts), images=media, return_tensors="pt", padding=True)
        with torch.no_grad():
            return self.model(**inputs).logits_per_image[0].numpy()


if __name__ == "__main__":
    suite = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fresh")
    out = evaluate_suite(HFClip(), suite, Path("results"), run_id="example")
    result = load_result(out)
    print(out)
    for task in dict.fromkeys(result["task"].tolist()):
        rows = result["task"] == task
        print(f"{task:8s} {result['correct'][rows, 0].mean():.3f}  ({rows.sum()} items, canonical wording)")
