"""Rendering: a scene description in, an image or a waveform out.

Both renderers are pure functions of the scene dictionary. Nothing random
happens here that is not seeded from the scene itself, so a manifest alone is
enough to rebuild every pixel and every sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (212, 38, 38),
    "blue": (30, 88, 204),
    "green": (36, 156, 60),
    "yellow": (248, 216, 32),
    "purple": (136, 56, 172),
    "orange": (244, 128, 20),
}
SHAPES = ("circle", "square", "triangle", "star")
BACKGROUND = (236, 236, 236)
SUPERSAMPLE = 4


@dataclass(frozen=True)
class Audio:
    """A mono waveform in [-1, 1] and its sample rate."""

    samples: np.ndarray
    rate: int


def shape_polygon(shape: str, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """Vertices of a shape whose circumscribed circle has radius ``r``."""
    if shape == "square":
        h = r / math.sqrt(2.0)
        return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    if shape == "triangle":
        return [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in (-90, 30, 150)]
    if shape == "star":
        pts = []
        for k in range(10):
            rad = r if k % 2 == 0 else 0.45 * r
            a = math.radians(-90 + 36 * k)
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        return pts
    raise ValueError(f"unknown shape {shape!r}")


def render_image(scene: dict[str, Any]) -> Image.Image:
    size = int(scene["size"])
    s = SUPERSAMPLE
    big = Image.new("RGB", (size * s, size * s), tuple(scene["background"]))
    draw = ImageDraw.Draw(big)
    for obj in scene.get("objects", []):
        fill = COLOURS[obj["colour"]]
        cx, cy, r = obj["cx"] * s, obj["cy"] * s, obj["r"] * s
        if obj["shape"] == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        else:
            draw.polygon(shape_polygon(obj["shape"], cx, cy, r), fill=fill)
    text = scene.get("text")
    if text is not None:
        font = ImageFont.load_default(size=int(text["font_size"]) * s)
        draw.text((text["cx"] * s, text["cy"] * s), text["string"], font=font, anchor="mm", fill=tuple(text["fill"]))
    img = big.resize((size, size), Image.BOX)
    sigma = float(scene.get("noise", 0.0))
    if sigma > 0:
        rng = np.random.default_rng(int(scene["noise_seed"]))
        arr = np.asarray(img, dtype=np.float64) + rng.normal(0.0, sigma, (size, size, 3))
        img = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    return img


def _ramp(n: int, rate: int, ms: float = 10.0) -> np.ndarray:
    """Raised-cosine fade in and out, so events do not click."""
    env = np.ones(n)
    k = min(int(rate * ms / 1000.0), n // 2)
    if k > 0:
        fade = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
        env[:k] = fade
        env[n - k :] = fade[::-1]
    return env


def render_audio(scene: dict[str, Any]) -> Audio:
    rate = int(scene["rate"])
    total = int(round(float(scene["duration"]) * rate))
    out = np.zeros(total, dtype=np.float64)
    rng = np.random.default_rng(int(scene["noise_seed"]))
    for ev in scene["events"]:
        start = int(round(ev["t0"] * rate))
        if ev["kind"] == "click":
            n = int(0.012 * rate)
            t = np.arange(n) / rate
            sig = np.sin(2 * np.pi * 2000.0 * t) * np.exp(-t / 0.002)
        else:
            n = int(round(ev["dur"] * rate))
            t = np.arange(n) / rate
            if ev["kind"] == "tone":
                f0, f1 = float(ev["f0"]), float(ev["f1"])
                if f0 == f1:
                    phase = 2 * np.pi * f0 * t
                else:  # exponential sweep: equal musical steps in equal time
                    k = math.log(f1 / f0) / ev["dur"]
                    phase = 2 * np.pi * f0 * (np.exp(k * t) - 1.0) / k
                sig = np.sin(phase)
            elif ev["kind"] == "noise":
                sig = rng.uniform(-1.0, 1.0, n)
            else:
                raise ValueError(f"unknown event kind {ev['kind']!r}")
            sig = sig * _ramp(n, rate)
        end = min(start + n, total)
        out[start:end] += ev["amp"] * sig[: end - start]
    floor = float(scene.get("noise", 0.0))
    if floor > 0:
        out += rng.normal(0.0, floor, total)
    return Audio(np.clip(out, -1.0, 1.0).astype(np.float32), rate)


def render(item) -> Image.Image | Audio:
    """Render the media of an item."""
    if item.modality == "image":
        return render_image(item.scene)
    if item.modality == "audio":
        return render_audio(item.scene)
    raise ValueError(f"unknown modality {item.modality!r}")
