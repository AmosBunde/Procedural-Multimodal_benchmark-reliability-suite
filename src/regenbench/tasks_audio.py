"""Audio tasks.

Clips are ten seconds at 48 kHz, built from sine tones, noise bursts and
clicks. Ten seconds is the window the CLAP models were trained on, so no
padding or repetition is applied before a clip reaches the model, and the
number of events the model hears is the number the caption states.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .tasks_vision import NUMBER_WORDS, _shuffle

RATE = 48_000
DURATION = 10.0
LEAD = 0.5  # silence kept at both ends
EVENT_AMP = 0.5


def _base_scene(rng: np.random.Generator, noise: float) -> dict[str, Any]:
    return {"rate": RATE, "duration": DURATION, "noise": noise, "noise_seed": int(rng.integers(0, 2**31 - 1)), "events": []}


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


# --------------------------------------------------------------------------
# bursts: how many beeps.

BURSTS = {  # count range, burst length, shortest silence between bursts, background noise
    1: ((1, 4), 0.30, 0.50, 0.0),
    2: ((2, 7), 0.20, 0.30, 0.0),
    3: ((3, 9), 0.12, 0.20, 0.05),
}
MAX_BURST_OPTION = 10


def gen_bursts(rng: np.random.Generator, level: int):
    (lo, hi), dur, gap, noise = BURSTS[level]
    n = int(rng.integers(lo, hi + 1))
    freq = round(_log_uniform(rng, 400.0, 1200.0), 1)
    # Spread the spare time at random before, between and after the bursts.
    spare = DURATION - 2 * LEAD - n * dur - (n - 1) * gap
    cuts = rng.dirichlet(np.ones(n + 1)) * spare
    scene = _base_scene(rng, noise)
    t = LEAD
    for k in range(n):
        t += float(cuts[k])
        scene["events"].append({"kind": "tone", "t0": round(t, 4), "dur": dur, "f0": freq, "f1": freq, "amp": EVENT_AMP})
        t += dur + gap
    start = int(rng.integers(max(1, n - 3), min(n, MAX_BURST_OPTION - 3) + 1))
    counts = [n] + [c for c in range(start, start + 4) if c != n]
    options, answer = _shuffle(rng, [{"n": c} for c in counts])
    return scene, options, answer, "How many beeps are in the recording?", {"n": n}


# --------------------------------------------------------------------------
# pitch: does a single tone rise or fall.

PITCH = {1: ((4.0, 4.0), 0.0), 2: ((1.5, 2.0), 0.0), 3: ((1.15, 1.3), 0.05)}  # frequency ratio, noise


def gen_pitch(rng: np.random.Generator, level: int):
    ratio_range, noise = PITCH[level]
    ratio = float(rng.uniform(*ratio_range))
    centre = _log_uniform(rng, 400.0, 1200.0)
    rising = bool(rng.integers(0, 2))
    low, high = centre / np.sqrt(ratio), centre * np.sqrt(ratio)
    f0, f1 = (low, high) if rising else (high, low)
    dur = float(rng.uniform(4.0, 5.0))
    t0 = float(rng.uniform(LEAD, DURATION - LEAD - dur))
    scene = _base_scene(rng, noise)
    scene["events"] = [{"kind": "tone", "t0": round(t0, 4), "dur": round(dur, 4), "f0": round(f0, 2), "f1": round(f1, 2), "amp": EVENT_AMP}]
    options, answer = _shuffle(rng, [{"rising": rising}, {"rising": not rising}])
    return scene, options, answer, "Does the pitch of the tone rise or fall?", {"rising": rising, "ratio": round(ratio, 3)}


# --------------------------------------------------------------------------
# order: a tone and a burst of noise, which comes first.

ORDER = {1: (1.5, 1.0, 0.0), 2: (0.6, 0.3, 0.0), 3: (0.3, 0.1, 0.03)}  # event length, gap, noise


def gen_order(rng: np.random.Generator, level: int):
    dur, gap, noise = ORDER[level]
    freq = round(_log_uniform(rng, 500.0, 1500.0), 1)
    tone_first = bool(rng.integers(0, 2))
    t0 = float(rng.uniform(LEAD, DURATION - LEAD - 2 * dur - gap))
    tone = {"kind": "tone", "dur": dur, "f0": freq, "f1": freq, "amp": EVENT_AMP}
    hiss = {"kind": "noise", "dur": dur, "amp": EVENT_AMP}
    first, second = (tone, hiss) if tone_first else (hiss, tone)
    scene = _base_scene(rng, noise)
    scene["events"] = [dict(first, t0=round(t0, 4)), dict(second, t0=round(t0 + dur + gap, 4))]
    options, answer = _shuffle(rng, [{"tone_first": tone_first}, {"tone_first": not tone_first}])
    return scene, options, answer, "Which comes first, the beep or the burst of noise?", {"tone_first": tone_first}


# --------------------------------------------------------------------------
# tempo: a train of clicks that speeds up or slows down.

TEMPO = {1: ((0.70, 0.12), 0.0), 2: ((0.42, 0.16), 0.0), 3: ((0.30, 0.19), 0.03)}  # slow and fast interval, noise


def click_times(first: float, last: float) -> list[float]:
    """Click onsets whose spacing moves geometrically from ``first`` to ``last`` seconds."""
    span = DURATION - 2 * LEAD
    times, t = [], LEAD
    while t <= DURATION - LEAD:
        times.append(round(t, 4))
        t += first * (last / first) ** ((t - LEAD) / span)
    return times


def gen_tempo(rng: np.random.Generator, level: int):
    (slow, fast), noise = TEMPO[level]
    speeds_up = bool(rng.integers(0, 2))
    jitter = float(rng.uniform(0.95, 1.05))  # so that clips are not all identical
    first, last = (slow * jitter, fast * jitter) if speeds_up else (fast * jitter, slow * jitter)
    scene = _base_scene(rng, noise)
    scene["events"] = [{"kind": "click", "t0": t, "amp": 0.8} for t in click_times(first, last)]
    options, answer = _shuffle(rng, [{"speeds_up": speeds_up}, {"speeds_up": not speeds_up}])
    return scene, options, answer, "Do the clicks speed up or slow down?", {"speeds_up": speeds_up}


def _beeps(o: dict, numeral: bool, noun: str) -> str:
    n = o["n"]
    return f"{n if numeral else NUMBER_WORDS[n]} {noun}{'' if n == 1 else 's'}"


TEMPLATES = {
    "bursts": [
        lambda o: f"the sound of {_beeps(o, False, 'beep')}",
        lambda o: f"a recording with exactly {_beeps(o, False, 'short tone')}",
        lambda o: f"{_beeps(o, True, 'beep')} in a row",
    ],
    "pitch": [
        lambda o: f"a tone {'rising' if o['rising'] else 'falling'} in pitch",
        lambda o: f"a sound whose pitch goes {'up' if o['rising'] else 'down'}",
        lambda o: f"a sine sweep from {'low to high' if o['rising'] else 'high to low'} frequency",
    ],
    "order": [
        lambda o: "a beep followed by a burst of noise" if o["tone_first"] else "a burst of noise followed by a beep",
        lambda o: "first a tone, then a hiss" if o["tone_first"] else "first a hiss, then a tone",
        lambda o: "white noise after a pure tone" if o["tone_first"] else "a pure tone after white noise",
    ],
    "tempo": [
        lambda o: f"clicks that {'speed up' if o['speeds_up'] else 'slow down'}",
        lambda o: f"a ticking sound that gets {'faster' if o['speeds_up'] else 'slower'}",
        lambda o: f"{'an accelerating' if o['speeds_up'] else 'a decelerating'} series of clicks",
    ],
}


def SHORT(task: str, o: dict) -> str:
    if task == "bursts":
        return NUMBER_WORDS[o["n"]]
    if task == "pitch":
        return "it rises" if o["rising"] else "it falls"
    if task == "order":
        return "the beep" if o["tone_first"] else "the burst of noise"
    return "they speed up" if o["speeds_up"] else "they slow down"


GENERATORS = {"bursts": gen_bursts, "pitch": gen_pitch, "order": gen_order, "tempo": gen_tempo}
