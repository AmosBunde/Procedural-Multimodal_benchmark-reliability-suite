"""Ground truth of the audio tasks, checked from the scene and from the waveform."""

import numpy as np

from regenbench.generate import generate_suite, task_names
from regenbench.render import render
from regenbench.tasks_audio import DURATION, NUMBER_WORDS, RATE

SEED = 2718
ITEMS = generate_suite("audio", SEED, 30)
BY_TASK = {t: [i for i in ITEMS if i.task == t] for t in task_names("audio")}


def _envelope(samples, rate, window=0.02):
    n = int(rate * window)
    frames = np.abs(samples[: len(samples) // n * n]).reshape(-1, n)
    return frames.mean(axis=1)


def _segments(active):
    """Number of runs of True in a boolean array."""
    padded = np.concatenate([[False], active])
    return int((~padded[:-1] & padded[1:]).sum())


def _dominant_frequency(samples, rate):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return np.fft.rfftfreq(len(samples), 1 / rate)[int(np.argmax(spectrum))]


def test_all_tasks_present_and_deterministic():
    assert set(BY_TASK) == {"bursts", "pitch", "order", "tempo"}
    assert generate_suite("audio", SEED, 30) == ITEMS


def test_clips_are_well_formed():
    for item in ITEMS[::5]:
        clip = render(item)
        assert clip.rate == RATE and clip.samples.shape == (int(RATE * DURATION),)
        assert clip.samples.dtype == np.float32 and np.abs(clip.samples).max() <= 1.0
        assert np.array_equal(clip.samples, render(item).samples)
        for ev in item.scene["events"]:
            assert ev["t0"] >= 0.49 and ev["t0"] + ev.get("dur", 0.012) <= DURATION - 0.49


def test_burst_count_matches_scene_and_waveform():
    for item in BY_TASK["bursts"]:
        n = item.truth["n"]
        events = item.scene["events"]
        assert len(events) == n and item.choices[item.answer] == NUMBER_WORDS[n]
        for a, b in zip(events, events[1:]):
            assert b["t0"] - (a["t0"] + a["dur"]) >= 0.19, "bursts must be separated by silence"
        clip = render(item)
        heard = _segments(_envelope(clip.samples, clip.rate) > 0.15)
        assert heard == n, f"{item.item_id}: {heard} bursts in the waveform, {n} in the caption"
        assert f"{NUMBER_WORDS[n]} beep" in item.captions[0][item.answer]


def test_pitch_direction_matches_waveform():
    for item in BY_TASK["pitch"]:
        (ev,) = item.scene["events"]
        clip = render(item)
        start, stop = int(ev["t0"] * RATE), int((ev["t0"] + ev["dur"]) * RATE)
        quarter = (stop - start) // 4
        early = _dominant_frequency(clip.samples[start : start + quarter], RATE)
        late = _dominant_frequency(clip.samples[stop - quarter : stop], RATE)
        assert (late > early) == item.truth["rising"], item.item_id
        assert (ev["f1"] > ev["f0"]) == item.truth["rising"]
        assert ("rising" in item.captions[0][item.answer]) == item.truth["rising"]
        assert max(ev["f0"], ev["f1"]) / min(ev["f0"], ev["f1"]) >= 1.14


def test_order_matches_waveform():
    for item in BY_TASK["order"]:
        first, second = item.scene["events"]
        assert first["t0"] + first["dur"] <= second["t0"]
        assert (first["kind"] == "tone") == item.truth["tone_first"]
        assert {first["kind"], second["kind"]} == {"tone", "noise"}
        clip = render(item)

        def crossings(ev):
            seg = clip.samples[int(ev["t0"] * RATE) : int((ev["t0"] + ev["dur"]) * RATE)]
            return np.mean(np.signbit(seg[:-1]) != np.signbit(seg[1:]))

        # White noise changes sign far more often than a tone below 1.5 kHz.
        assert (crossings(first) < crossings(second)) == item.truth["tone_first"], item.item_id
        assert item.captions[0][item.answer].startswith("a beep") == item.truth["tone_first"]


def test_tempo_direction_matches_click_spacing():
    for item in BY_TASK["tempo"]:
        times = np.array([ev["t0"] for ev in item.scene["events"]])
        gaps = np.diff(times)
        assert len(gaps) >= 10
        shrinking = np.all(np.diff(gaps) < 0)
        growing = np.all(np.diff(gaps) > 0)
        assert shrinking == item.truth["speeds_up"] and growing == (not item.truth["speeds_up"])
        assert ("speed up" in item.captions[0][item.answer]) == item.truth["speeds_up"]
        if item.level < 3:  # no background noise: clicks can be counted in the waveform
            clip = render(item)
            assert _segments(_envelope(clip.samples, clip.rate, 0.01) > 0.02) == len(times)
