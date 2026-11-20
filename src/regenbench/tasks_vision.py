"""Image tasks.

Every generator takes a seeded random generator and a difficulty level and
returns the scene, the structured options, the index of the true option and a
question. Captions are built from the structured options by the templates at
the bottom of the file, so the same item can be worded in several ways.

Shapes never overlap and never touch the border: each one is kept inside its
circumscribed circle, and circles are kept apart by a gap.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .render import BACKGROUND, COLOURS, SHAPES

SIZE = 224
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
COLOUR_NAMES = tuple(COLOURS)
INVERSE = {"left": "right", "right": "left", "above": "below", "below": "above"}
RELATION_PHRASE = {"left": "to the left of", "right": "to the right of", "above": "above", "below": "below"}
HARD_NOISE = 14.0  # standard deviation of pixel noise at level 3


class PlacementError(RuntimeError):
    pass


def _base_scene(rng: np.random.Generator, level: int) -> dict[str, Any]:
    return {
        "size": SIZE,
        "background": list(BACKGROUND),
        "noise": HARD_NOISE if level == 3 else 0.0,
        "noise_seed": int(rng.integers(0, 2**31 - 1)),
        "objects": [],
    }


def _free(cx: float, cy: float, r: float, placed: list[dict], gap: float) -> bool:
    if not (r + 4 <= cx <= SIZE - r - 4 and r + 4 <= cy <= SIZE - r - 4):
        return False
    return all((cx - o["cx"]) ** 2 + (cy - o["cy"]) ** 2 >= (r + o["r"] + gap) ** 2 for o in placed)


def _scatter(rng: np.random.Generator, specs: list[dict], placed: list[dict] | None = None, gap: float = 6.0) -> list[dict]:
    """Place objects (dicts with shape, colour, r) at random without overlap."""
    for _ in range(200):
        out = list(placed or [])
        ok = True
        for spec in specs:
            for _ in range(400):
                r = spec["r"]
                cx = float(rng.uniform(r + 4, SIZE - r - 4))
                cy = float(rng.uniform(r + 4, SIZE - r - 4))
                if _free(cx, cy, r, out, gap):
                    out.append(dict(spec, cx=round(cx, 2), cy=round(cy, 2)))
                    break
            else:
                ok = False
                break
        if ok:
            return out[len(placed or []) :]
    raise PlacementError("could not place objects without overlap")


def _pick(rng: np.random.Generator, pool, k: int = 1, exclude=()) -> list:
    pool = [p for p in pool if p not in exclude]
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in idx]


def _distractors(rng, n: int, r_range, colours_out, shapes_out) -> list[dict]:
    """Extra objects that share no colour and no shape with the tested ones."""
    specs = []
    for _ in range(n):
        specs.append(
            {
                "shape": _pick(rng, SHAPES, exclude=shapes_out)[0],
                "colour": _pick(rng, COLOUR_NAMES, exclude=colours_out)[0],
                "r": round(float(rng.uniform(*r_range)), 2),
                "role": "distractor",
            }
        )
    return specs


def _shuffle(rng: np.random.Generator, options: list[dict]) -> tuple[list[dict], int]:
    """Shuffle options whose first element is the true one; return the new index."""
    order = rng.permutation(len(options))
    return [options[int(i)] for i in order], int(np.argwhere(order == 0)[0, 0])


# --------------------------------------------------------------------------
# object: one shape, name its colour and shape. A control task.


def gen_object(rng: np.random.Generator, level: int):
    colour, other_colour = _pick(rng, COLOUR_NAMES, 2)
    shape, other_shape = _pick(rng, SHAPES, 2)
    scene = _base_scene(rng, level)
    if level == 1:
        r = float(rng.uniform(52, 70))
        cx = cy = SIZE / 2
    else:
        r = float(rng.uniform(26, 40)) if level == 2 else float(rng.uniform(13, 20))
        cx, cy = (float(v) for v in rng.uniform(r + 4, SIZE - r - 4, 2))
    scene["objects"] = [{"shape": shape, "colour": colour, "r": round(r, 2), "cx": round(cx, 2), "cy": round(cy, 2), "role": "target"}]
    options = [
        {"colour": colour, "shape": shape},
        {"colour": colour, "shape": other_shape},
        {"colour": other_colour, "shape": shape},
        {"colour": other_colour, "shape": other_shape},
    ]
    options, answer = _shuffle(rng, options)
    return scene, options, answer, "What is shown in the image?", {"colour": colour, "shape": shape}


# --------------------------------------------------------------------------
# count: how many shapes of one kind.

COUNT_RANGE = {1: (1, 4), 2: (2, 7), 3: (3, 9)}
COUNT_RADIUS = {1: (20, 20), 2: (13, 21), 3: (10, 17)}
MAX_COUNT_OPTION = 10


def gen_count(rng: np.random.Generator, level: int):
    lo, hi = COUNT_RANGE[level]
    n = int(rng.integers(lo, hi + 1))
    colour = _pick(rng, COLOUR_NAMES)[0]
    shape = _pick(rng, SHAPES)[0]
    r_lo, r_hi = COUNT_RADIUS[level]
    specs = [{"shape": shape, "colour": colour, "r": round(float(rng.uniform(r_lo, r_hi)), 2), "role": "target"} for _ in range(n)]
    if level == 3:
        specs += _distractors(rng, int(rng.integers(2, 5)), (r_lo, r_hi), [colour], [shape])
    scene = _base_scene(rng, level)
    scene["objects"] = _scatter(rng, specs)
    # Four consecutive counts that include the true one.
    start = int(rng.integers(max(1, n - 3), min(n, MAX_COUNT_OPTION - 3) + 1))
    counts = [n] + [c for c in range(start, start + 4) if c != n]
    options = [{"n": c, "colour": colour, "shape": shape} for c in counts]
    options, answer = _shuffle(rng, options)
    return scene, options, answer, f"How many {colour} {shape}s are in the image?", {"n": n, "colour": colour, "shape": shape}


# --------------------------------------------------------------------------
# binding: which colour goes with which shape.

BINDING_RADIUS = {1: (32, 40), 2: (20, 30), 3: (15, 24)}


def gen_binding(rng: np.random.Generator, level: int):
    ca, cb = _pick(rng, COLOUR_NAMES, 2)
    sa, sb = _pick(rng, SHAPES, 2)
    r_lo, r_hi = BINDING_RADIUS[level]
    specs = [
        {"shape": sa, "colour": ca, "r": round(float(rng.uniform(r_lo, r_hi)), 2), "role": "first"},
        {"shape": sb, "colour": cb, "r": round(float(rng.uniform(r_lo, r_hi)), 2), "role": "second"},
    ]
    if level == 3:
        specs += _distractors(rng, 2, (r_lo, r_hi), [ca, cb], [sa, sb])
    scene = _base_scene(rng, level)
    scene["objects"] = _scatter(rng, specs)
    options = [
        {"first": {"colour": ca, "shape": sa}, "second": {"colour": cb, "shape": sb}},
        {"first": {"colour": cb, "shape": sa}, "second": {"colour": ca, "shape": sb}},
    ]
    options, answer = _shuffle(rng, options)
    truth = {"pairs": sorted([[ca, sa], [cb, sb]])}
    return scene, options, answer, "Which description matches the image?", truth


# --------------------------------------------------------------------------
# spatial: where one object is relative to another.

SPATIAL = {  # radius range, separation range along the main axis, largest sideways offset as a share of it
    1: ((28, 36), (96, 120), 0.08),
    2: ((20, 28), (62, 96), 0.20),
    3: ((14, 22), (48, 80), 0.30),
}


def gen_spatial(rng: np.random.Generator, level: int):
    r_range, sep_range, skew = SPATIAL[level]
    ca, cb = _pick(rng, COLOUR_NAMES, 2)
    sa, sb = _pick(rng, SHAPES, 2)
    relation = ["left", "right", "above", "below"][int(rng.integers(0, 4))]
    ra, rb = (round(float(v), 2) for v in rng.uniform(*r_range, 2))
    for _ in range(1000):
        sep = float(rng.uniform(*sep_range))
        side = float(rng.uniform(-skew, skew)) * sep
        # Vector from the first object to the second, in image coordinates (y grows downwards).
        dx, dy = {"left": (sep, side), "right": (-sep, side), "above": (side, sep), "below": (side, -sep)}[relation]
        ax, ay = (float(v) for v in rng.uniform(ra + 4, SIZE - ra - 4, 2))
        bx, by = ax + dx, ay + dy
        if _free(bx, by, rb, [{"cx": ax, "cy": ay, "r": ra}], gap=6.0):
            break
    else:
        raise PlacementError("could not place the pair")
    pair = [
        {"shape": sa, "colour": ca, "r": ra, "cx": round(ax, 2), "cy": round(ay, 2), "role": "first"},
        {"shape": sb, "colour": cb, "r": rb, "cx": round(bx, 2), "cy": round(by, 2), "role": "second"},
    ]
    extra = []
    if level == 3:
        extra = _scatter(rng, _distractors(rng, 2, r_range, [ca, cb], [sa, sb]), placed=pair)
    scene = _base_scene(rng, level)
    scene["objects"] = pair + extra
    first, second = {"colour": ca, "shape": sa}, {"colour": cb, "shape": sb}
    options = [
        {"first": first, "second": second, "relation": relation},
        {"first": first, "second": second, "relation": INVERSE[relation]},
    ]
    options, answer = _shuffle(rng, options)
    question = f"Where is the {ca} {sa} relative to the {cb} {sb}?"
    return scene, options, answer, question, {"relation": relation}


# --------------------------------------------------------------------------
# size: which of two same-shaped objects is larger.

SIZE_RATIO = {1: (2.0, 2.6), 2: (1.5, 1.8), 3: (1.25, 1.4)}


def gen_size(rng: np.random.Generator, level: int):
    ca, cb = _pick(rng, COLOUR_NAMES, 2)
    shape = _pick(rng, SHAPES)[0]
    ratio = float(rng.uniform(*SIZE_RATIO[level]))
    small = float(rng.uniform(16, 22))
    first_is_large = bool(rng.integers(0, 2))
    ra, rb = (small * ratio, small) if first_is_large else (small, small * ratio)
    specs = [
        {"shape": shape, "colour": ca, "r": round(ra, 2), "role": "first"},
        {"shape": shape, "colour": cb, "r": round(rb, 2), "role": "second"},
    ]
    scene = _base_scene(rng, level)
    scene["objects"] = _scatter(rng, specs)
    first, second = {"colour": ca, "shape": shape}, {"colour": cb, "shape": shape}
    options = [
        {"first": first, "second": second, "first_is_large": first_is_large},
        {"first": first, "second": second, "first_is_large": not first_is_large},
    ]
    options, answer = _shuffle(rng, options)
    question = f"Which is larger, the {ca} {shape} or the {cb} {shape}?"
    return scene, options, answer, question, {"larger": ca if first_is_large else cb, "ratio": round(ratio, 3)}


# --------------------------------------------------------------------------
# digits: read a rendered number.

DIGITS = {1: (2, 96), 2: (3, 64), 3: (4, 40)}  # number of digits, font size


def _one_digit_off(rng: np.random.Generator, s: str) -> str:
    pos = int(rng.integers(0, len(s)))
    pool = [d for d in "0123456789" if d != s[pos] and not (pos == 0 and d == "0")]
    return s[:pos] + pool[int(rng.integers(0, len(pool)))] + s[pos + 1 :]


def gen_digits(rng: np.random.Generator, level: int):
    n_digits, font = DIGITS[level]
    lo, hi = 10 ** (n_digits - 1), 10**n_digits
    truth = str(int(rng.integers(lo, hi)))
    foils: list[str] = []
    while len(foils) < 3:
        # Level 1 foils are unrelated numbers; harder levels differ from the truth in one digit.
        cand = str(int(rng.integers(lo, hi))) if level == 1 else _one_digit_off(rng, truth)
        if cand != truth and cand not in foils:
            foils.append(cand)
    scene = _base_scene(rng, level)
    jitter = {1: 8, 2: 16, 3: 36}[level]
    cx, cy = (SIZE / 2 + float(v) for v in rng.uniform(-jitter, jitter, 2))
    grey = 20 if level < 3 else 70
    scene["text"] = {"string": truth, "font_size": font, "cx": round(cx, 2), "cy": round(cy, 2), "fill": [grey, grey, grey]}
    options, answer = _shuffle(rng, [{"string": truth}] + [{"string": f} for f in foils])
    return scene, options, answer, "Which number is written in the image?", {"string": truth}


# --------------------------------------------------------------------------
# Caption templates. Template 0 is the canonical wording; 1 and 2 are paraphrases.


def _np(o: dict) -> str:
    return f"{o['colour']} {o['shape']}"


def _anp(o: dict) -> str:
    """Noun phrase with its article: "a red star", "an orange star"."""
    return f"{'an' if o['colour'][0] in 'aeiou' else 'a'} {_np(o)}"


def _counted(o: dict, numeral: bool) -> str:
    n = o["n"]
    return f"{n if numeral else NUMBER_WORDS[n]} {o['colour']} {o['shape']}{'' if n == 1 else 's'}"


def _size_words(o: dict) -> tuple[str, str]:
    return ("large", "small") if o["first_is_large"] else ("small", "large")


TEMPLATES = {
    "object": [
        lambda o: f"a photo of {_anp(o)}",
        lambda o: f"an image of a single {_np(o)}",
        lambda o: f"a picture showing one {o['shape']} that is {o['colour']}",
    ],
    "count": [
        lambda o: f"a photo of {_counted(o, False)}",
        lambda o: f"an image containing exactly {_counted(o, False)}",
        lambda o: f"a picture showing {_counted(o, True)}",
    ],
    "binding": [
        lambda o: f"a photo of {_anp(o['first'])} and {_anp(o['second'])}",
        lambda o: f"an image containing {_anp(o['first'])} and {_anp(o['second'])}",
        lambda o: (
            f"a picture showing one {o['first']['shape']} that is {o['first']['colour']} "
            f"and one {o['second']['shape']} that is {o['second']['colour']}"
        ),
    ],
    "spatial": [
        lambda o: f"a photo of {_anp(o['first'])} {RELATION_PHRASE[o['relation']]} {_anp(o['second'])}",
        lambda o: f"an image in which {_anp(o['first'])} is {RELATION_PHRASE[o['relation']]} {_anp(o['second'])}",
        # Same fact with subject and object exchanged.
        lambda o: f"a picture showing {_anp(o['second'])} {RELATION_PHRASE[INVERSE[o['relation']]]} {_anp(o['first'])}",
    ],
    "size": [
        lambda o: f"a photo of a {_size_words(o)[0]} {_np(o['first'])} and a {_size_words(o)[1]} {_np(o['second'])}",
        lambda o: (
            f"an image in which the {_np(o['first'])} is "
            f"{'bigger' if o['first_is_large'] else 'smaller'} than the {_np(o['second'])}"
        ),
        lambda o: (
            f"a picture showing {_anp(o['second'])} that is "
            f"{'smaller' if o['first_is_large'] else 'larger'} than the {_np(o['first'])}"
        ),
    ],
    "digits": [
        lambda o: f"a photo of the number {o['string']}",
        lambda o: f"an image with the digits {o['string']} written on it",
        lambda o: f'a picture showing the text "{o["string"]}"',
    ],
}


def _short(task: str, o: dict) -> str:
    """Short answer text for the question/answer rendering."""
    if task == "count":
        return NUMBER_WORDS[o["n"]]
    if task == "digits":
        return o["string"]
    if task == "spatial":
        return RELATION_PHRASE[o["relation"]].replace(" of", "")
    if task == "size":
        return f"the {_np(o['first'] if o['first_is_large'] else o['second'])}"
    if task == "object":
        return f"{_anp(o)}"
    return f"{_anp(o['first'])} and {_anp(o['second'])}"


GENERATORS = {
    "object": gen_object,
    "count": gen_count,
    "binding": gen_binding,
    "spatial": gen_spatial,
    "size": gen_size,
    "digits": gen_digits,
}
SHORT = _short
