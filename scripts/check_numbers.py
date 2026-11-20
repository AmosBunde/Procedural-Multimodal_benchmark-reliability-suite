"""Check that every number quoted in README.md and REPORT.md comes from the results.

A number in the prose is accepted if it equals a value in results/summary.json (or in
results/run_notes.json) after ordinary rounding: as it stands, or as a percentage, to
0, 1 or 2 decimals. Numbers that describe the design and not a result (image size,
sample rate, citation years and so on) are listed in ALLOWED with the reason.

Run: uv run python scripts/check_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED = {
    # design constants stated in the code
    "224": "image side in pixels",
    "48": "audio sample rate in kHz",
    "10": "clip length in seconds; items per audio level; Holm level",
    "77": "CLIP text context length",
    "25": "caption length bound in tokens; chance level in per cent",
    "50": "chance level in per cent",
    "95": "confidence level; ceiling threshold",
    "5": "significance level in per cent",
    "0.05": "significance level; H7 tolerance",
    "0.10": "H7 threshold",
    "0.70": "H1 threshold",
    "0.4": "anti-correlation threshold",
    "-0.4": "anti-correlation threshold",
    "90": "H6 threshold in per cent; audio clips per task pooled",
    "80": "H6 threshold in per cent",
    "1.25": "size ratio bound; verdict rule",
    "1.4": "size ratio bound",
    "1.5": "size ratio bound",
    "1.8": "size ratio bound",
    "2.0": "size ratio bound",
    "2.6": "size ratio bound",
    "15": "pitch sweep in per cent; lines in the adapter example",
    "30": "pitch sweep in per cent; audio items per task; example suite size",
    "24": "image items per level per suite",
    "20": "tests lower bound; date",
    "21": "date",
    "2026": "year; example seed",
    "1000": "subsampling draws",
    "2000": "bootstrap resamples",
    "200": "split-half repetitions",
    "3.11": "Python version",
    "0.2": "",
}


def numbers_from_json(obj, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if re.fullmatch(r"\d+", k):  # suite sizes are stored as keys
                out.add(k)
                out.add(f"{int(k):,}")
            numbers_from_json(v, out)
    elif isinstance(obj, list):
        for v in obj:
            numbers_from_json(v, out)
    elif isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    elif isinstance(obj, int):
        out.add(str(obj))
        out.add(f"{obj:,}")
    elif isinstance(obj, float):
        for scale in (1.0, 100.0):
            for digits in (0, 1, 2, 3):
                text = f"{abs(obj) * scale:.{digits}f}"
                out.add(text)
                out.add(f"{obj * scale:.{digits}f}")
                if digits == 0:
                    out.add(f"{round(abs(obj) * scale):,}")


TOKEN = re.compile(r"(?<![\w.#/-])-?\d[\d,]*(?:\.\d+)?(?![\w/])")


def strip_non_prose(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # code blocks
    text = re.sub(r"`[^`]*`", " ", text)  # inline code
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)  # links and images
    text = re.sub(r"^## References.*", " ", text, flags=re.S | re.M)  # venues and years
    text = re.sub(r"^#+ .*$", " ", text, flags=re.M)  # headings carry section numbers
    return text


def main() -> int:
    allowed: set[str] = set(ALLOWED)
    for name in ("summary.json", "run_notes.json"):
        path = ROOT / "results" / name
        if path.exists():
            numbers_from_json(json.loads(path.read_text()), allowed)
    bad = []
    for doc in ("README.md", "REPORT.md"):
        prose = strip_non_prose((ROOT / doc).read_text())
        for lineno, line in enumerate(prose.splitlines(), 1):
            for match in TOKEN.finditer(line):
                token = match.group().rstrip(".,")
                plain = token.replace(",", "")
                if token in allowed or plain in allowed or plain.lstrip("-") in allowed:
                    continue
                if re.fullmatch(r"\d", plain):  # single digits: counts of tasks, levels, section numbers
                    continue
                bad.append((doc, token, line.strip()[:110]))
    for doc, token, line in bad:
        print(f"{doc}: {token!r} not found in the results  |  {line}")
    print(f"{len(bad)} unverified number(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
