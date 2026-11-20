"""The HTML health report."""

from regenbench.figures import make_figures
from regenbench.report import build_report, verdicts


def test_report_is_self_contained(run, tmp_path):
    tmp, _ = run
    make_figures(tmp / "results", tmp_path / "img")
    page = build_report(tmp / "results", tmp_path / "img", tmp_path / "report.html").read_text()
    assert page.startswith("<!DOCTYPE html>") and "Benchmark health report" in page
    # Images are embedded and nothing is fetched from elsewhere.
    assert page.count("data:image/png;base64,") >= 8
    assert "http://" not in page and "https://" not in page and "<script" not in page and "@import" not in page
    for name in ("strong", "middling", "fragile", "guesser"):
        assert name in page


def test_report_builds_without_figures(run, tmp_path):
    tmp, _ = run
    page = build_report(tmp / "results", tmp_path / "no-such-dir", tmp_path / "report.html").read_text()
    assert "<img" not in page and "Scores and ranks" in page


def test_verdicts_follow_the_stated_rules(run):
    _, summary = run
    cards = {v["title"]: v for v in verdicts(summary["vision"])}
    assert set(cards) >= {"Ranking stability", "Suite size", "Regeneration", "Wording sensitivity", "Item quality", "Floor and ceiling"}
    assert all(v["level"] in {"ok", "caution", "problem"} for v in cards.values())
    # Four fake models with very different skills, no dead task.
    assert cards["Ranking stability"]["level"] == "ok"
    assert cards["Floor and ceiling"]["level"] == "ok"
    # One model collapses under the third wording, which dwarfs the gaps between models.
    assert cards["Wording sensitivity"]["level"] in {"caution", "problem"}
