"""A self-contained HTML "benchmark health report".

One file, inline CSS, figures embedded as base64, no scripts and no external
requests. It reads ``results/summary.json`` and the figures drawn from it, and
turns the seven reliability checks into verdicts by fixed, stated rules.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

VERDICT_WORD = {"ok": "Holds", "caution": "Caution", "problem": "Problem"}

CSS = """
:root{--ink:#1f2430;--muted:#5a6678;--rule:#d9dde3;--paper:#fbfaf7;--card:#ffffff;--blue:#2f5d9b;--red:#ba3d1a;--green:#2f7a58;--amber:#8a6100}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:56px 28px 80px}
h1,h2,h3{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;font-weight:600;line-height:1.2;margin:0}
h1{font-size:2.3rem;letter-spacing:-.01em}
h2{font-size:1.5rem;margin-top:64px;padding-top:20px;border-top:1px solid var(--rule)}
h3{font-size:1.12rem;margin-top:36px}
p{margin:12px 0;max-width:70ch}
.kicker{font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
.lede{font-size:1.14rem;color:#333a47;max-width:66ch}
.meta{color:var(--muted);font-size:.9rem;margin-top:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px;margin:26px 0 8px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:16px 18px 14px;border-left:4px solid var(--muted)}
.card.ok{border-left-color:var(--green)}.card.caution{border-left-color:var(--amber)}.card.problem{border-left-color:var(--red)}
.card h4{margin:0 0 2px;font-size:.98rem;font-weight:600}
.card p{margin:6px 0 0;font-size:.9rem;color:#3a4150;line-height:1.5}
.verdict{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px}
.ok .verdict{color:var(--green)}.caution .verdict{color:var(--amber)}.problem .verdict{color:var(--red)}
.verdict::before{content:"";display:inline-block;width:9px;height:9px;margin-right:7px;border-radius:50%;background:currentColor}
.caution .verdict::before{border-radius:0;transform:rotate(45deg) scale(.85)}
.problem .verdict::before{border-radius:1px}
table{border-collapse:collapse;width:100%;margin:18px 0;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td.l,th.l{text-align:left;white-space:normal}
thead th{font-weight:600;color:var(--muted);font-size:.78rem;letter-spacing:.04em;text-transform:uppercase;border-bottom:1.5px solid var(--ink)}
tbody tr:last-child td{border-bottom:none}
.scroll{overflow-x:auto}
figure{margin:22px 0 8px;background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:14px}
figure img{display:block;max-width:100%;height:auto;margin:0 auto}
figcaption{font-size:.85rem;color:var(--muted);margin-top:10px;max-width:80ch}
.tag{display:inline-block;font-size:.74rem;font-weight:600;padding:1px 8px;border-radius:10px;border:1px solid currentColor}
.tag.floor,.tag.failed{color:var(--red)}.tag.ceiling,.tag.partly{color:var(--amber)}.tag.informative,.tag.supported{color:var(--green)}.tag.na{color:var(--muted)}
.note{font-size:.88rem;color:var(--muted)}
code{font:0.86em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f0eee8;padding:1px 5px;border-radius:3px}
footer{margin-top:72px;padding-top:18px;border-top:1px solid var(--rule);color:var(--muted);font-size:.85rem}
@media (max-width:640px){main{padding:32px 16px 56px}h1{font-size:1.8rem}}
@media print{body{background:#fff}figure,.card{break-inside:avoid}}
"""


def _e(text) -> str:
    return html.escape(str(text))


def _pc(x, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _pp(x, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f} pp"


def _num(x, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _rel(x) -> str:
    """A reliability coefficient. Negative values carry no meaning beyond "none"."""
    if x is None:
        return "n/a"
    return "below 0" if x < 0 else f"{x:.2f}"


def _img(img_dir: Path, name: str, caption: str) -> str:
    path = img_dir / name
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img alt="{_e(caption)}" src="data:image/png;base64,{data}"><figcaption>{_e(caption)}</figcaption></figure>'


def _table(head: list[str], rows: list[list[str]], left: tuple[int, ...] = ()) -> str:
    th = "".join(f'<th class="l">{h}</th>' if i in left else f"<th>{h}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="l">{c}</td>' if i in left else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def verdicts(block: dict) -> list[dict]:
    """The seven checks as verdicts. The rules are fixed and stated in the report."""
    rank, seeds, tmpl, items = block["ranking_pooled"], block["seeds"], block["templates"], block["items"]
    fc = block["floor_ceiling_pooled"]
    n_suite = str(block["design"]["n_items_per_task_per_suite"])
    need = block["items_needed"]["resampled"].get(n_suite)
    top_k = block["items_needed"]["top_k"]
    top_word = {1: "top model", 2: "top two", 3: "top three"}.get(top_k, f"top {top_k}")
    out = []

    share = rank["n_pairs_separated"] / rank["n_pairs"]
    out.append({
        "title": "Ranking stability",
        "level": "ok" if share >= 0.8 else ("caution" if share >= 0.5 else "problem"),
        "text": f"{rank['n_pairs_separated']} of {rank['n_pairs']} model pairs are separated by a paired bootstrap on "
                f"{rank['n_items']:,} items. The top model stays on top in {_pc(rank['p_same_top1'], 0)} of resamples; "
                f"the complete order survives in {_pc(rank['p_same_order'], 0)}.",
    })
    if need:
        level = "ok" if need["p_same_top1"] >= 0.95 and need["p_same_topk"] >= 0.9 else ("caution" if need["p_same_top1"] >= 0.8 else "problem")
        out.append({
            "title": "Suite size",
            "level": level,
            "text": f"Two independent suites of {n_suite} items per task name the same top model {_pc(need['p_same_top1'], 0)} of the time, "
                    f"the same {top_word} {_pc(need['p_same_topk'], 0)}, and the same complete order {_pc(need['p_same_order'], 0)}.",
        })
    consistent = seeds["share_cells_rejected"] <= 0.10
    out.append({
        "title": "Regeneration",
        "level": "problem" if not consistent else ("ok" if seeds["mean_tau"] >= 0.8 else "caution"),
        "text": f"Across {len(seeds['suites'])} regenerated suites the overall score moves by {_pp(seeds['mean_sd'], 2)} (standard deviation; "
                f"sampling alone predicts {_pp(seeds['mean_sd_expected'], 2)}). {seeds['n_cells_rejected']} of {seeds['n_cells']} model-task cells "
                f"differ between seeds more than chance allows. Mean rank agreement between suites: tau {_num(seeds['mean_tau'])}.",
    })
    gap = rank["median_adjacent_gap"]
    ratio = tmpl["mean_range"] / gap if gap else float("inf")
    out.append({
        "title": "Wording sensitivity",
        "level": "ok" if ratio <= 1 else ("caution" if ratio <= 2 else "problem"),
        "text": f"Rewording the captions moves a model's overall score by {_pp(tmpl['mean_range'])} on average (largest: {_pp(tmpl['max_range'])}). "
                f"The median gap between neighbouring models is {_pp(gap)}. Rank agreement between wordings: tau {_num(tmpl['mean_tau'])}.",
    })
    exp = items["expected_anti_correlated_by_chance"]
    dead = (items["n_none_right"] + items["n_all_right"]) / items["n_items"]
    out.append({
        "title": "Item quality",
        "level": "ok" if items["n_anti_correlated"] <= 1.25 * exp and dead <= 0.25 else "caution",
        "text": f"Of {items['n_items']:,} items, {items['n_none_right']} are answered by no model and {items['n_all_right']} by every model, so they "
                f"cannot separate anyone. {items['n_anti_correlated']} items run against overall ability; {exp:.0f} would do so by chance alone.",
    })
    floor = [t for t, b in fc.items() if b["status"] == "floor"]
    ceiling = [t for t, b in fc.items() if b["status"] == "ceiling"]
    dead = len(floor) + len(ceiling)
    parts = []
    if floor:
        parts.append(f"At floor (no model above chance): {', '.join(floor)}.")
    if ceiling:
        parts.append(f"At ceiling (every model at or above 95%): {', '.join(ceiling)}.")
    if not dead:
        parts.append("No task is at floor or at ceiling.")
    parts.append(f"{len(fc) - dead} of {len(fc)} tasks can contribute to a ranking.")
    out.append({
        "title": "Floor and ceiling",
        "level": "ok" if dead == 0 else ("caution" if dead <= len(fc) / 3 else "problem"),
        "text": " ".join(parts),
    })
    live = [p for p in block["redundancy"]["pairs"] if p["spearman"] is not None and fc[p["a"]]["status"] == "informative" and fc[p["b"]]["status"] == "informative"]
    if live:
        top = max(live, key=lambda p: p["spearman"])
        out.append({
            "title": "Task redundancy",
            "level": "caution" if top["spearman"] >= 0.9 else "ok",
            "text": f"Among informative tasks, the most similar pair is {top['a']} and {top['b']} (Spearman {top['spearman']:+.2f}, "
                    f"95% interval {top['lo']:+.2f} to {top['hi']:+.2f}). With this few models the interval is wide.",
        })
    return out


def _section(summary: dict, modality: str, img_dir: Path) -> str:
    block = summary[modality]
    d = block["design"]
    models = block["ranking_pooled"]["ranking"]
    label = {m: summary["models"][m]["label"] for m in d["models"]}
    tasks = d["tasks"]
    fc = block["floor_ceiling_pooled"]
    name = {"vision": "Image suite", "audio": "Audio suite"}[modality]
    parts = [f'<h2 id="{modality}">{name}</h2>']
    parts.append(
        f"<p>{len(d['models'])} models, {len(d['suites'])} suites generated from seeds {', '.join(str(s) for s in d['seeds'])}, "
        f"{d['n_items_per_task_per_suite']} items per task per suite, {d['n_items_pooled']:,} items in all. "
        f"Scores use the canonical caption wording unless stated.</p>"
    )
    parts.append('<div class="grid">')
    for v in verdicts(block):
        parts.append(f'<div class="card {v["level"]}"><span class="verdict">{VERDICT_WORD[v["level"]]}</span><h4>{_e(v["title"])}</h4><p>{_e(v["text"])}</p></div>')
    parts.append("</div>")

    parts.append("<h3>Scores and ranks</h3>")
    rows = []
    for m in models:
        r = block["ranking_pooled"]["models"][m]
        lo, hi = int(r["rank_lo"]), int(round(r["rank_hi"]))
        rows.append([_e(label[m]), _pc(r["score"]), f"{_pc(r['score_lo'])} to {_pc(r['score_hi'])}", str(lo) if lo == hi else f"{lo} to {hi}"]
                    + [_pc(block["task_accuracy_pooled"][m][t], 0) for t in tasks])
    parts.append(_table(["Model", "Overall", "95% interval", "Rank"] + tasks, rows))
    if modality == "vision":
        parts.append(_img(img_dir, "headline.png", "Overall score of each model with a 95% bootstrap interval, and the interval on its rank."))
        parts.append(_img(img_dir, "pairwise.png", "Paired differences between every two models. A pair is separated when the interval excludes zero."))

    parts.append("<h3>Which tasks carry signal</h3>")
    rows = []
    for t in tasks:
        b = fc[t]
        rows.append([
            _e(t), f'<span class="tag {b["status"]}">{b["status"]}</span>', _pc(b["chance"], 0), _pc(b["min_accuracy"]), _pc(b["max_accuracy"]),
            f"{b['n_models_above_chance']} of {len(models)}", f"{b['n_pairs_separated']} of {b['n_pairs']}",
            _rel(block["items"]["tasks"][t]["kr20"]), _num(block["items"]["tasks"][t]["mean_discrimination"]),
        ])
    parts.append(_table(["Task", "Status", "Chance", "Weakest", "Strongest", "Above chance", "Pairs separated", "KR-20", "Item-ability r"], rows))
    parts.append(_img(img_dir, f"tasks_{modality}.png", "Accuracy per task and model with 95% Wilson intervals. The dashed line is the chance level."))
    if modality == "vision":
        parts.append(_img(img_dir, "levels.png", "Accuracy at each difficulty level. The band runs from the weakest to the strongest model."))

    parts.append("<h3>Regenerated suites and reworded captions</h3>")
    rows = []
    for m in models:
        s, t = block["seeds"], block["templates"]
        rows.append([_e(label[m]), _pp(s["sd"][m], 2), _pp(s["sd_expected_from_sampling"][m], 2), _pp(s["range"][m]), _pp(t["sd"][m], 2), _pp(t["range"][m]), _pc(t["flip_rate"][m], 0)])
    parts.append(_table(["Model", "Seeds: SD", "expected", "range", "Wordings: SD", "range", "Items that flip"], rows))
    if modality == "vision":
        parts.append(_img(img_dir, "seeds_vs_templates.png", "Each model's overall score on five regenerated suites, and on one suite under three caption wordings."))
        parts.append("<h3>How many items are enough</h3>")
        need = block["items_needed"]["resampled"]
        rows = [[f"{int(k):,}", _num(v["mean_tau"]), _pc(v["p_same_top1"], 0), _pc(v["p_same_topk"], 0), _pc(v["p_same_order"], 0)] for k, v in sorted(need.items(), key=lambda kv: int(kv[0]))]
        parts.append("<p>Two suites of the given size are drawn independently, with replacement, from the pooled items, and their rankings compared.</p>")
        parts.append(_table(["Items per task", "Kendall tau", "Same top model", "Same top three", "Same complete order"], rows))
        parts.append(_img(img_dir, "items_needed.png", "Agreement between two suites as their size grows. Left: independent resamples of the pool. Right: two disjoint parts of the pool, which understates agreement as the parts approach half the pool."))
        parts.append("<h3>Items and tasks</h3>")
        parts.append(_img(img_dir, "items.png", "Left: mean correlation between getting an item right and overall ability. Right: items that run against ability, against the number expected by chance."))
        parts.append(_img(img_dir, "redundancy.png", "Rank correlation between tasks of the models' accuracies."))
    worst = block["items"]["most_anti_correlated"][:6]
    if worst:
        rows = [[f"<code>{_e(w['item'])}</code>", _num(w["discrimination"]), f"{w['n_right']} of {block['items']['n_models']}"] for w in worst]
        parts.append("<p class=\"note\">Items most at odds with overall ability. Each can be rebuilt from its suite seed and identifier.</p>")
        parts.append(_table(["Item", "Item-ability r", "Models right"], rows))
    return "\n".join(parts)


HYPOTHESIS_TEXT = {
    "H1": "Regenerated suites give the same ranking and scores consistent with sampling noise.",
    "H2": "Rewording captions moves scores more than regenerating the items does.",
    "H3": "The effect of wording exceeds the typical gap between neighbouring models.",
    "H4": "At least two image tasks are at floor (spatial and binding expected).",
    "H5": "The control task is at ceiling at its easiest level.",
    "H6": "One suite's worth of items fixes the top of the ranking but not the complete order.",
    "H7": "Items in floor tasks do not follow ability; items in informative tasks do.",
    "H8": "Accuracy falls with the difficulty level in every task that is not at floor.",
    "H9": "At least two audio tasks are at floor (order and tempo expected).",
}


def build_report(results_root: Path, img_dir: Path, out_path: Path) -> Path:
    with open(Path(results_root) / "summary.json") as f:
        summary = json.load(f)
    img_dir = Path(img_dir)
    modalities = [m for m in ("vision", "audio") if m in summary]
    total_items = sum(summary[m]["design"]["n_items_pooled"] for m in modalities)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Benchmark health report</title>",
        f"<style>{CSS}</style></head><body><main>",
        '<div class="kicker">Benchmark health report</div>',
        "<h1>Can this benchmark be trusted to rank models?</h1>",
        '<p class="lede">A benchmark is a measuring instrument. This report checks the instrument itself: whether its ranking is stable, '
        "whether a freshly generated copy gives the same answer, how much the wording of the captions matters, and which tasks and items carry any signal.</p>",
        f'<p class="meta">Run <code>{_e(summary["run_id"])}</code> &middot; {len(summary["models"])} models &middot; {total_items:,} items &middot; '
        f'{summary["analysis"]["n_boot"]:,} bootstrap resamples &middot; analysis seed {summary["analysis"]["seed"]}</p>',
        _img(img_dir, "gallery_vision.png", "One generated item per image task and difficulty level, with its true caption and one wrong candidate."),
    ]
    for modality in modalities:
        parts.append(_section(summary, modality, img_dir))
        if modality == "audio":
            parts.append(_img(img_dir, "gallery_audio.png", "Spectrograms of one generated clip per audio task and difficulty level, with the true caption."))

    hyp = summary.get("hypotheses", {})
    if hyp:
        parts.append('<h2 id="hypotheses">Hypotheses written before the run</h2>')
        css = {"supported": "supported", "failed": "failed", "partly supported": "partly", "not testable": "na"}
        rows = [[k, _e(HYPOTHESIS_TEXT.get(k, "")), f'<span class="tag {css.get(v["outcome"], "na")}">{_e(v["outcome"])}</span>'] for k, v in sorted(hyp.items())]
        parts.append(_table(["", "Prediction", "Outcome"], rows, left=(1,)))

    parts.append('<h2 id="rules">How the verdicts are decided</h2>')
    rules = [
        ["Ranking stability", "Share of model pairs whose paired-bootstrap 95% interval excludes zero.", "at least 80%", "at least 50%"],
        ["Suite size", "Agreement between two suites of the size actually used, resampled independently from the pool.", "top model 95% and top group 90%", "top model 80%"],
        ["Regeneration", "Chi-square test that each model-task accuracy is the same in every seed; rank agreement between seeds.", "at most 10% of cells rejected and tau at least 0.8", "at most 10% of cells rejected"],
        ["Wording sensitivity", "Mean range of the overall score across caption templates, against the median gap between neighbouring models.", "range no larger than the gap", "no larger than twice the gap"],
        ["Item quality", "Items with item-ability correlation at or below the threshold, against an exact permutation count; and the share of items that every model answers alike.", "no more than 1.25 times chance, and at most a quarter of items answered alike", "otherwise"],
        ["Floor and ceiling", "Floor: no model above chance (exact binomial, Holm-adjusted). Ceiling: every model at 95% or more.", "no such task", "at most a third of tasks"],
        ["Task redundancy", "Largest Spearman correlation between two informative tasks.", "below 0.9", "0.9 or more"],
    ]
    parts.append(_table(["Check", "Measured by", "Holds if", "Caution if"], rules, left=(1, 2, 3)))
    parts.append('<p class="note">Anything short of the caution rule is reported as a problem. The rules are conventions chosen for this report, not established standards.</p>')

    parts.append('<h2 id="models">Models</h2>')
    rows = [[_e(v["label"]), f"<code>{_e(v['identifier'])}</code>", _e(v["architecture"]), _e(v["training_data"]), _e(v["modality"])] for v in summary["models"].values()]
    parts.append(_table(["Model", "Checkpoint", "Architecture", "Training data", "Input"], rows, left=(1, 2, 3)))
    parts.append("<footer>Generated by <code>regenbench report</code> from <code>results/summary.json</code>. Zero-shot caption matching only; no generative model was evaluated.</footer>")
    parts.append("</main></body></html>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(p for p in parts if p) + "\n")
    return out_path
