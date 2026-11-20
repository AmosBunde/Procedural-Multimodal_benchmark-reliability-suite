"""Figures for the README, the report and the HTML health report.

Everything is drawn from ``results/summary.json``; the two galleries are drawn
from the generator itself. Two series colours carry the comparisons (blue and
red); grey marks reference values and green is used only where a fourth
series is unavoidable, always with a different marker and line style as well.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .generate import make_item, task_names  # noqa: E402
from .render import render  # noqa: E402

BLUE, RED, GREY, GREEN = "#2F5D9B", "#BA3D1A", "#5A6678", "#3F8F6B"
INK, MUTED, RULE = "#1F2430", "#5A6678", "#D9DDE3"
PALE_BLUE = "#B9C9E0"
GALLERY_SEED = 7

TASK_TITLES = {
    "object": "object (control)",
    "count": "count",
    "binding": "binding",
    "spatial": "spatial",
    "size": "size",
    "digits": "digits",
    "bursts": "bursts",
    "pitch": "pitch",
    "order": "order",
    "tempo": "tempo",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#ECEEF2",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.dpi": 100,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _save(fig, out_dir: Path, name: str, dpi: int | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _labels(summary: dict, models: list[str]) -> list[str]:
    return [summary["models"][m]["label"] for m in models]


def _pct(ax, axis: str = "x") -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{round(100 * v, 1):g}%")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


# ------------------------------------------------------------------ galleries


def fig_gallery_vision(out_dir: Path) -> Path:
    tasks = task_names("vision")
    fig, axes = plt.subplots(3, len(tasks), figsize=(13.2, 9.2))
    for col, task in enumerate(tasks):
        for row, level in enumerate((1, 2, 3)):
            item = make_item("vision", task, level, 0, GALLERY_SEED)
            ax = axes[row, col]
            ax.imshow(render(item))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(RULE)
            true = item.captions[0][item.answer]
            foil = item.captions[0][(item.answer + 1) % item.n_choices]
            ax.set_xlabel(
                textwrap.fill(true, 34) + "\n" + textwrap.fill("not: " + foil, 34),
                fontsize=7.4,
                color=INK,
                labelpad=5,
                linespacing=1.3,
            )
            if row == 0:
                ax.set_title(TASK_TITLES[task], loc="center", pad=8)
            if col == 0:
                ax.set_ylabel(f"level {level}", fontsize=9.5, color=MUTED, labelpad=8)
    fig.subplots_adjust(hspace=0.52, wspace=0.14)
    return _save(fig, out_dir, "gallery_vision.png", dpi=120)


def _spectrogram(samples: np.ndarray, rate: int, n_fft: int = 2048, hop: int = 1024):
    window = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(samples, n_fft)[::hop] * window
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return 10 * np.log10(power.T + 1e-9), rate / n_fft


def fig_gallery_audio(out_dir: Path) -> Path:
    tasks = task_names("audio")
    fig, axes = plt.subplots(3, len(tasks), figsize=(13.2, 7.2), sharex=True, sharey=True)
    cmap = LinearSegmentedColormap.from_list("ink", ["#FFFFFF", PALE_BLUE, BLUE, "#16294A"])
    for col, task in enumerate(tasks):
        for row, level in enumerate((1, 2, 3)):
            item = make_item("audio", task, level, 0, GALLERY_SEED)
            clip = render(item)
            spec, hz_per_bin = _spectrogram(clip.samples.astype(float), clip.rate)
            top = int(4000 / hz_per_bin)
            ax = axes[row, col]
            ax.imshow(spec[:top], origin="lower", aspect="auto", cmap=cmap, vmin=-35, vmax=45, extent=[0, 10, 0, 4])
            ax.grid(False)
            true = item.captions[0][item.answer]
            ax.text(0.02, 0.95, f'"{true}"', transform=ax.transAxes, va="top", fontsize=8, color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
            if row == 0:
                ax.set_title(TASK_TITLES[task], loc="center", pad=8)
            if col == 0:
                ax.set_ylabel(f"level {level}\nkHz", color=MUTED)
            if row == 2:
                ax.set_xlabel("seconds")
            ax.set_xticks([0, 2, 4, 6, 8])
    fig.subplots_adjust(hspace=0.12, wspace=0.06)
    return _save(fig, out_dir, "gallery_audio.png", dpi=120)


# -------------------------------------------------------------- result figures


def fig_headline(summary: dict, out_dir: Path) -> Path:
    v = summary["vision"]
    rank = v["ranking_pooled"]
    order = rank["ranking"][::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(order) + 1.5))
    for y, m in enumerate(order):
        row = rank["models"][m]
        ax.plot([row["score_lo"], row["score_hi"]], [y, y], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(row["score"], y, "o", color=BLUE, ms=7, mec="white", mew=1.2)
        lo, hi = int(row["rank_lo"]), int(round(row["rank_hi"]))
        where = f"rank {lo}" if lo == hi else f"rank {lo} to {hi}"
        ax.text(row["score_hi"] + 0.006, y, f"{row['score']:.1%}   {where}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(order)), _labels(summary, order))
    ax.tick_params(axis="y", labelcolor=INK, length=0)
    ax.grid(axis="y", visible=False)
    lo = min(r["score_lo"] for r in rank["models"].values())
    hi = max(r["score_hi"] for r in rank["models"].values())
    ax.set_xlim(lo - 0.02, hi + 0.12)
    _pct(ax)
    ax.set_xlabel("overall score: mean accuracy over the six image tasks")
    n = v["design"]["n_items_pooled"]
    ax.set_title(f"Overall score with 95% bootstrap interval, {n:,} items")
    return _save(fig, out_dir, "headline.png")


def fig_tasks(summary: dict, modality: str, out_dir: Path) -> Path:
    block = summary[modality]
    tasks = block["design"]["tasks"]
    order = block["ranking_pooled"]["ranking"][::-1]
    cols = 3 if len(tasks) > 4 else 2
    rows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, (0.3 * len(order) + 1.0) * rows), sharey=True, squeeze=False)
    for ax, task in zip(axes.ravel(), tasks):
        fc = block["floor_ceiling_pooled"][task]
        ax.axvline(fc["chance"], color=GREY, lw=1.2, ls=(0, (4, 3)))
        for y, m in enumerate(order):
            row = fc["models"][m]
            colour = BLUE if row["above_chance"] else GREY
            ax.plot([row["lo"], row["hi"]], [y, y], color=colour, lw=1.8, solid_capstyle="round")
            ax.plot(row["accuracy"], y, "o" if row["above_chance"] else "s", color=colour, ms=6, mec="white", mew=1)
        ax.set_xlim(-0.02, 1.04)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        _pct(ax)
        ax.grid(axis="y", visible=False)
        verdict = {"floor": "at floor", "ceiling": "at ceiling", "informative": "informative"}[fc["status"]]
        ax.set_title(f"{TASK_TITLES[task]}: {verdict}")
        ax.set_yticks(range(len(order)), _labels(summary, order))
        ax.tick_params(axis="y", labelcolor=INK, length=0)
    for ax in axes.ravel()[len(tasks) :]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], color=BLUE, marker="o", lw=1.8, ms=6, mec="white", label="above chance (Holm-adjusted p < 0.05)"),
        Line2D([], [], color=GREY, marker="s", lw=1.8, ms=6, mec="white", label="not above chance"),
        Line2D([], [], color=GREY, lw=1.2, ls=(0, (4, 3)), label="chance level"),
    ]
    fig.subplots_adjust(hspace=0.42, wspace=0.16)
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.02 if rows > 1 else -0.08))
    return _save(fig, out_dir, f"tasks_{modality}.png")


def fig_seeds_templates(summary: dict, out_dir: Path) -> Path:
    v = summary["vision"]
    order = v["ranking_pooled"]["ranking"][::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(order) + 1.8))
    for y, m in enumerate(order):
        seeds, templates = v["seeds"]["scores"][m], v["templates"]["scores"][m]
        ax.plot([min(seeds), max(seeds)], [y + 0.14] * 2, color=BLUE, lw=1.2, alpha=0.6)
        ax.plot(seeds, [y + 0.14] * len(seeds), "o", color=BLUE, ms=6, mec="white", mew=1)
        ax.plot([min(templates), max(templates)], [y - 0.14] * 2, color=RED, lw=1.2, alpha=0.6)
        ax.plot(templates, [y - 0.14] * len(templates), "D", color=RED, ms=5.5, mec="white", mew=1)
    ax.set_yticks(range(len(order)), _labels(summary, order))
    ax.tick_params(axis="y", labelcolor=INK, length=0)
    ax.grid(axis="y", visible=False)
    _pct(ax)
    ax.set_xlabel("overall score on one suite")
    n = v["design"]["n_items_per_suite"]
    ax.set_title(f"Regenerating the items against rewording the captions ({n} items per suite)")
    handles = [
        Line2D([], [], color=BLUE, marker="o", lw=1.2, ms=6, mec="white", label=f"{len(v['design']['suites'])} regenerated suites, canonical wording"),
        Line2D([], [], color=RED, marker="D", lw=1.2, ms=5.5, mec="white", label=f"{v['templates']['n_templates']} caption templates, same items (seed 0)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    return _save(fig, out_dir, "seeds_vs_templates.png")


def fig_items_needed(summary: dict, out_dir: Path) -> Path:
    block = summary["vision"]["items_needed"]
    used = summary["vision"]["design"]["n_items_per_task_per_suite"]
    series = [
        ("p_same_top1", "same top model", BLUE, "o", "-"),
        ("p_same_topk", "same top three", RED, "D", "-"),
        ("mean_tau", "Kendall tau (mean)", GREY, "s", (0, (4, 2))),
        ("p_same_order", "same complete order", GREEN, "^", (0, (1, 1.5))),
    ]
    panels = [
        ("resampled", "Two independent suites (pool resampled)"),
        ("independent", "Two disjoint parts of the pool (as planned; pessimistic)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, (key, title) in zip(axes, panels):
        need = block[key]
        sizes = sorted(int(k) for k in need)
        if used in sizes:
            ax.axvline(used, color=GREY, lw=1, ls=(0, (2, 2)))
            ax.text(used, 1.045, "size used", ha="center", va="bottom", fontsize=8, color=MUTED)
        for name, label, colour, marker, ls in series:
            ys = [need[str(n)][name] for n in sizes]
            ax.plot(sizes, ys, color=colour, marker=marker, ls=ls, lw=2, ms=6, mec="white", mew=1, label=label)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
        ax.minorticks_off()
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("items per task in each suite")
        ax.set_title(title, pad=18)
    axes[0].set_ylabel("agreement between the two suites")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.9, -0.17), ncol=4)
    fig.subplots_adjust(wspace=0.06)
    return _save(fig, out_dir, "items_needed.png")


def fig_pairwise(summary: dict, out_dir: Path) -> Path:
    rank = summary["vision"]["ranking_pooled"]
    label = {m: summary["models"][m]["label"] for m in rank["models"]}
    pairs = sorted(rank["pairs"], key=lambda p: abs(p["difference"]))
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(pairs) + 1.7))
    ax.axvline(0, color=GREY, lw=1)
    for y, p in enumerate(pairs):
        flip = p["difference"] < 0  # always show the better model first
        d, lo, hi = (-p["difference"], -p["hi"], -p["lo"]) if flip else (p["difference"], p["lo"], p["hi"])
        a, b = (p["b"], p["a"]) if flip else (p["a"], p["b"])
        colour = BLUE if p["separated"] else GREY
        ax.plot([lo, hi], [y, y], color=colour, lw=1.8, solid_capstyle="round")
        ax.plot(d, y, "o" if p["separated"] else "s", color=colour, ms=6, mec="white", mew=1)
        p["_label"] = f"{label[a]}  minus  {label[b]}"
    ax.set_yticks(range(len(pairs)), [p["_label"] for p in pairs])
    ax.tick_params(axis="y", labelcolor=INK, length=0, labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:+.0f}"))
    ax.set_xlabel("difference in overall score, percentage points (paired bootstrap, 95% interval)")
    ax.set_title(f"{rank['n_pairs_separated']} of {rank['n_pairs']} model pairs are separated")
    handles = [
        Line2D([], [], color=BLUE, marker="o", lw=1.8, ms=6, mec="white", label="interval excludes zero"),
        Line2D([], [], color=GREY, marker="s", lw=1.8, ms=6, mec="white", label="interval includes zero"),
    ]
    ax.legend(handles=handles, loc="lower right")
    return _save(fig, out_dir, "pairwise.png")


def fig_levels(summary: dict, out_dir: Path) -> Path:
    v = summary["vision"]
    tasks = v["design"]["tasks"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(13.2, 2.9), sharey=True)
    for ax, task in zip(axes, tasks):
        lv = v["levels"][task]
        chance = v["floor_ceiling_pooled"][task]["chance"]
        ax.axhline(chance, color=GREY, lw=1.2, ls=(0, (4, 3)))
        ax.fill_between([1, 2, 3], lv["min_per_level"], lv["max_per_level"], color=PALE_BLUE, alpha=0.55, lw=0)
        ax.plot([1, 2, 3], lv["mean"], color=BLUE, marker="o", lw=2, ms=6, mec="white", mew=1)
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.8, 3.2)
        ax.set_ylim(0, 1.03)
        ax.set_title(TASK_TITLES[task])
        ax.set_xlabel("difficulty level")
    _pct(axes[0], "y")
    axes[0].set_ylabel("accuracy")
    handles = [
        Line2D([], [], color=BLUE, marker="o", lw=2, ms=6, mec="white", label="mean over models"),
        matplotlib.patches.Patch(color=PALE_BLUE, alpha=0.55, label="weakest to strongest model"),
        Line2D([], [], color=GREY, lw=1.2, ls=(0, (4, 3)), label="chance level"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16))
    return _save(fig, out_dir, "levels.png")


def fig_items(summary: dict, out_dir: Path) -> Path:
    v = summary["vision"]
    tasks = v["design"]["tasks"]
    info = v["items"]["tasks"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.6))
    x = np.arange(len(tasks))
    disc = [info[t]["mean_discrimination"] or 0.0 for t in tasks]
    a.bar(x, disc, width=0.55, color=BLUE)
    a.axhline(0, color=GREY, lw=1)
    a.set_xticks(x, tasks)
    a.tick_params(axis="x", labelcolor=INK)
    a.grid(axis="x", visible=False)
    a.set_ylabel("mean item-ability correlation")
    a.set_title("Do items follow overall ability?")
    obs = [info[t]["n_anti_correlated"] for t in tasks]
    exp = [info[t]["expected_anti_correlated_by_chance"] for t in tasks]
    b.bar(x - 0.2, obs, width=0.36, color=RED, label="observed")
    b.bar(x + 0.2, exp, width=0.36, color=GREY, label="expected if items were unrelated to ability")
    b.set_xticks(x, tasks)
    b.tick_params(axis="x", labelcolor=INK)
    b.grid(axis="x", visible=False)
    b.set_ylabel("items")
    b.set_title(f"Items that run against ability (r at or below {v['items']['anti_correlation_threshold']})")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.subplots_adjust(wspace=0.25)
    return _save(fig, out_dir, "items.png")


def fig_redundancy(summary: dict, out_dir: Path) -> Path:
    red = summary["vision"]["redundancy"]
    tasks = red["tasks"]
    n = len(tasks)
    mat = np.full((n, n), np.nan)
    shown = np.full((n, n), np.nan)  # coloured only where the bootstrap interval excludes zero
    for p in red["pairs"]:
        i, j = tasks.index(p["a"]), tasks.index(p["b"])
        if p["spearman"] is not None:
            mat[i, j] = mat[j, i] = p["spearman"]
            if p["lo"] is not None and (p["lo"] > 0 or p["hi"] < 0):
                shown[i, j] = shown[j, i] = p["spearman"]
    cmap = LinearSegmentedColormap.from_list("div", [RED, "#F1F1F1", BLUE])
    cmap = cmap.with_extremes(bad="#F6F6F4")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(shown, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n), tasks)
    ax.set_yticks(range(n), tasks)
    ax.tick_params(labelcolor=INK, length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(mat[i, j]):
                solid = np.isfinite(shown[i, j])
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8.5,
                        color=("white" if abs(mat[i, j]) > 0.6 else INK) if solid else MUTED)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.set_label("Spearman correlation of model accuracies")
    ax.set_title("Do tasks order the models the same way?")
    ax.set_xlabel("Coloured cells: 95% bootstrap interval excludes zero.\nGrey cells: it does not.", fontsize=8.5, color=MUTED, labelpad=10)
    return _save(fig, out_dir, "redundancy.png")


def make_figures(results_root: Path, out_dir: Path) -> list[Path]:
    _style()
    with open(Path(results_root) / "summary.json") as f:
        summary = json.load(f)
    out_dir = Path(out_dir)
    made = [fig_gallery_vision(out_dir), fig_gallery_audio(out_dir)]
    if "vision" in summary:
        made += [
            fig_headline(summary, out_dir),
            fig_tasks(summary, "vision", out_dir),
            fig_seeds_templates(summary, out_dir),
            fig_items_needed(summary, out_dir),
            fig_pairwise(summary, out_dir),
            fig_levels(summary, out_dir),
            fig_items(summary, out_dir),
            fig_redundancy(summary, out_dir),
        ]
    if "audio" in summary:
        made.append(fig_tasks(summary, "audio", out_dir))
    return made
