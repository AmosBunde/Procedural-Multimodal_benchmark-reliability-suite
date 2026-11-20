"""Command line interface: ``regenbench <command>``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .names import validate_name


def _name(what: str):
    def parse(value: str) -> str:
        try:
            return validate_name(value, what)
        except ValueError as err:
            raise argparse.ArgumentTypeError(str(err)) from None

    return parse


def cmd_generate(args: argparse.Namespace) -> int:
    from . import study
    from .generate import write_generated_suite

    out = Path(args.out)
    if args.study:
        jobs = study.suites()
    else:
        jobs = [{"modality": args.modality, "seed": args.seed, "n_per_task": args.n_per_task, "n_templates": args.n_templates}]
    for job in jobs:
        meta = write_generated_suite(out, write_media=args.write_media, suite_id=None if args.study else args.suite_id, **job)
        print(f"{out / meta['suite_id']}: {meta['n_items']} items, {meta['n_templates']} template(s), sha256 {meta['sha256'][:16]}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from . import study
    from .evaluate import evaluate_suite, result_path
    from .items import META_NAME
    from .models import get_model
    import json

    data, results = Path(args.data), Path(args.results)
    if args.study:
        suite_dirs = [data / f"{job['modality']}-seed{job['seed']}" for job in study.suites()]
    else:
        suite_dirs = [Path(s) for s in args.suite]
    if not suite_dirs:
        print("nothing to do: give --suite or --study", file=sys.stderr)
        return 2
    plan: dict[str, list[tuple[Path, str]]] = {}
    for directory in suite_dirs:
        with open(directory / META_NAME) as f:
            meta = json.load(f)
        suite_id = validate_name(meta["suite_id"], "suite id")
        keys = args.models or list(study.models_for(meta["modality"]))
        for key in keys:
            spec = get_model(key)
            if spec.modality != {"vision": "image", "audio": "audio"}[meta["modality"]]:
                continue
            if result_path(results, args.run_id, suite_id, key).exists() and not args.force:
                continue
            plan.setdefault(key, []).append((directory, suite_id))
    started = time.perf_counter()
    for key, jobs in plan.items():  # one model in memory at a time
        adapter = get_model(key).load(threads=args.threads)
        for directory, suite_id in jobs:
            t0 = time.perf_counter()
            out = evaluate_suite(
                adapter,
                directory,
                results,
                args.run_id,
                cache_root=Path(args.cache),
                chunk_size=args.chunk_size,
                progress=(lambda msg: print(msg, flush=True)) if args.verbose else None,
            )
            print(f"{out}  ({time.perf_counter() - t0:.0f} s)", flush=True)
        adapter.close()
        del adapter
    print(f"done: {sum(len(j) for j in plan.values())} result file(s) written in {time.perf_counter() - started:.0f} s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regenbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a suite of items from a seed")
    g.add_argument("--modality", choices=["vision", "audio"], default="vision")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--n-per-task", type=int, default=120, help="items per task, a multiple of 3")
    g.add_argument("--n-templates", type=int, default=3, choices=[1, 2, 3], help="caption templates to keep")
    g.add_argument("--suite-id", type=_name("suite id"), default=None, help="default: <modality>-seed<seed>")
    g.add_argument("--out", default="data", help="directory that holds one folder per suite")
    g.add_argument("--write-media", action="store_true", help="also write PNG or WAV files for inspection")
    g.add_argument("--study", action="store_true", help="generate every suite used in the study")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="run models on suites and store item-level scores")
    e.add_argument("--suite", action="append", default=[], help="suite directory; may be repeated")
    e.add_argument("--study", action="store_true", help="every suite and every model of the study")
    e.add_argument("--models", nargs="*", type=_name("model key"), help="model keys (default: the study's models)")
    e.add_argument("--run-id", type=_name("run id"), default="main")
    e.add_argument("--data", default="data")
    e.add_argument("--results", default="results")
    e.add_argument("--cache", default=".cache", help="item-level cache used to resume")
    e.add_argument("--threads", type=int, default=6)
    e.add_argument("--chunk-size", type=int, default=32)
    e.add_argument("--force", action="store_true", help="re-evaluate even if a result file exists")
    e.add_argument("--verbose", action="store_true")
    e.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
