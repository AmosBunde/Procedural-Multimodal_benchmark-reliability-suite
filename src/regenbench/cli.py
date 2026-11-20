"""Command line interface: ``regenbench <command>``."""

from __future__ import annotations

import argparse
import sys
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
