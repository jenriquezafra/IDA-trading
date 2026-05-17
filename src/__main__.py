from __future__ import annotations

import argparse

from src.alpha.research import main as alpha_research_main


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src")
    subparsers = parser.add_subparsers(dest="command", required=True)
    alpha = subparsers.add_parser("alpha-research", help="run declarative alpha research")
    alpha.add_argument("--config", default="configs/alpha/alpha_research_v1.yaml")
    alpha.add_argument("--output-dir", default=None)
    alpha.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.command == "alpha-research":
        forwarded = ["--config", args.config]
        if args.output_dir:
            forwarded.extend(["--output-dir", args.output_dir])
        if args.dry_run:
            forwarded.append("--dry-run")
        alpha_research_main(forwarded)


if __name__ == "__main__":
    main()
