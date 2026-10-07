"""Command line entry point: `outline-tracker` and its subcommands (SPEC 11).

Nothing heavy is imported here: torch, transformers and Qt are loaded only by the commands that
need them, so `--version`, `export` and `probe` start at once.
"""

from __future__ import annotations

import argparse

from outline_tracker import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="outline-tracker",
        description="Track objects in videos with EdgeTAM; export positions and outline shapes.",
    )
    parser.add_argument("--version", action="version", version=f"outline-tracker {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
