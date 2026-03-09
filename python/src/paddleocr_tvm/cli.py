"""Command-line entry point for the PaddleOCR-TVM Python package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from paddleocr_tvm import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddleocr-tvm",
        description="PaddleOCR-TVM Python project scaffold.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the package version and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    print("PaddleOCR-TVM Python scaffold is ready.")
    return 0
