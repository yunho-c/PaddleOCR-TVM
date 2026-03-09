"""Command-line entry points for PaddleOCR-TVM."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from paddleocr_tvm import __version__
from paddleocr_tvm.parity import run_mobile_parity
from paddleocr_tvm.pipeline import load_mobile_ocr, prepare_mobile_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddleocr-tvm",
        description="PP-OCRv5 mobile OCR tooling backed by TVM Relax.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser(
        "prepare-mobile",
        help="Download model artifacts, convert them to ONNX, and compile TVM artifacts.",
    )
    prepare_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".artifacts"),
        help="Artifact cache directory.",
    )
    prepare_parser.add_argument(
        "--target",
        default="llvm",
        help="TVM target to compile for.",
    )

    ocr_parser = subparsers.add_parser(
        "ocr-mobile",
        help="Run the mobile OCR pipeline on a single image.",
    )
    ocr_parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    ocr_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".artifacts"),
        help="Artifact cache directory.",
    )
    ocr_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )

    parity_parser = subparsers.add_parser(
        "parity-mobile",
        help="Compare Paddle and TVM OCR outputs on a directory of images.",
    )
    parity_parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing parity images.",
    )
    parity_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".artifacts"),
        help="Artifact cache directory.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.command == "prepare-mobile":
        prepare_mobile_models(args.artifacts_dir, target=args.target)
        print(f"Prepared PP-OCRv5 mobile artifacts in {args.artifacts_dir}")
        return 0

    if args.command == "ocr-mobile":
        pipeline = load_mobile_ocr(args.artifacts_dir)
        result = pipeline(args.image)
        payload = result.to_dict()
        if args.pretty:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.command == "parity-mobile":
        summary = run_mobile_parity(args.images, args.artifacts_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    parser.print_help()
    return 0
