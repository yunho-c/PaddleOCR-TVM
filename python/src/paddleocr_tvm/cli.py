"""Command-line entry points for PaddleOCR-TVM."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from paddleocr_tvm import __version__
from paddleocr_tvm.benchmark import (
    DEFAULT_BENCHMARK_BACKENDS,
    benchmark_mobile,
    write_benchmark_csv,
    write_benchmark_summary,
)
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
    parity_parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the parity summary to a JSON file.",
    )
    parity_parser.add_argument(
        "--visualizations-dir",
        type=Path,
        help="Write minimal Paddle-vs-TVM OCR visualizations into this directory.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark-mobile",
        help="Benchmark mobile OCR backends on a directory of images.",
    )
    benchmark_parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing benchmark images.",
    )
    benchmark_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".artifacts"),
        help="Artifact cache directory.",
    )
    benchmark_parser.add_argument(
        "--backend",
        dest="backends",
        action="append",
        default=None,
        help=(
            "Benchmark backend preset. May be repeated. "
            f"Defaults to {', '.join(DEFAULT_BENCHMARK_BACKENDS)}."
        ),
    )
    benchmark_parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup passes over the dataset before timing.",
    )
    benchmark_parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of timed passes over the dataset.",
    )
    benchmark_parser.add_argument(
        "--output-json",
        type=Path,
        help="Write benchmark results to a JSON file.",
    )
    benchmark_parser.add_argument(
        "--output-csv",
        type=Path,
        help="Write aggregate benchmark rows to a CSV file.",
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
        parity_summary = run_mobile_parity(
            args.images,
            args.artifacts_dir,
            visualizations_dir=args.visualizations_dir,
        )
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(parity_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Wrote parity summary to {args.output_json}")
        else:
            print(json.dumps(parity_summary, indent=2, ensure_ascii=False))
        if args.visualizations_dir is not None:
            print(f"Wrote parity visualizations to {args.visualizations_dir}")
        return 0

    if args.command == "benchmark-mobile":
        benchmark_summary = benchmark_mobile(
            args.images,
            args.artifacts_dir,
            backends=args.backends,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        if args.output_json is not None:
            write_benchmark_summary(benchmark_summary, args.output_json)
            print(f"Wrote benchmark summary to {args.output_json}")
        else:
            print(json.dumps(benchmark_summary, indent=2, ensure_ascii=False))
        if args.output_csv is not None:
            write_benchmark_csv(benchmark_summary, args.output_csv)
            print(f"Wrote benchmark CSV to {args.output_csv}")
        return 0

    parser.print_help()
    return 0
