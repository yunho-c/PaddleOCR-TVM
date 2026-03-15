"""Benchmark helpers for comparing PP-OCRv5 mobile backends."""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

from paddleocr_tvm.backend_specs import PADDLE_MKLDNN_BACKEND, parse_backend_spec
from paddleocr_tvm.errors import ArtifactPreparationError
from paddleocr_tvm.geometry import get_rotate_crop_image, load_bgr_image, sorted_boxes
from paddleocr_tvm.pipeline import load_mobile_detector, load_mobile_ocr, load_mobile_recognizer

DEFAULT_BENCHMARK_BACKENDS = ("paddle", "onnxruntime", "tvm-llvm")


class BenchmarkStats(TypedDict):
    """Aggregate timing statistics for one benchmark scope."""

    calls: int
    logical_items: int
    total_ms: float
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    throughput_calls_per_s: float
    throughput_items_per_s: float


class BackendBenchmarkResult(TypedDict):
    """Per-backend benchmark summary."""

    backend: str
    kind: str
    target: str | None
    device: str | None
    detector: BenchmarkStats
    recognizer: BenchmarkStats
    end_to_end: BenchmarkStats


class BenchmarkDatasetSummary(TypedDict):
    """Dataset metadata emitted alongside benchmark numbers."""

    images_dir: str
    images: int
    recognition_batches: int
    recognition_crops: int


class BenchmarkSummary(TypedDict):
    """Full benchmark output payload."""

    dataset: BenchmarkDatasetSummary
    warmup: int
    repeat: int
    backends: list[BackendBenchmarkResult]


@dataclass(frozen=True)
class BenchmarkCase:
    """A timed benchmark call."""

    label: str
    call: Callable[[], object]
    logical_items: int


def benchmark_mobile(
    images_dir: Path | str,
    artifacts_dir: Path | str | None,
    *,
    backends: Sequence[str] | None = None,
    warmup: int = 1,
    repeat: int = 5,
) -> BenchmarkSummary:
    """Benchmark detector, recognizer, and end-to-end OCR across backends."""

    if warmup < 0:
        raise ArtifactPreparationError("warmup must be >= 0")
    if repeat <= 0:
        raise ArtifactPreparationError("repeat must be > 0")

    image_dir = Path(images_dir)
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise ArtifactPreparationError(f"No benchmark images found under {image_dir}.")

    loaded_images = [(path.name, load_bgr_image(path)) for path in image_paths]
    crop_groups = _extract_reference_crops(artifacts_dir, loaded_images)
    backend_specs = [parse_backend_spec(name) for name in (backends or DEFAULT_BENCHMARK_BACKENDS)]

    backend_results: list[BackendBenchmarkResult] = []
    for backend in backend_specs:
        detector = load_mobile_detector(artifacts_dir, backend=backend)
        recognizer = load_mobile_recognizer(artifacts_dir, backend=backend)
        pipeline = load_mobile_ocr(artifacts_dir, det_backend=backend, rec_backend=backend)

        detector_cases = [
            _make_detector_case(name, image, detector)
            for name, image in loaded_images
        ]
        recognizer_cases = [
            _make_recognizer_case(name, crops, recognizer)
            for (name, _), crops in zip(loaded_images, crop_groups, strict=False)
            if crops
        ]
        end_to_end_cases = [
            _make_pipeline_case(name, image, pipeline)
            for name, image in loaded_images
        ]

        backend_results.append(
            {
                "backend": backend.name,
                "kind": backend.kind,
                "target": backend.target,
                "device": backend.device,
                "detector": _benchmark_cases(detector_cases, warmup=warmup, repeat=repeat),
                "recognizer": _benchmark_cases(recognizer_cases, warmup=warmup, repeat=repeat)
                if recognizer_cases
                else _empty_stats(),
                "end_to_end": _benchmark_cases(end_to_end_cases, warmup=warmup, repeat=repeat),
            }
        )

    return {
        "dataset": {
            "images_dir": str(image_dir),
            "images": len(loaded_images),
            "recognition_batches": sum(1 for crops in crop_groups if crops),
            "recognition_crops": sum(len(crops) for crops in crop_groups),
        },
        "warmup": warmup,
        "repeat": repeat,
        "backends": backend_results,
    }


def write_benchmark_summary(summary: BenchmarkSummary, output_path: Path) -> None:
    """Write benchmark results to a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_benchmark_csv(summary: BenchmarkSummary, output_path: Path) -> None:
    """Write aggregate benchmark rows as CSV."""

    rows: list[dict[str, object]] = []
    for backend_row in summary["backends"]:
        scope_stats = (
            ("detector", backend_row["detector"]),
            ("recognizer", backend_row["recognizer"]),
            ("end_to_end", backend_row["end_to_end"]),
        )
        for scope, stats in scope_stats:
            rows.append(
                {
                    "backend": backend_row["backend"],
                    "kind": backend_row["kind"],
                    "target": backend_row["target"],
                    "device": backend_row["device"],
                    "scope": scope,
                    "calls": stats["calls"],
                    "logical_items": stats["logical_items"],
                    "total_ms": stats["total_ms"],
                    "mean_ms": stats["mean_ms"],
                    "median_ms": stats["median_ms"],
                    "p90_ms": stats["p90_ms"],
                    "p95_ms": stats["p95_ms"],
                    "min_ms": stats["min_ms"],
                    "max_ms": stats["max_ms"],
                    "throughput_calls_per_s": stats["throughput_calls_per_s"],
                    "throughput_items_per_s": stats["throughput_items_per_s"],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["backend"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_reference_crops(
    artifacts_dir: Path | str | None,
    loaded_images: Sequence[tuple[str, np.ndarray]],
) -> list[list[np.ndarray]]:
    reference_detector = load_mobile_detector(artifacts_dir, backend=PADDLE_MKLDNN_BACKEND)
    crop_groups: list[list[np.ndarray]] = []
    for _, image in loaded_images:
        boxes = reference_detector(image)
        if not boxes:
            crop_groups.append([])
            continue
        sorted_points = sorted_boxes(np.asarray([box.points for box in boxes], dtype=np.float32))
        crop_groups.append([get_rotate_crop_image(image, points) for points in sorted_points])
    return crop_groups


def _make_detector_case(
    label: str,
    image: np.ndarray,
    detector: Callable[[np.ndarray], object],
) -> BenchmarkCase:
    def invoke() -> object:
        return detector(image)

    return BenchmarkCase(label=label, call=invoke, logical_items=1)


def _make_recognizer_case(
    label: str,
    crops: list[np.ndarray],
    recognizer: Callable[[list[np.ndarray]], object],
) -> BenchmarkCase:
    def invoke() -> object:
        return recognizer(crops)

    return BenchmarkCase(label=label, call=invoke, logical_items=max(len(crops), 1))


def _make_pipeline_case(
    label: str,
    image: np.ndarray,
    pipeline: Callable[[np.ndarray], object],
) -> BenchmarkCase:
    def invoke() -> object:
        return pipeline(image)

    return BenchmarkCase(label=label, call=invoke, logical_items=1)


def _benchmark_cases(
    cases: Sequence[BenchmarkCase],
    *,
    warmup: int,
    repeat: int,
) -> BenchmarkStats:
    if not cases:
        return _empty_stats()

    for _ in range(warmup):
        for case in cases:
            case.call()

    samples_ms: list[float] = []
    total_ns = 0
    logical_items = 0
    for _ in range(repeat):
        for case in cases:
            start_ns = time.perf_counter_ns()
            case.call()
            elapsed_ns = time.perf_counter_ns() - start_ns
            total_ns += elapsed_ns
            logical_items += case.logical_items
            samples_ms.append(elapsed_ns / 1_000_000.0)

    total_ms = total_ns / 1_000_000.0
    total_seconds = total_ns / 1_000_000_000.0
    return {
        "calls": len(samples_ms),
        "logical_items": logical_items,
        "total_ms": round(total_ms, 6),
        "mean_ms": round(statistics.fmean(samples_ms), 6),
        "median_ms": round(statistics.median(samples_ms), 6),
        "p90_ms": round(_percentile(samples_ms, 90), 6),
        "p95_ms": round(_percentile(samples_ms, 95), 6),
        "min_ms": round(min(samples_ms), 6),
        "max_ms": round(max(samples_ms), 6),
        "throughput_calls_per_s": round(len(samples_ms) / total_seconds, 6),
        "throughput_items_per_s": round(logical_items / total_seconds, 6),
    }


def _empty_stats() -> BenchmarkStats:
    return {
        "calls": 0,
        "logical_items": 0,
        "total_ms": 0.0,
        "mean_ms": 0.0,
        "median_ms": 0.0,
        "p90_ms": 0.0,
        "p95_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
        "throughput_calls_per_s": 0.0,
        "throughput_items_per_s": 0.0,
    }


def _percentile(values: Sequence[float], percentile: int) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
