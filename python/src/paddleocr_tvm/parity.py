"""Parity helpers for Paddle and TVM comparisons."""

from __future__ import annotations

from pathlib import Path

from paddleocr_tvm.artifacts import ArtifactLayout, resolve_artifacts_dir, unpack_model_tarball
from paddleocr_tvm.backends import PaddleInferenceRunner
from paddleocr_tvm.pipeline import (
    MobileDetector,
    MobileOCRPipeline,
    MobileRecognizer,
    load_mobile_ocr,
)


def run_mobile_parity(
    images_dir: Path | str,
    artifacts_dir: Path | str | None,
) -> dict[str, object]:
    """Compare Paddle and TVM OCR outputs over a directory of images."""

    layout = resolve_artifacts_dir(artifacts_dir)
    image_dir = Path(images_dir)
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    paddle_pipeline = _load_paddle_mobile_ocr(layout)
    tvm_pipeline = load_mobile_ocr(layout.root)

    records: list[dict[str, object]] = []
    exact_matches = 0
    max_score_delta = 0.0

    for image_path in image_paths:
        paddle_result = paddle_pipeline(image_path)
        tvm_result = tvm_pipeline(image_path)
        paddle_lines = paddle_result.to_dict()["lines"]
        tvm_lines = tvm_result.to_dict()["lines"]
        is_match = paddle_lines == tvm_lines
        if is_match:
            exact_matches += 1
        if paddle_lines and tvm_lines:
            deltas = [
                abs(float(paddle["score"]) - float(tvm["score"]))
                for paddle, tvm in zip(paddle_lines, tvm_lines, strict=False)
            ]
            if deltas:
                max_score_delta = max(max_score_delta, max(deltas))
        records.append(
            {
                "image": image_path.name,
                "exact_match": is_match,
                "paddle": paddle_lines,
                "tvm": tvm_lines,
            }
        )

    return {
        "images": len(image_paths),
        "exact_matches": exact_matches,
        "max_score_delta": max_score_delta,
        "records": records,
    }


def _load_paddle_mobile_ocr(layout: ArtifactLayout) -> MobileOCRPipeline:
    det_dir = unpack_model_tarball(layout, "mobile_det")
    rec_dir = unpack_model_tarball(layout, "mobile_rec")
    detector = MobileDetector(PaddleInferenceRunner(det_dir))
    recognizer = MobileRecognizer(PaddleInferenceRunner(rec_dir))
    return MobileOCRPipeline(detector, recognizer)
