"""Parity helpers for Paddle and TVM comparisons."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paddleocr_tvm.artifacts import (
    ArtifactLayout,
    load_character_dict,
    resolve_artifacts_dir,
    unpack_model_tarball,
)
from paddleocr_tvm.backends import PaddleInferenceRunner
from paddleocr_tvm.constants import DEFAULT_DICT_PATH
from paddleocr_tvm.geometry import load_bgr_image
from paddleocr_tvm.pipeline import (
    MobileDetector,
    MobileOCRPipeline,
    MobileRecognizer,
    load_mobile_ocr,
)


def run_mobile_parity(
    images_dir: Path | str,
    artifacts_dir: Path | str | None,
    *,
    visualizations_dir: Path | str | None = None,
) -> dict[str, object]:
    """Compare Paddle and TVM OCR outputs over a directory of images."""

    layout = resolve_artifacts_dir(artifacts_dir)
    image_dir = Path(images_dir)
    visualization_root = Path(visualizations_dir) if visualizations_dir is not None else None
    if visualization_root is not None:
        visualization_root.mkdir(parents=True, exist_ok=True)
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
        if visualization_root is not None:
            visualization_path = visualization_root / f"{image_path.stem}.png"
            save_parity_visualization(
                image_path,
                paddle_lines=cast(list[dict[str, object]], paddle_lines),
                tvm_lines=cast(list[dict[str, object]], tvm_lines),
                output_path=visualization_path,
            )
            records[-1]["visualization"] = str(visualization_path)

    return {
        "images": len(image_paths),
        "exact_matches": exact_matches,
        "max_score_delta": max_score_delta,
        "records": records,
    }


def write_parity_summary(summary: Mapping[str, object], output_path: Path) -> None:
    """Write a parity summary to disk as UTF-8 JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def save_parity_visualization(
    image_path: Path,
    *,
    paddle_lines: list[dict[str, object]],
    tvm_lines: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save a minimal side-by-side OCR visualization for Paddle and TVM outputs."""

    original_bgr = load_bgr_image(image_path)
    paddle_panel = _annotate_ocr_image(original_bgr, paddle_lines, "Paddle")
    tvm_panel = _annotate_ocr_image(original_bgr, tvm_lines, "TVM")

    width = paddle_panel.width + tvm_panel.width
    height = max(paddle_panel.height, tvm_panel.height)
    combined = Image.new("RGB", (width, height), color=(255, 255, 255))
    combined.paste(paddle_panel, (0, 0))
    combined.paste(tvm_panel, (paddle_panel.width, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)


def _annotate_ocr_image(
    image_bgr: np.ndarray,
    lines: list[dict[str, object]],
    title: str,
) -> Image.Image:
    rgb = image_bgr[:, :, ::-1]
    image = Image.fromarray(rgb)
    font = ImageFont.load_default()
    header_height = 28
    panel = Image.new("RGB", (image.width, image.height + header_height), color=(255, 255, 255))
    panel.paste(image, (0, header_height))

    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, header_height), fill=(35, 35, 35))
    draw.text((8, 8), title, fill=(255, 255, 255), font=font)

    for index, line in enumerate(lines, start=1):
        points = _points_array(line["points"])
        shifted = points.copy()
        shifted[:, 1] += header_height
        polygon = [tuple(point.astype(int)) for point in shifted]
        draw.line([*polygon, polygon[0]], fill=(22, 163, 74), width=3)

        min_x = int(np.min(shifted[:, 0]))
        min_y = int(np.min(shifted[:, 1]))
        caption = f"{index}. {_line_caption(line)}"
        text_bbox = draw.textbbox((min_x, min_y), caption, font=font)
        background = (
            text_bbox[0] - 2,
            text_bbox[1] - 2,
            text_bbox[2] + 2,
            text_bbox[3] + 2,
        )
        draw.rectangle(background, fill=(255, 255, 255))
        draw.text((min_x, min_y), caption, fill=(12, 12, 12), font=font)

    return panel


def _line_caption(line: Mapping[str, object]) -> str:
    text = str(line.get("text", "")).replace("\n", " ").strip()
    if len(text) > 40:
        text = f"{text[:37]}..."
    score = float(cast(float | int | str, line.get("score", 0.0)))
    return f"{text} ({score:.2f})"


def _points_array(points: object) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    return array.reshape(4, 2)


def _load_paddle_mobile_ocr(layout: ArtifactLayout) -> MobileOCRPipeline:
    det_dir = unpack_model_tarball(layout, "mobile_det")
    rec_dir = unpack_model_tarball(layout, "mobile_rec")
    detector = MobileDetector(PaddleInferenceRunner(det_dir))
    recognizer = MobileRecognizer(
        PaddleInferenceRunner(rec_dir),
        dict_source=load_character_dict(rec_dir) or DEFAULT_DICT_PATH,
    )
    return MobileOCRPipeline(detector, recognizer)
