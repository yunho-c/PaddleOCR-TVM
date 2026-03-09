"""High-level PP-OCRv5 mobile pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paddleocr_tvm.artifacts import ensure_directories, resolve_artifacts_dir
from paddleocr_tvm.backends import PaddleInferenceRunner, TvmRelaxRunner
from paddleocr_tvm.constants import DEFAULT_DICT_PATH, DET_IMAGE_SHAPE, REC_IMAGE_SHAPE
from paddleocr_tvm.conversion import convert_paddle_to_onnx
from paddleocr_tvm.errors import ArtifactPreparationError
from paddleocr_tvm.geometry import get_rotate_crop_image, load_bgr_image, sorted_boxes
from paddleocr_tvm.postprocess import CTCLabelDecoder, DBPostProcess
from paddleocr_tvm.preprocess import prepare_det_batch, prepare_rec_batch
from paddleocr_tvm.types import OCRBox, OCRResult, OCRTextLine


class MobileDetector:
    """Detector wrapper for the PP-OCRv5 mobile detector."""

    def __init__(self, runner: TvmRelaxRunner | PaddleInferenceRunner):
        self._runner = runner
        self._postprocess = DBPostProcess()

    def __call__(self, image: np.ndarray | str | Path) -> list[OCRBox]:
        bgr = load_bgr_image(image)
        batch, shape = prepare_det_batch(bgr)
        outputs = self._runner.run(batch)
        if not outputs:
            raise ArtifactPreparationError("Detector backend returned no outputs.")
        boxes_batch = self._postprocess(np.asarray(outputs[0], dtype=np.float32), shape)
        if not boxes_batch:
            return []
        return [OCRBox(points=box) for box in boxes_batch[0]]


class MobileRecognizer:
    """Recognizer wrapper for the PP-OCRv5 mobile recognizer."""

    def __init__(
        self,
        runner: TvmRelaxRunner | PaddleInferenceRunner,
        *,
        dict_path: Path = DEFAULT_DICT_PATH,
    ):
        self._runner = runner
        self._decoder = CTCLabelDecoder(dict_path)

    def __call__(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        if not images:
            return []
        batch, indices, _, _ = prepare_rec_batch(images, REC_IMAGE_SHAPE)
        outputs = self._runner.run(batch)
        if not outputs:
            raise ArtifactPreparationError("Recognizer backend returned no outputs.")
        decoded = self._decoder.decode(np.asarray(outputs[0], dtype=np.float32))
        result: list[tuple[str, float]] = [("", 0.0)] * len(images)
        for position, decoded_item in zip(indices, decoded, strict=False):
            result[position] = decoded_item
        return result


class MobileOCRPipeline:
    """End-to-end mobile OCR pipeline."""

    def __init__(self, detector: MobileDetector, recognizer: MobileRecognizer):
        self._detector = detector
        self._recognizer = recognizer

    def __call__(self, image: np.ndarray | str | Path) -> OCRResult:
        bgr = load_bgr_image(image)
        boxes = self._detector(bgr)
        if not boxes:
            return OCRResult(lines=[])
        sorted_points = sorted_boxes(np.asarray([box.points for box in boxes], dtype=np.float32))
        crops = [get_rotate_crop_image(bgr, points) for points in sorted_points]
        rec_results = self._recognizer(crops)
        lines = [
            OCRTextLine(points=points, text=text, score=score)
            for points, (text, score) in zip(sorted_points, rec_results, strict=False)
        ]
        return OCRResult(lines=lines)


def prepare_mobile_models(artifacts_dir: Path | str | None, *, target: str = "llvm") -> None:
    """Prepare Paddle, ONNX, and TVM artifacts for the mobile detector and recognizer."""

    layout = resolve_artifacts_dir(artifacts_dir)
    ensure_directories(layout)
    det_onnx = convert_paddle_to_onnx(layout, "mobile_det")
    rec_onnx = convert_paddle_to_onnx(layout, "mobile_rec")
    TvmRelaxRunner(
        layout,
        "mobile_det",
        det_onnx,
        target=target,
        shape_dict={"x": [1, *DET_IMAGE_SHAPE]},
    )
    TvmRelaxRunner(
        layout,
        "mobile_rec",
        rec_onnx,
        target=target,
        shape_dict={"x": [1, *REC_IMAGE_SHAPE]},
    )


def load_mobile_detector(artifacts_dir: Path | str | None) -> MobileDetector:
    """Load the TVM-backed mobile detector."""

    layout = resolve_artifacts_dir(artifacts_dir)
    runner = TvmRelaxRunner(
        layout,
        "mobile_det",
        convert_paddle_to_onnx(layout, "mobile_det"),
        shape_dict={"x": [1, *DET_IMAGE_SHAPE]},
    )
    return MobileDetector(runner)


def load_mobile_recognizer(artifacts_dir: Path | str | None) -> MobileRecognizer:
    """Load the TVM-backed mobile recognizer."""

    layout = resolve_artifacts_dir(artifacts_dir)
    runner = TvmRelaxRunner(
        layout,
        "mobile_rec",
        convert_paddle_to_onnx(layout, "mobile_rec"),
        shape_dict={"x": [1, *REC_IMAGE_SHAPE]},
    )
    return MobileRecognizer(runner)


def load_mobile_ocr(artifacts_dir: Path | str | None) -> MobileOCRPipeline:
    """Load the TVM-backed mobile OCR pipeline."""

    return MobileOCRPipeline(
        detector=load_mobile_detector(artifacts_dir),
        recognizer=load_mobile_recognizer(artifacts_dir),
    )
