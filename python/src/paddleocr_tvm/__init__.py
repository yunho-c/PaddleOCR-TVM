"""Top-level package for PaddleOCR-TVM."""

from paddleocr_tvm.pipeline import (
    MobileDetector,
    MobileOCRPipeline,
    MobileRecognizer,
    load_mobile_detector,
    load_mobile_ocr,
    load_mobile_recognizer,
    prepare_mobile_models,
)
from paddleocr_tvm.types import OCRBox, OCRResult, OCRTextLine

__all__ = [
    "MobileDetector",
    "MobileOCRPipeline",
    "MobileRecognizer",
    "OCRBox",
    "OCRResult",
    "OCRTextLine",
    "__version__",
    "load_mobile_detector",
    "load_mobile_ocr",
    "load_mobile_recognizer",
    "prepare_mobile_models",
]

__version__ = "0.1.0"
