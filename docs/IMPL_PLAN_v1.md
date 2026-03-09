# PP-OCRv5 Mobile End-to-End Relax Port Plan v1

## Summary

Implement a CPU-only v1 of the PP-OCRv5 mobile OCR pipeline in the `python/` package, using TVM Relax for the detector and recognizer graphs while keeping Paddle-compatible preprocessing, DB postprocess, crop extraction, ordering, and CTC decode in Python.

The first implementation milestone includes:

- `PP-OCRv5_mobile_det`
- `PP-OCRv5_mobile_rec`
- end-to-end OCR pipeline on CPU
- `quad` box mode only
- no angle classifier
- no server models
- ONNX as the import boundary into Relax

## Implementation Changes

### 1. Planning artifact

Treat this file as the implementation-facing execution spec for v1.

### 2. Artifact pipeline

Add an artifact workflow under `python/` with stable locations for:

- Paddle inference model tarballs and unpacked contents
- converted ONNX models
- compiled Relax CPU artifacts

Implement idempotent utilities that:

- download official `PP-OCRv5_mobile_det` and `PP-OCRv5_mobile_rec` inference artifacts
- unpack them into a local artifact cache
- convert Paddle inference models to ONNX using PaddleX `paddle2onnx`
- import ONNX into Relax
- compile for `llvm`
- reuse cached outputs on subsequent runs

Use official Paddle inference models as the only upstream model source in v1.

### 3. Runtime interfaces

Implement these public Python entry points:

- `prepare_mobile_models(artifacts_dir: Path, target: str = "llvm") -> None`
- `load_mobile_detector(artifacts_dir: Path) -> MobileDetector`
- `load_mobile_recognizer(artifacts_dir: Path) -> MobileRecognizer`
- `load_mobile_ocr(artifacts_dir: Path) -> MobileOCRPipeline`

Implement result dataclasses:

- `OCRBox`
- `OCRTextLine`
- `OCRResult`

`MobileOCRPipeline.__call__` should accept `np.ndarray | PIL.Image.Image | str | Path` and return `OCRResult`.

### 4. Host-side parity logic

Implement Python logic that mirrors PaddleOCR mobile inference behavior exactly enough for parity:

- detector preprocessing:
  - BGR image handling
  - limit-side resize
  - output sizes rounded to multiples of 32
  - Paddle detector normalization
  - CHW conversion
  - preserve `[src_h, src_w, ratio_h, ratio_w]`
- detector postprocessing:
  - DB thresholding
  - contour extraction
  - `fast` score mode
  - `unclip_ratio=1.5`
  - `quad` output only
- system glue:
  - Paddle-style box sorting
  - perspective crop extraction matching PaddleOCR
  - no angle classifier in v1
- recognizer preprocessing:
  - height `48`
  - dynamic width batching by aspect ratio
  - `[-1, 1]` normalization
  - right-padding to batch max width
- recognizer postprocessing:
  - greedy CTC decode
  - blank removal
  - duplicate collapse
  - `ppocrv5_dict.txt`

Do not move DB postprocess or CTC decode into TVM in v1.

### 5. CLI and project wiring

Replace the placeholder CLI with:

- `paddleocr-tvm prepare-mobile --artifacts-dir .artifacts --target llvm`
- `paddleocr-tvm ocr-mobile --image <path> --artifacts-dir .artifacts`
- `paddleocr-tvm parity-mobile --images <dir> --artifacts-dir .artifacts`

Update `pyproject.toml` and Pixi tasks to include:

- runtime dependencies for OCR + Relax + ONNX import
- a parity environment or dependency group for `paddlepaddle` and `paddlex`
- tasks for `prepare-mobile`, `ocr-mobile`, `parity-mobile`, `test`, and `check`

### 6. Exact implementation order

Implement in this order:

1. Write `docs/IMPL_PLAN_v1.md`
2. Add artifact manager and download/unpack code
3. Add Paddle-to-ONNX conversion wrapper
4. Implement Relax import/build/load for the recognizer
5. Implement recognizer preprocessing + CTC decode + parity harness
6. Implement Relax import/build/load for the detector
7. Implement detector preprocessing + DB postprocess parity harness
8. Implement end-to-end mobile OCR pipeline
9. Replace CLI scaffold and wire Pixi tasks
10. Refresh docs with concrete usage after parity is passing

Do not begin detector work until recognizer parity is passing.

## Test Plan

### Unit tests

Add deterministic tests for:

- detector resize and ratio metadata
- recognizer resize/padding with mixed aspect ratios
- CTC decode blank removal and duplicate collapse
- box sort order
- perspective crop extraction on synthetic quads
- artifact path resolution and idempotent prepare behavior

### Parity tests

Create a small committed fixture set for:

- recognition crop images
- full images for detector and end-to-end parity

Recognizer parity acceptance:

- decoded text matches Paddle exactly on all crop fixtures
- logits are numerically close with `rtol=1e-3`, `atol=1e-3`

Detector parity acceptance:

- detector maps are numerically close before DB postprocess with `rtol=1e-3`, `atol=1e-3`
- postprocessed box count matches Paddle on all fixtures
- matched quad coordinates are within 2 pixels mean absolute point error after resize reversal

End-to-end acceptance:

- final text outputs match PaddleOCR exactly on the fixture set
- line ordering matches PaddleOCR output order
- confidence delta per line is <= `0.02`

## Assumptions and defaults

- v1 is CPU-only and targets `llvm`
- v1 scope is mobile end-to-end only
- ONNX is the only import boundary into Relax in v1
- official Paddle inference models are used as upstream inputs
- host-side Python keeps Paddle-compatible preprocessing and postprocessing
- only `quad` boxes are supported in v1
- angle classification is out of scope for v1
- if TVM is not already importable in the environment, the implementation should fail fast with a clear setup error rather than bootstrap TVM automatically
