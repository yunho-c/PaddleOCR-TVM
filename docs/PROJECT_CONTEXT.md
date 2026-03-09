# Project Context

## Goal

This repository is intended to port the latest PP-OCRv5 models from PaddleOCR into Apache TVM using Relax.

At the moment, the repository itself is very small. The practical source of truth is the PaddleOCR submodule in `reference/PaddleOCR`, currently at commit `acfd89b1`.

There is no existing TVM or Relax implementation in this repository yet. The current work is therefore a greenfield port that should be grounded in the PaddleOCR inference codepaths rather than only the training configs.

## Relevant Upstream Scope

The PP-OCRv5 models exposed in the checked-in PaddleOCR tree are:

- `PP-OCRv5_server_det`
- `PP-OCRv5_mobile_det`
- `PP-OCRv5_server_rec`
- `PP-OCRv5_mobile_rec`

Useful entry points:

- `reference/PaddleOCR/docs/version3.x/model_list.md`
- `reference/PaddleOCR/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md`
- `reference/PaddleOCR/configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml`
- `reference/PaddleOCR/configs/det/PP-OCRv5/PP-OCRv5_server_det.yml`
- `reference/PaddleOCR/configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml`
- `reference/PaddleOCR/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml`

## Model Breakdown

### Detection

PP-OCRv5 uses DB-style text detection for both mobile and server variants.

Mobile detector:

- Backbone: `PPLCNetV3`
- Neck: `RSEFPN`
- Head: `DBHead`
- Config: `reference/PaddleOCR/configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml`

Server detector:

- Backbone: `PPHGNetV2_B4`
- Neck: `LKPAN`
- Head: `PFHeadLocal`
- Config: `reference/PaddleOCR/configs/det/PP-OCRv5/PP-OCRv5_server_det.yml`

Implementation files:

- `reference/PaddleOCR/ppocr/modeling/necks/db_fpn.py`
- `reference/PaddleOCR/ppocr/modeling/heads/det_db_head.py`
- `reference/PaddleOCR/ppocr/postprocess/db_postprocess.py`

Important inference detail:

- At inference time, the detector returns a probability map, not final boxes.
- Box extraction is done in Python host code by `DBPostProcess`.
- This means a practical Relax port can compile the network body first and keep DB postprocess in Python initially.

### Recognition

PP-OCRv5 recognition uses two different named algorithms depending on size tier:

Mobile recognizer:

- Algorithm: `SVTR_LCNet`
- Backbone: `PPLCNetV3`
- Head: `MultiHead`
- Config: `reference/PaddleOCR/configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml`

Server recognizer:

- Algorithm: `SVTR_HGNet`
- Backbone: `PPHGNetV2_B4`
- Head: `MultiHead`
- Config: `reference/PaddleOCR/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml`

Implementation files:

- `reference/PaddleOCR/ppocr/modeling/heads/rec_multi_head.py`
- `reference/PaddleOCR/tools/infer/predict_rec.py`
- `reference/PaddleOCR/ppocr/postprocess/rec_postprocess.py`

Important inference simplification:

- `MultiHead` contains both a CTC path and an NRTR path during training.
- In eval mode, `MultiHead.forward()` returns only the CTC output.
- This is a major simplification for the Relax port: inference does not need the full training-time dual-head behavior.

## End-to-End Inference Pipeline

The end-to-end OCR pipeline is implemented in:

- `reference/PaddleOCR/tools/infer/predict_system.py`

The flow is:

1. Run text detection.
2. Apply DB postprocess to get polygons or quads.
3. Sort boxes top-to-bottom, left-to-right.
4. Crop text regions with perspective transform.
5. Optionally run angle classification.
6. Batch text crops for recognition.
7. Decode recognizer logits with CTC.

This matters because the port target is not just "four neural nets". The practical system also depends on:

- detector preprocessing
- detector postprocessing
- crop extraction and ordering
- recognizer preprocessing
- CTC decoding

If the goal is PP-OCRv5 parity, these runtime details must be preserved.

## Preprocessing and Postprocessing Details That Matter

### Detection preprocessing

Detector resize logic lives in:

- `reference/PaddleOCR/ppocr/data/imaug/operators.py`

Important behavior:

- Images are resized according to limit-side rules.
- Final detector input sides are rounded to multiples of 32.
- Shape metadata is preserved as `[src_h, src_w, ratio_h, ratio_w]`.

This metadata is consumed later by `DBPostProcess` to project predicted boxes back to source-image coordinates.

### Detection postprocessing

`DBPostProcess` lives in:

- `reference/PaddleOCR/ppocr/postprocess/db_postprocess.py`

Important behavior:

- Threshold the predicted map.
- Extract contours.
- Score contours using either fast or slow scoring.
- Expand polygons via `pyclipper` (`unclip_ratio`).
- Produce either `quad` or `poly` boxes.

This is currently Python/OpenCV/Shapely/Pyclipper-based logic. It is not part of the neural network graph.

### Recognition preprocessing

Recognizer runtime preprocessing is in:

- `reference/PaddleOCR/tools/infer/predict_rec.py`

Important behavior for PP-OCRv5:

- Crops are grouped by aspect ratio to reduce padding waste.
- Recognition inputs use height `48`.
- Width is dynamic at inference time.
- Images are normalized to `[-1, 1]` after division by `255`.
- Width padding is based on the largest aspect ratio in the batch.

For parity, the Relax path should reproduce this batching and padding behavior.

### Recognition postprocessing

CTC decoding is in:

- `reference/PaddleOCR/ppocr/postprocess/rec_postprocess.py`

Important behavior:

- Greedy `argmax` over logits.
- Remove duplicate repeated tokens.
- Ignore the CTC blank token.
- Decode with `ppocrv5_dict.txt`.

## Export Path

Relevant export files:

- `reference/PaddleOCR/tools/export_model.py`
- `reference/PaddleOCR/ppocr/utils/export_model.py`
- `reference/PaddleOCR/docs/version3.x/deployment/obtaining_onnx_models.md`

Important details:

- PaddleOCR supports exporting static inference models.
- For `SVTR_LCNet` and `SVTR_HGNet`, Paddle export uses dynamic width input shape `[None, 3, 48, -1]`.
- PaddleOCR documents a conversion path from Paddle static models to ONNX using `paddlex --install paddle2onnx` and `paddlex --paddle2onnx ...`.

Practical implication:

- The cleanest initial Relax path is likely:
  1. export Paddle inference model
  2. convert to ONNX
  3. import ONNX into TVM Relax
  4. keep preprocessing/postprocessing in Python first

This is lower risk than trying to reproduce Paddle model construction directly in Relax from scratch on day one.

## Recommended First Milestone

The best starting point is `PP-OCRv5_mobile_rec`.

Reasons:

- It is much smaller than the server recognizer.
- Inference behavior is simpler than the training config suggests because only the CTC branch is used.
- Recognition parity is easier to validate than full detection plus cropping plus system orchestration.
- It lets the repository establish a basic TVM/Relax model-loading and parity-testing workflow before tackling detector postprocess.

Suggested sequence:

1. Export `PP-OCRv5_mobile_rec` to Paddle inference format.
2. Convert it to ONNX.
3. Import into Relax.
4. Build a parity harness that compares Paddle/ONNX/Relax outputs on text crops.
5. Add CTC decode using `ppocrv5_dict.txt`.
6. After recognizer parity is stable, port `PP-OCRv5_mobile_det`.
7. Keep DB postprocess and crop extraction in Python initially.
8. Only then attempt the full end-to-end OCR pipeline.

## Architecture Notes for Future Repository Work

When implementing the TVM side in this repository, it will likely be useful to separate:

- model export/conversion utilities
- Relax model import/build/runtime
- Paddle parity harnesses
- OCR runtime preprocessing/postprocessing
- end-to-end pipeline wrappers

The current `python/` package is a clean starting point for that layout.

## Summary

The main conclusions from this initial code-reading pass are:

- The useful source of truth is `reference/PaddleOCR`.
- PP-OCRv5 detection and recognition are both present in the checked-in PaddleOCR tree.
- Detector deployment is network output plus Python DB postprocess.
- Recognizer deployment is simpler than training because `MultiHead` returns only the CTC branch in eval mode.
- A sensible first Relax target is `PP-OCRv5_mobile_rec`.
- Full PP-OCR parity requires reproducing the system runtime pipeline, not only the neural network graphs.
