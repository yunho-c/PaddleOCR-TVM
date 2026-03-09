# PaddleOCR-TVM Python

CPU-first PP-OCRv5 mobile OCR tooling for this repository, with an artifact pipeline
for Paddle inference models, ONNX conversion, and TVM Relax integration.

## Quickstart

Create the base environment:

```bash
pixi install
pixi run -e dev check
```

Prepare Paddle and ONNX artifacts:

```bash
pixi run prepare-mobile
```

Run OCR on an image:

```bash
pixi run ocr-mobile -- --image /path/to/image.png --artifacts-dir .artifacts
```

## TVM note

This package expects a modern Python-importable TVM build with Relax support. The
default environment in this repository does not bootstrap TVM automatically because a
current Relax-capable wheel was not available from the default channels in this
environment. Commands that need TVM fail fast with a clear setup error.

## Available tasks

- `pixi run prepare-mobile`
- `pixi run ocr-mobile -- --image /path/to/image.png`
- `pixi run -e parity parity-mobile -- --images /path/to/images`
- `pixi run -e dev format`
- `pixi run -e dev lint`
- `pixi run -e dev typecheck`
- `pixi run -e dev test`
- `pixi run -e dev test-cov`
- `pixi run -e dev check`
