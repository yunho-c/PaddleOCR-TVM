# PaddleOCR-TVM Python

CPU-first PP-OCRv5 mobile OCR tooling for this repository, with an artifact pipeline
for Paddle inference models, ONNX conversion, and TVM Relax integration.

## Quickstart

Create the base environment:

```bash
pixi install
pixi run -e dev check
```

Build the vendored TVM checkout:

```bash
pixi run build-tvm
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

This package defaults to the vendored TVM checkout in
[`../external/tvm`](../external/tvm). At runtime it prepends
[`../external/tvm/python`](../external/tvm/python) to `sys.path` and points TVM at
[`../external/tvm/build`](../external/tvm/build) for the native libraries. The Pixi
environment still installs `apache-tvm-ffi` and TVM's Python-side dependencies, but
the `tvm` module itself is loaded from the submodule source tree.

If TVM-backed commands fail, rebuild the native libraries with `pixi run build-tvm`
and retry from the `python/` project directory.

## Available tasks

- `pixi run build-tvm`
- `pixi run prepare-mobile`
- `pixi run ocr-mobile -- --image /path/to/image.png`
- `pixi run -e parity parity-mobile -- --images /path/to/images`
- `pixi run -e dev format`
- `pixi run -e dev lint`
- `pixi run -e dev typecheck`
- `pixi run -e dev test`
- `pixi run -e dev test-cov`
- `pixi run -e dev check`
