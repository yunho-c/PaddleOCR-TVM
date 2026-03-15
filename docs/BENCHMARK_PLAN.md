# Benchmark Plan

## Goal

Add a repeatable benchmark setup for PP-OCRv5 mobile inference that can compare:

- Paddle Inference on CPU
- TVM Relax on CPU via `llvm`
- ONNX Runtime on CPU
- future TVM targets such as `metal`, `opencl`, `vulkan`, or Qualcomm-specific runtimes

The benchmark setup should be useful for both local regression tracking and backend-to-backend comparison.

## Design Principles

### Separate correctness from speed

Parity and benchmarking solve different problems:

- parity verifies output behavior
- benchmarking measures latency and throughput

The benchmark command should not depend on exact-output matching at runtime, although it may reuse the same artifact preparation and OCR pipeline pieces.

### Separate compile time from steady-state inference

TVM introduces compile and shape-specialization effects that are not comparable to steady-state runtime latency.

The benchmark flow should therefore distinguish:

- one-time preparation
- warmup
- timed steady-state execution

### Benchmark multiple scopes

A single end-to-end OCR number is too coarse because this repository still keeps substantial Python-side logic outside the neural network graphs.

The benchmark setup should report at least:

- detector latency
- recognizer latency
- end-to-end OCR latency

Recognizer benchmarking should use canonical crops extracted once outside the timed loop so detector variability does not pollute recognizer-only measurements.

### Make backend choice explicit

Backend selection should be represented by an explicit configuration object rather than hidden in the loader implementation.

This allows the same benchmark harness to compare:

- `paddle`
- `onnxruntime`
- `tvm-llvm`
- future `tvm-metal`
- future `tvm-opencl`
- future remote/device-specific backends

## Current Repository Constraints

The current codebase already has these building blocks:

- Paddle inference runner
- ONNX Runtime runner
- TVM Relax runner
- shared Paddle-compatible preprocessing and postprocessing
- artifact preparation for Paddle, ONNX, and TVM

However, there are a few constraints:

- the default mobile pipeline currently assumes TVM unless told otherwise
- TVM artifact storage was originally `relax/llvm` specific
- `PaddleInferenceRunner` is CPU-only today
- the current TVM build script enables `llvm` but disables `metal`, `opencl`, and `vulkan`

The initial implementation should therefore focus on the backends that already exist in the repo:

- `paddle`
- `onnxruntime`
- `tvm-llvm`

## Recommended Architecture

### 1. Backend specification layer

Introduce a small `BackendSpec` type with fields such as:

- backend kind
- human-readable name
- TVM target
- TVM device
- Paddle-specific options such as MKLDNN enablement

Add preset names for the first slice:

- `paddle`
- `paddle-mkldnn`
- `onnxruntime`
- `tvm-llvm`
- `tvm-metal`

The initial benchmark command should default to:

- `paddle`
- `onnxruntime`
- `tvm-llvm`

### 2. Target-aware TVM artifact layout

TVM artifacts should be stored under target-specific directories, for example:

- `.artifacts/relax/llvm/`
- `.artifacts/relax/metal/`
- `.artifacts/relax/opencl/`

This avoids collisions between backends and makes benchmark runs reproducible.

### 3. Backend-aware pipeline loading

Keep the existing `MobileDetector`, `MobileRecognizer`, and `MobileOCRPipeline` classes, but allow loading them with an explicit backend spec.

That means the benchmark harness can instantiate:

- a Paddle detector
- an ONNX Runtime recognizer
- a TVM OCR pipeline

without duplicating pipeline logic.

### 4. Dedicated benchmark module

Add a benchmark module that:

- resolves a backend list
- loads canonical benchmark inputs
- builds recognizer crops once using a reference detector
- performs warmup
- times detector, recognizer, and end-to-end OCR scopes
- writes JSON and optional CSV summaries

### 5. CLI entry point

Add a command like:

```bash
paddleocr-tvm benchmark-mobile \
  --images /path/to/images \
  --backend paddle \
  --backend onnxruntime \
  --backend tvm-llvm \
  --warmup 1 \
  --repeat 5 \
  --output-json .artifacts/benchmarks/latest.json \
  --output-csv .artifacts/benchmarks/latest.csv
```

## Benchmark Methodology

### Dataset handling

The benchmark command should:

1. load all full images once
2. extract recognizer crops once with a canonical reference detector
3. use the same full images and crop groups for every backend

This keeps the inputs stable across runs.

### Warmup

Warmup should execute the full case list for each scope before timing.

This is especially important for TVM because:

- the first execution may compile a shape-specialized executable
- the first device invocation may include runtime initialization costs

### Timed metrics

For each backend and scope, report:

- `calls`
- `logical_items`
- `total_ms`
- `mean_ms`
- `median_ms`
- `p90_ms`
- `p95_ms`
- `min_ms`
- `max_ms`
- `throughput_calls_per_s`
- `throughput_items_per_s`

For detector and end-to-end scopes, `logical_items` should be the number of images.

For recognizer scope, `logical_items` should be the number of text crops processed.

## Scope of the First Implementation

The first implementation in this repository should include:

- backend specs for `paddle`, `onnxruntime`, and `tvm-llvm`
- target-aware TVM metadata/artifact storage
- `benchmark-mobile` CLI
- JSON output
- CSV output
- detector, recognizer, and end-to-end timing
- a Pixi task for running the benchmark from the parity environment

The first implementation should not try to solve:

- remote-device orchestration
- Qualcomm-specific deployment runtimes
- automatic Metal/OpenCL TVM builds
- profiler-level kernel breakdowns
- throughput benchmarking for large batch serving

## Future Extensions

After the first benchmark slice is stable, the next useful extensions are:

1. enable additional TVM targets in the local build script
2. add `tvm-metal` and `tvm-opencl` benchmark presets
3. record environment metadata such as host CPU, OS, TVM commit, and Python version
4. add detector-map-only and recognizer-logit-only microbenchmarks
5. support benchmark fixture manifests instead of plain image directories
6. add remote runner abstractions for Android/Qualcomm devices

## Acceptance Criteria

The benchmark implementation is considered usable when:

- a single CLI command runs benchmark comparisons for the current local backends
- results are written as stable JSON and CSV files
- TVM `llvm` and future TVM targets do not overwrite each other’s artifacts
- the benchmark code reuses the existing pipeline/preprocess/postprocess logic instead of reimplementing OCR separately
- the implementation passes repo checks and unit tests
