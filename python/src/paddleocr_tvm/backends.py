"""Inference backend adapters."""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from paddleocr_tvm.artifacts import (
    ArtifactLayout,
    read_metadata,
    relax_metadata_path_for,
)
from paddleocr_tvm.errors import ArtifactPreparationError, DependencyUnavailableError


class InferenceRunner(ABC):
    """Minimal array-in array-out interface for model backends."""

    @abstractmethod
    def run(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Run inference."""


class OnnxRuntimeRunner(InferenceRunner):
    """ONNX Runtime adapter."""

    def __init__(self, model_path: Path):
        onnxruntime = _import_optional("onnxruntime", "ONNX Runtime is required.")
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [item.name for item in self._session.get_inputs()]
        self._output_names = [item.name for item in self._session.get_outputs()]

    def run(self, *inputs: np.ndarray) -> list[np.ndarray]:
        feed = {
            name: np.asarray(array, dtype=np.float32)
            for name, array in zip(self._input_names, inputs, strict=False)
        }
        return [np.asarray(output) for output in self._session.run(self._output_names, feed)]


class PaddleInferenceRunner(InferenceRunner):
    """Paddle inference adapter used by the parity harness."""

    def __init__(self, model_dir: Path):
        paddle = _import_optional("paddle", "PaddlePaddle is required for parity mode.")
        config = paddle.inference.Config(
            str(model_dir / "inference.pdmodel"),
            str(model_dir / "inference.pdiparams"),
        )
        config.disable_gpu()
        config.enable_mkldnn()
        config.switch_ir_optim(True)
        config.switch_use_feed_fetch_ops(False)
        self._predictor = paddle.inference.create_predictor(config)
        self._input_names = list(self._predictor.get_input_names())
        self._output_names = list(self._predictor.get_output_names())

    def run(self, *inputs: np.ndarray) -> list[np.ndarray]:
        for name, array in zip(self._input_names, inputs, strict=False):
            handle = self._predictor.get_input_handle(name)
            handle.copy_from_cpu(np.asarray(array, dtype=np.float32))
        self._predictor.run()
        outputs: list[np.ndarray] = []
        for name in self._output_names:
            handle = self._predictor.get_output_handle(name)
            outputs.append(np.asarray(handle.copy_to_cpu()))
        return outputs


class TvmRelaxRunner(InferenceRunner):
    """TVM Relax adapter that compiles from ONNX when necessary."""

    def __init__(
        self,
        layout: ArtifactLayout,
        model_key: str,
        onnx_path: Path,
        *,
        target: str = "llvm",
        shape_dict: dict[str, list[int]] | None = None,
    ):
        self._tvm = _import_optional(
            "tvm",
            "TVM with Relax support is required. Install a Python-importable TVM build first.",
        )
        self._layout = layout
        self._model_key = model_key
        self._onnx_path = onnx_path
        self._target = target
        self._shape_dict = shape_dict
        self._vm = self._build_vm()

    def _build_vm(self) -> Any:
        onnx = _import_optional("onnx", "onnx is required for TVM import.")
        tvm = self._tvm
        metadata_path = relax_metadata_path_for(self._layout, self._model_key)
        metadata = read_metadata(metadata_path) or {}
        if (
            metadata.get("onnx_path") != str(self._onnx_path)
            or metadata.get("target") != self._target
        ):
            metadata = {"onnx_path": str(self._onnx_path), "target": self._target}

        model = onnx.load(str(self._onnx_path))
        mod = tvm.relax.frontend.onnx.from_onnx(model, shape_dict=self._shape_dict)
        executable = tvm.relax.build(mod, target=self._target)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if hasattr(executable, "save_to_file"):
                executable.save_to_file(str(metadata_path.with_suffix(".tvm")))
            elif hasattr(executable, "mod") and hasattr(executable.mod, "export_library"):
                executable.mod.export_library(str(metadata_path.with_suffix(".so")))
        except Exception:
            # Best-effort cache write; in-memory execution is still valid.
            pass
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return tvm.relax.VirtualMachine(executable, tvm.cpu())

    def run(self, *inputs: np.ndarray) -> list[np.ndarray]:
        tvm = self._tvm
        nd_inputs = [tvm.nd.array(np.asarray(array, dtype=np.float32)) for array in inputs]
        result = self._vm["main"](*nd_inputs)
        return _normalize_tvm_outputs(result)


def _normalize_tvm_outputs(result: Any) -> list[np.ndarray]:
    if hasattr(result, "numpy"):
        return [np.asarray(result.numpy())]
    if isinstance(result, (list, tuple)):
        normalized: list[np.ndarray] = []
        for item in result:
            normalized.extend(_normalize_tvm_outputs(item))
        return normalized
    raise ArtifactPreparationError(f"Unsupported TVM output type: {type(result)!r}")


def _import_optional(module_name: str, message: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise DependencyUnavailableError(message) from exc
