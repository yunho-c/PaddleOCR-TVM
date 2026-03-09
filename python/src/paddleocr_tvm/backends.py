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
        model_path = model_dir / "inference.pdmodel"
        if not model_path.exists():
            model_path = model_dir / "inference.json"
        config = paddle.inference.Config(
            str(model_path),
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
        self._input_names = self._load_input_names()
        self._vm_cache: dict[str, Any] = {}
        if shape_dict is not None:
            self._vm_cache[self._shape_cache_key(shape_dict)] = self._build_vm(shape_dict)

    def _build_vm(self, shape_dict: dict[str, list[int]]) -> Any:
        onnx = _import_optional("onnx", "onnx is required for TVM import.")
        tvm = self._tvm
        relax_onnx_frontend = importlib.import_module("tvm.relax.frontend.onnx")
        metadata_path = relax_metadata_path_for(
            self._layout,
            f"{self._model_key}__{self._shape_cache_key(shape_dict)}",
        )
        metadata = read_metadata(metadata_path) or {}
        if (
            metadata.get("onnx_path") != str(self._onnx_path)
            or metadata.get("target") != self._target
        ):
            metadata = {
                "onnx_path": str(self._onnx_path),
                "shape_dict": shape_dict,
                "target": self._target,
            }

        model = onnx.load(str(self._onnx_path))
        mod = relax_onnx_frontend.from_onnx(model, shape_dict=shape_dict)
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
        vm = self._get_vm(inputs)
        nd_inputs = [tvm.nd.array(np.asarray(array, dtype=np.float32)) for array in inputs]
        result = vm["main"](*nd_inputs)
        return _normalize_tvm_outputs(result)

    def _get_vm(self, inputs: tuple[np.ndarray, ...]) -> Any:
        shape_dict = self._shape_dict_for_inputs(inputs)
        cache_key = self._shape_cache_key(shape_dict)
        if cache_key not in self._vm_cache:
            self._vm_cache[cache_key] = self._build_vm(shape_dict)
        return self._vm_cache[cache_key]

    def _load_input_names(self) -> list[str]:
        onnx = _import_optional("onnx", "onnx is required for TVM import.")
        model = onnx.load(str(self._onnx_path))
        return [value.name for value in model.graph.input]

    def _shape_dict_for_inputs(self, inputs: tuple[np.ndarray, ...]) -> dict[str, list[int]]:
        if len(inputs) != len(self._input_names):
            raise ArtifactPreparationError(
                f"Expected {len(self._input_names)} inputs for {self._model_key}, "
                f"got {len(inputs)}."
            )
        return {
            name: list(np.asarray(array).shape)
            for name, array in zip(self._input_names, inputs, strict=False)
        }

    @staticmethod
    def _shape_cache_key(shape_dict: dict[str, list[int]]) -> str:
        parts: list[str] = []
        for name in sorted(shape_dict):
            dims = "x".join(str(dim) for dim in shape_dict[name])
            parts.append(f"{name}_{dims}")
        return "__".join(parts)


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
