"""Conversion helpers for Paddle inference models."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np

from paddleocr_tvm.artifacts import ArtifactLayout, onnx_path_for, unpack_model_tarball
from paddleocr_tvm.errors import ArtifactPreparationError, DependencyUnavailableError


def ensure_conversion_cli() -> tuple[str, str]:
    """Return the preferred Paddle-to-ONNX CLI and its mode."""

    paddle2onnx = shutil.which("paddle2onnx")
    if paddle2onnx is not None:
        return paddle2onnx, "paddle2onnx"

    paddlex = shutil.which("paddlex")
    if paddlex is not None:
        return paddlex, "paddlex"

    raise DependencyUnavailableError(
        "Paddle-to-ONNX conversion requires either `paddle2onnx` or `paddlex` "
        "on PATH. Install the parity environment first."
    )


def convert_paddle_to_onnx(
    layout: ArtifactLayout,
    model_key: str,
    *,
    force: bool = False,
) -> Path:
    """Convert an upstream Paddle inference directory to ONNX."""

    output_path = onnx_path_for(layout, model_key)
    if output_path.exists() and not force:
        canonicalize_onnx_model(output_path)
        return output_path

    inference_dir = unpack_model_tarball(layout, model_key, force=force)
    converter, mode = ensure_conversion_cli()
    temp_dir = layout.onnx_dir / f"{model_key}_build"
    temp_dir.mkdir(parents=True, exist_ok=True)
    if mode == "paddle2onnx":
        model_filename = (
            "inference.json"
            if (inference_dir / "inference.json").exists()
            else "inference.pdmodel"
        )
        command = [
            converter,
            "--model_dir",
            str(inference_dir),
            "--model_filename",
            model_filename,
            "--params_filename",
            "inference.pdiparams",
            "--save_file",
            str(temp_dir / f"{model_key}.onnx"),
            "--opset_version",
            "17",
        ]
    else:
        command = [
            converter,
            "--paddle2onnx",
            "--paddle_model_dir",
            str(inference_dir),
            "--onnx_model_dir",
            str(temp_dir),
            "--opset_version",
            "17",
        ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            },
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr or exc.stdout or str(exc)
        raise ArtifactPreparationError(
            f"Paddle-to-ONNX conversion failed for {model_key}: {message}"
        ) from exc

    generated = sorted(temp_dir.rglob("*.onnx"))
    if not generated:
        raise ArtifactPreparationError(
            f"PaddleX completed without producing an ONNX file for {model_key}."
        )
    generated[0].replace(output_path)
    canonicalize_onnx_model(output_path)
    return output_path


def canonicalize_onnx_model(model_path: Path) -> None:
    """Rewrite exported ONNX into a form the local Relax frontend handles."""

    onnx = _import_onnx()
    numpy_helper = onnx.numpy_helper
    model = onnx.load(model_path)

    replacements: dict[str, str] = {}
    kept_nodes = []
    for node in model.graph.node:
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            replacements[node.output[0]] = node.input[0]
        else:
            kept_nodes.append(node)

    def rewrite_name(name: str) -> str:
        seen: set[str] = set()
        while name in replacements and name not in seen:
            seen.add(name)
            name = replacements[name]
        return name

    for node in kept_nodes:
        for index, name in enumerate(node.input):
            node.input[index] = rewrite_name(name)

    for output in model.graph.output:
        output.name = rewrite_name(output.name)
    for value in model.graph.value_info:
        value.name = rewrite_name(value.name)

    model.graph.ClearField("node")
    model.graph.node.extend(kept_nodes)

    resize_roi_inputs = {
        node.input[1]
        for node in model.graph.node
        if node.op_type == "Resize" and len(node.input) > 1
    }
    for node in model.graph.node:
        if (
            node.op_type != "Constant"
            or len(node.output) != 1
            or node.output[0] not in resize_roi_inputs
        ):
            continue
        for attribute in node.attribute:
            if attribute.name != "value":
                continue
            value = numpy_helper.to_array(attribute.t)
            if value.shape == (0,) and value.dtype == np.float32:
                tensor = numpy_helper.from_array(
                    np.zeros((8,), dtype=np.float32),
                    name=attribute.t.name or node.output[0],
                )
                attribute.t.CopyFrom(tensor)

    onnx.save(model, model_path)


def _import_onnx() -> ModuleType:
    try:
        import onnx
    except ImportError as exc:
        raise DependencyUnavailableError("onnx is required for ONNX canonicalization.") from exc
    return onnx
