"""Conversion helpers for Paddle inference models."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from paddleocr_tvm.artifacts import ArtifactLayout, onnx_path_for, unpack_model_tarball
from paddleocr_tvm.errors import ArtifactPreparationError, DependencyUnavailableError


def ensure_paddlex_cli() -> str:
    """Return the PaddleX CLI path or raise a clear error."""

    paddlex = shutil.which("paddlex")
    if paddlex is None:
        raise DependencyUnavailableError(
            "The `paddlex` CLI is required for Paddle-to-ONNX conversion. "
            "Install the parity environment or make `paddlex` available on PATH."
        )
    return paddlex


def convert_paddle_to_onnx(
    layout: ArtifactLayout,
    model_key: str,
    *,
    force: bool = False,
) -> Path:
    """Convert an upstream Paddle inference directory to ONNX."""

    output_path = onnx_path_for(layout, model_key)
    if output_path.exists() and not force:
        return output_path

    inference_dir = unpack_model_tarball(layout, model_key, force=force)
    ensure_paddlex_cli()
    temp_dir = layout.onnx_dir / f"{model_key}_build"
    temp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "paddlex",
        "--paddle2onnx",
        "--paddle_model_dir",
        str(inference_dir),
        "--onnx_model_dir",
        str(temp_dir),
        "--opset_version",
        "17",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
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
    return output_path
