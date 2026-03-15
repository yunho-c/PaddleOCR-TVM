"""Artifact management for Paddle, ONNX, and TVM build products."""

from __future__ import annotations

import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import requests  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from paddleocr_tvm.constants import DEFAULT_ARTIFACTS_DIR, MODEL_SPECS, ModelSpec
from paddleocr_tvm.errors import ArtifactPreparationError


@dataclass(frozen=True)
class ArtifactLayout:
    """Resolved artifact directories for a workspace."""

    root: Path
    paddle_dir: Path
    onnx_dir: Path
    relax_root: Path
    relax_dir: Path


def resolve_artifacts_dir(artifacts_dir: Path | str | None = None) -> ArtifactLayout:
    """Resolve the artifact directory layout."""

    root = Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
    return ArtifactLayout(
        root=root,
        paddle_dir=root / "paddle",
        onnx_dir=root / "onnx",
        relax_root=root / "relax",
        relax_dir=root / "relax" / "llvm",
    )


def ensure_directories(layout: ArtifactLayout) -> None:
    """Create the artifact directory tree."""

    layout.paddle_dir.mkdir(parents=True, exist_ok=True)
    layout.onnx_dir.mkdir(parents=True, exist_ok=True)
    layout.relax_root.mkdir(parents=True, exist_ok=True)
    layout.relax_dir.mkdir(parents=True, exist_ok=True)


def get_model_spec(model_key: str) -> ModelSpec:
    """Return the model specification for a named upstream model."""

    try:
        return MODEL_SPECS[model_key]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_SPECS))
        raise ArtifactPreparationError(
            f"Unknown model key {model_key!r}. Expected one of: {known}."
        ) from exc


def download_model_tarball(layout: ArtifactLayout, model_key: str, *, force: bool = False) -> Path:
    """Download an upstream Paddle inference tarball if it is not present."""

    spec = get_model_spec(model_key)
    destination = layout.paddle_dir / spec.filename
    if destination.exists() and not force:
        return destination

    response = requests.get(spec.url, stream=True, timeout=60)
    response.raise_for_status()
    with destination.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    return destination


def unpack_model_tarball(layout: ArtifactLayout, model_key: str, *, force: bool = False) -> Path:
    """Unpack a Paddle inference tarball and return the directory with inference files."""

    tarball = download_model_tarball(layout, model_key, force=force)
    destination = layout.paddle_dir / model_key
    if destination.exists() and not force and find_inference_dir(destination) is not None:
        inference_dir = find_inference_dir(destination)
        if inference_dir is None:
            raise ArtifactPreparationError(
                f"Unable to resolve inference files under {destination}."
            )
        return inference_dir

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as archive:
        archive.extractall(destination)

    inference_dir = find_inference_dir(destination)
    if inference_dir is None:
        raise ArtifactPreparationError(
            f"Expected inference.pdmodel under {destination}, but none was found."
        )
    return inference_dir


def find_inference_dir(root: Path) -> Path | None:
    """Find the directory containing Paddle inference files."""

    for pdmodel in root.rglob("inference.pdmodel"):
        if (pdmodel.parent / "inference.pdiparams").exists():
            return pdmodel.parent
    for json_model in root.rglob("inference.json"):
        if (json_model.parent / "inference.pdiparams").exists():
            return json_model.parent
    return None


def load_character_dict(inference_dir: Path) -> list[str] | None:
    """Load an inline recognition dictionary from a Paddle inference bundle."""

    inference_yml = inference_dir / "inference.yml"
    if not inference_yml.exists():
        return None

    payload = cast(dict[str, Any], yaml.safe_load(inference_yml.read_text(encoding="utf-8")))
    postprocess = payload.get("PostProcess")
    if not isinstance(postprocess, dict):
        return None

    character_dict = postprocess.get("character_dict")
    if not isinstance(character_dict, list):
        return None

    return [str(item) for item in character_dict]


def onnx_path_for(layout: ArtifactLayout, model_key: str) -> Path:
    """Return the canonical ONNX output path for a model."""

    return layout.onnx_dir / f"{model_key}.onnx"


def relax_dir_for_target(layout: ArtifactLayout, target: str = "llvm") -> Path:
    """Return the Relax artifact directory for a specific target."""

    return layout.relax_root / _target_slug(target)


def relax_metadata_path_for(
    layout: ArtifactLayout,
    model_key: str,
    *,
    target: str = "llvm",
) -> Path:
    """Return the canonical Relax metadata path for a model."""

    return relax_dir_for_target(layout, target) / f"{model_key}.json"


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Write a small JSON metadata file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_metadata(path: Path) -> dict[str, Any] | None:
    """Read a metadata JSON file if it exists."""

    if not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _target_slug(target: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", target).strip("_").lower() or "llvm"
