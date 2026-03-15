"""Backend configuration presets for inference and benchmarking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from paddleocr_tvm.errors import ArtifactPreparationError

BackendKind = Literal["paddle", "onnxruntime", "tvm"]


@dataclass(frozen=True)
class BackendSpec:
    """Configuration for a concrete inference backend."""

    name: str
    kind: BackendKind
    target: str | None = None
    device: str | None = None
    paddle_use_mkldnn: bool = True


PADDLE_BACKEND = BackendSpec(name="paddle", kind="paddle", paddle_use_mkldnn=False)
PADDLE_MKLDNN_BACKEND = BackendSpec(
    name="paddle-mkldnn",
    kind="paddle",
    paddle_use_mkldnn=True,
)
ONNXRUNTIME_BACKEND = BackendSpec(name="onnxruntime", kind="onnxruntime")
TVM_LLVM_BACKEND = BackendSpec(name="tvm-llvm", kind="tvm", target="llvm", device="cpu")
TVM_METAL_BACKEND = BackendSpec(name="tvm-metal", kind="tvm", target="metal", device="metal")

BACKEND_PRESETS: dict[str, BackendSpec] = {
    PADDLE_BACKEND.name: PADDLE_BACKEND,
    PADDLE_MKLDNN_BACKEND.name: PADDLE_MKLDNN_BACKEND,
    ONNXRUNTIME_BACKEND.name: ONNXRUNTIME_BACKEND,
    TVM_LLVM_BACKEND.name: TVM_LLVM_BACKEND,
    TVM_METAL_BACKEND.name: TVM_METAL_BACKEND,
}


def parse_backend_spec(value: str) -> BackendSpec:
    """Parse a benchmark/runtime backend name into a backend spec."""

    if value in BACKEND_PRESETS:
        return BACKEND_PRESETS[value]
    if value.startswith("tvm:"):
        target = value.split(":", 1)[1].strip()
        if not target:
            raise ArtifactPreparationError("TVM backend target cannot be empty.")
        return BackendSpec(
            name=f"tvm-{_slug(target)}",
            kind="tvm",
            target=target,
            device=_default_device_for_target(target),
        )
    raise ArtifactPreparationError(
        f"Unknown backend {value!r}. Known presets: {', '.join(sorted(BACKEND_PRESETS))}, "
        "or use tvm:<target>."
    )


def _default_device_for_target(target: str) -> str:
    target_lower = target.lower()
    if "metal" in target_lower:
        return "metal"
    if "cuda" in target_lower:
        return "cuda"
    if "opencl" in target_lower:
        return "opencl"
    if "vulkan" in target_lower:
        return "vulkan"
    if "rocm" in target_lower:
        return "rocm"
    if "hexagon" in target_lower:
        return "hexagon"
    return "cpu"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
