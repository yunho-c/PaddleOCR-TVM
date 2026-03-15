from pathlib import Path

from paddleocr_tvm.artifacts import (
    get_model_spec,
    onnx_path_for,
    relax_dir_for_target,
    relax_metadata_path_for,
    resolve_artifacts_dir,
)


def test_resolve_artifacts_dir_uses_expected_layout(tmp_path: Path) -> None:
    layout = resolve_artifacts_dir(tmp_path)
    assert layout.paddle_dir == tmp_path / "paddle"
    assert layout.onnx_dir == tmp_path / "onnx"
    assert layout.relax_root == tmp_path / "relax"
    assert layout.relax_dir == tmp_path / "relax" / "llvm"


def test_get_model_spec_returns_mobile_detector() -> None:
    spec = get_model_spec("mobile_det")
    assert spec.key == "mobile_det"
    assert spec.filename.endswith(".tar")


def test_onnx_path_for_is_stable(tmp_path: Path) -> None:
    layout = resolve_artifacts_dir(tmp_path)
    assert onnx_path_for(layout, "mobile_rec") == tmp_path / "onnx" / "mobile_rec.onnx"


def test_relax_target_paths_are_target_aware(tmp_path: Path) -> None:
    layout = resolve_artifacts_dir(tmp_path)
    assert relax_dir_for_target(layout, "metal") == tmp_path / "relax" / "metal"
    assert (
        relax_metadata_path_for(layout, "mobile_rec", target="llvm -mcpu=apple-m2")
        == tmp_path / "relax" / "llvm_mcpu_apple_m2" / "mobile_rec.json"
    )
