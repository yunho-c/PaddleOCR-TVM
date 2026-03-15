from paddleocr_tvm.backend_specs import TVM_LLVM_BACKEND, parse_backend_spec


def test_parse_backend_spec_returns_preset() -> None:
    assert parse_backend_spec("tvm-llvm") == TVM_LLVM_BACKEND


def test_parse_backend_spec_supports_custom_tvm_target() -> None:
    spec = parse_backend_spec("tvm:metal")
    assert spec.kind == "tvm"
    assert spec.target == "metal"
    assert spec.device == "metal"
    assert spec.name == "tvm-metal"
