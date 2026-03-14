import os
import sys
from pathlib import Path

from paddleocr_tvm.tvm_bootstrap import bootstrap_local_tvm, local_tvm_paths


def test_local_tvm_paths_uses_repo_layout(tmp_path: Path) -> None:
    paths = local_tvm_paths(tmp_path)
    assert paths.root == tmp_path / "external" / "tvm"
    assert paths.python_dir == tmp_path / "external" / "tvm" / "python"
    assert paths.build_dir == tmp_path / "external" / "tvm" / "build"
    assert paths.build_lib_dir == tmp_path / "external" / "tvm" / "build" / "lib"


def test_bootstrap_local_tvm_sets_env_and_sys_path(monkeypatch, tmp_path: Path) -> None:
    tvm_root = tmp_path / "external" / "tvm"
    (tvm_root / "python").mkdir(parents=True)
    (tvm_root / "build" / "lib").mkdir(parents=True)

    monkeypatch.setattr(sys, "path", ["existing-path"])
    for name in ["TVM_HOME", "TVM_SOURCE_DIR", "TVM_LIBRARY_PATH", "PATH", "DYLD_LIBRARY_PATH"]:
        monkeypatch.delenv(name, raising=False)

    paths = bootstrap_local_tvm(local_tvm_paths(tmp_path))

    assert sys.path[0] == str(paths.python_dir)
    assert os.environ["TVM_HOME"] == str(paths.root)
    assert os.environ["TVM_SOURCE_DIR"] == str(paths.root)
    assert os.environ["TVM_LIBRARY_PATH"] == str(paths.build_dir)

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert path_entries[0] == str(paths.build_lib_dir)
    assert path_entries[1] == str(paths.build_dir)

    if sys.platform.startswith("darwin"):
        dyld_entries = os.environ["DYLD_LIBRARY_PATH"].split(os.pathsep)
        assert dyld_entries[0] == str(paths.build_lib_dir)
        assert dyld_entries[1] == str(paths.build_dir)
