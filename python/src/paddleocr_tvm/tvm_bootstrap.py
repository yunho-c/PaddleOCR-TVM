"""Bootstrap helpers for repo-local TVM imports."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalTvmPaths:
    """Paths for the vendored TVM checkout."""

    root: Path
    python_dir: Path
    build_dir: Path
    build_lib_dir: Path


def local_tvm_paths(project_root: Path | None = None) -> LocalTvmPaths:
    """Return canonical paths for the vendored TVM checkout."""

    root = (project_root or Path(__file__).resolve().parents[3]) / "external" / "tvm"
    return LocalTvmPaths(
        root=root,
        python_dir=root / "python",
        build_dir=root / "build",
        build_lib_dir=root / "build" / "lib",
    )


def bootstrap_local_tvm(paths: LocalTvmPaths | None = None) -> LocalTvmPaths:
    """Expose the vendored TVM checkout to the current Python process."""

    resolved = paths or local_tvm_paths()
    if not resolved.python_dir.is_dir():
        return resolved

    _prepend_sys_path(resolved.python_dir)
    os.environ.setdefault("TVM_HOME", str(resolved.root))
    os.environ.setdefault("TVM_SOURCE_DIR", str(resolved.root))

    if resolved.build_dir.is_dir():
        os.environ.setdefault("TVM_LIBRARY_PATH", str(resolved.build_dir))
        _prepend_env_path("PATH", resolved.build_dir)
        if resolved.build_lib_dir.is_dir():
            _prepend_env_path("PATH", resolved.build_lib_dir)

        if sys.platform.startswith("darwin"):
            _prepend_env_path("DYLD_LIBRARY_PATH", resolved.build_dir)
            if resolved.build_lib_dir.is_dir():
                _prepend_env_path("DYLD_LIBRARY_PATH", resolved.build_lib_dir)
        elif sys.platform.startswith(("linux", "freebsd")):
            _prepend_env_path("LD_LIBRARY_PATH", resolved.build_dir)
            if resolved.build_lib_dir.is_dir():
                _prepend_env_path("LD_LIBRARY_PATH", resolved.build_lib_dir)

    return resolved


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _prepend_env_path(name: str, path: Path) -> None:
    if not path.is_dir():
        return

    value = str(path)
    current = os.environ.get(name, "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if value in entries:
        entries.remove(value)
    os.environ[name] = os.pathsep.join([value, *entries])
