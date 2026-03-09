"""Custom exceptions for PaddleOCR-TVM."""

from __future__ import annotations


class PaddleOcrTvmError(RuntimeError):
    """Base runtime error for the package."""


class DependencyUnavailableError(PaddleOcrTvmError):
    """Raised when an optional runtime dependency is unavailable."""


class ArtifactPreparationError(PaddleOcrTvmError):
    """Raised when model artifacts cannot be prepared."""
