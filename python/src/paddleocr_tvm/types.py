"""Public result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OCRBox:
    """A detected quadrilateral."""

    points: np.ndarray
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"points": self.points.astype(float).tolist()}
        if self.score is not None:
            payload["score"] = self.score
        return payload


@dataclass(frozen=True)
class OCRTextLine:
    """A recognized text line attached to its quadrilateral."""

    points: np.ndarray
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": self.points.astype(float).tolist(),
            "text": self.text,
            "score": self.score,
        }


@dataclass(frozen=True)
class OCRResult:
    """Full OCR output for an image."""

    lines: list[OCRTextLine]

    def to_dict(self) -> dict[str, Any]:
        return {"lines": [line.to_dict() for line in self.lines]}
