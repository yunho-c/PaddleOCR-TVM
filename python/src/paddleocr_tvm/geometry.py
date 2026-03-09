"""Geometry helpers mirrored from PaddleOCR inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_bgr_image(image: np.ndarray | Image.Image | str | Path) -> np.ndarray:
    """Load an image into BGR uint8 format."""

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    path = Path(image)
    loaded = cv2.imread(str(path))
    if loaded is None:
        raise FileNotFoundError(f"Unable to load image at {path}.")
    return loaded


def sorted_boxes(dt_boxes: np.ndarray) -> list[np.ndarray]:
    """Sort detected boxes top-to-bottom, then left-to-right."""

    if dt_boxes.size == 0:
        return []
    num_boxes = dt_boxes.shape[0]
    boxes = list(sorted(dt_boxes, key=lambda box: (box[0][1], box[0][0])))
    for index in range(num_boxes - 1):
        for inner in range(index, -1, -1):
            if abs(boxes[inner + 1][0][1] - boxes[inner][0][1]) < 10 and (
                boxes[inner + 1][0][0] < boxes[inner][0][0]
            ):
                boxes[inner], boxes[inner + 1] = boxes[inner + 1], boxes[inner]
            else:
                break
    return boxes


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Perspective crop matching PaddleOCR's quad crop logic."""

    if len(points) != 4:
        raise ValueError("Expected a 4-point quadrilateral.")
    points = np.asarray(points, dtype=np.float32)
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
    )
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
    )
    pts_std = np.array(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        transform,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if dst_img.shape[0] / max(dst_img.shape[1], 1) >= 1.5:
        dst_img = np.rot90(dst_img)
    return np.asarray(dst_img)
