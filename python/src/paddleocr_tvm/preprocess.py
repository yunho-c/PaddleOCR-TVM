"""Preprocessing helpers that mirror PaddleOCR mobile inference."""

from __future__ import annotations

import math

import cv2
import numpy as np

from paddleocr_tvm.constants import DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, DET_MEAN, DET_STD


def det_resize_for_test(
    image: np.ndarray,
    *,
    limit_side_len: int = DET_LIMIT_SIDE_LEN,
    limit_type: str = DET_LIMIT_TYPE,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a detector input and return shape metadata."""

    src_h, src_w, _ = image.shape
    if src_h + src_w < 64:
        padded = np.zeros((max(32, src_h), max(32, src_w), 3), dtype=np.uint8)
        padded[:src_h, :src_w, :] = image
        image = padded
        src_h, src_w, _ = image.shape

    max_side = max(src_h, src_w)
    min_side = min(src_h, src_w)
    if limit_type == "max":
        ratio = float(limit_side_len) / max_side if max_side > limit_side_len else 1.0
    elif limit_type == "min":
        ratio = float(limit_side_len) / min_side if min_side < limit_side_len else 1.0
    elif limit_type == "resize_long":
        ratio = float(limit_side_len) / max_side
    else:
        raise ValueError(f"Unsupported limit_type: {limit_type}")

    resize_h = max(int(round((src_h * ratio) / 32) * 32), 32)
    resize_w = max(int(round((src_w * ratio) / 32) * 32), 32)
    resized = cv2.resize(image, (resize_w, resize_h))
    ratio_h = resize_h / float(src_h)
    ratio_w = resize_w / float(src_w)
    shape = np.array([src_h, src_w, ratio_h, ratio_w], dtype=np.float32)
    return resized, shape


def normalize_det_image(image: np.ndarray) -> np.ndarray:
    """Normalize and convert a detector image to CHW."""

    normalized = image.astype("float32") / 255.0
    normalized -= np.asarray(DET_MEAN, dtype=np.float32)
    normalized /= np.asarray(DET_STD, dtype=np.float32)
    return normalized.transpose(2, 0, 1)


def prepare_det_batch(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prepare a single detector batch."""

    resized, shape = det_resize_for_test(image)
    tensor = normalize_det_image(resized)[np.newaxis, :].astype(np.float32)
    return tensor, shape[np.newaxis, :]


def resize_norm_rec_image(
    image: np.ndarray,
    max_wh_ratio: float,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Resize, normalize, and pad a recognition crop."""

    img_c, img_h, img_w = image_shape
    if img_c != image.shape[2]:
        raise ValueError(f"Expected {img_c} channels, got {image.shape[2]}.")
    padded_width = max(img_w, int(math.ceil(img_h * max_wh_ratio)))
    height, width = image.shape[:2]
    ratio = width / float(height)
    resized_width = int(math.ceil(img_h * ratio))
    resized_w = padded_width if resized_width > padded_width else resized_width
    resized = cv2.resize(image, (resized_w, img_h)).astype("float32")
    resized = resized.transpose((2, 0, 1)) / 255.0
    resized -= 0.5
    resized /= 0.5
    padded = np.zeros((img_c, img_h, padded_width), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded


def prepare_rec_batch(
    images: list[np.ndarray],
    image_shape: tuple[int, int, int],
) -> tuple[np.ndarray, list[int], list[float], float]:
    """Prepare a sorted recognition batch and return sort metadata."""

    if not images:
        raise ValueError("Recognition batch cannot be empty.")
    widths = [image.shape[1] / float(image.shape[0]) for image in images]
    indices = list(np.argsort(np.asarray(widths)))
    img_c, img_h, img_w = image_shape
    max_wh_ratio = img_w / img_h
    sorted_ratios: list[float] = []
    for index in indices:
        ratio = images[index].shape[1] / float(images[index].shape[0])
        max_wh_ratio = max(max_wh_ratio, ratio)
        sorted_ratios.append(ratio)
    batch = [
        resize_norm_rec_image(images[index], max_wh_ratio, image_shape)[np.newaxis, :]
        for index in indices
    ]
    return np.concatenate(batch).astype(np.float32), indices, sorted_ratios, max_wh_ratio
