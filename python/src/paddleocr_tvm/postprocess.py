"""Detector and recognizer postprocessing logic."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pyclipper  # type: ignore[import-untyped]
from shapely.geometry import Polygon  # type: ignore[import-untyped]


class DBPostProcess:
    """DBNet-style box extraction."""

    def __init__(
        self,
        *,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        score_mode: str = "fast",
    ):
        if score_mode not in {"fast", "slow"}:
            raise ValueError("score_mode must be 'fast' or 'slow'.")
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode

    def __call__(self, pred: np.ndarray, shape_list: np.ndarray) -> list[np.ndarray]:
        maps = pred[:, 0, :, :]
        segmentation = maps > self.thresh
        boxes_batch: list[np.ndarray] = []
        for batch_index in range(maps.shape[0]):
            src_h, src_w, _, _ = shape_list[batch_index]
            boxes, _ = self._boxes_from_bitmap(
                maps[batch_index],
                segmentation[batch_index],
                int(src_w),
                int(src_h),
            )
            boxes_batch.append(boxes)
        return boxes_batch

    def _boxes_from_bitmap(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
        dest_width: int,
        dest_height: int,
    ) -> tuple[np.ndarray, list[float]]:
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        num_contours = min(len(contours), self.max_candidates)
        boxes: list[np.ndarray] = []
        scores: list[float] = []
        height, width = bitmap.shape
        for contour in contours[:num_contours]:
            points, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points_arr = np.asarray(points)
            score = (
                self._box_score_fast(pred, points_arr.reshape(-1, 2))
                if self.score_mode == "fast"
                else self._box_score_slow(pred, contour)
            )
            if score < self.box_thresh:
                continue
            box = self._unclip(points_arr, self.unclip_ratio)
            if len(box) != 1:
                continue
            expanded = np.asarray(box).reshape(-1, 1, 2)
            expanded_points, sside = self._get_mini_boxes(expanded)
            if sside < self.min_size + 2:
                continue
            quad = np.asarray(expanded_points)
            quad[:, 0] = np.clip(np.round(quad[:, 0] / width * dest_width), 0, dest_width)
            quad[:, 1] = np.clip(np.round(quad[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(quad.astype(np.float32))
            scores.append(score)
        if boxes:
            return np.stack(boxes), scores
        return np.zeros((0, 4, 2), dtype=np.float32), scores

    @staticmethod
    def _unclip(box: np.ndarray, unclip_ratio: float) -> list[np.ndarray]:
        polygon = Polygon(box)
        distance = polygon.area * unclip_ratio / polygon.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return cast(list[np.ndarray], offset.Execute(distance))

    @staticmethod
    def _get_mini_boxes(contour: np.ndarray) -> tuple[list[np.ndarray], float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda point: point[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
        height, width = bitmap.shape[:2]
        box = box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, width - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, width - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, height - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, height - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, [box.reshape(-1, 2).astype("int32")], 1)
        return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])

    @staticmethod
    def _box_score_slow(bitmap: np.ndarray, contour: np.ndarray) -> float:
        height, width = bitmap.shape[:2]
        contour = contour.copy().reshape(-1, 2)
        xmin = np.clip(np.min(contour[:, 0]), 0, width - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, width - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, height - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, height - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin
        cv2.fillPoly(mask, [contour.astype("int32")], 1)
        return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


class CTCLabelDecoder:
    """Minimal CTC decoder compatible with PP-OCRv5 dictionaries."""

    def __init__(self, dict_path: Path, *, use_space_char: bool = True):
        with dict_path.open("r", encoding="utf-8") as handle:
            self.character = ["blank"] + [line.strip() for line in handle if line.strip()]
        if use_space_char and " " not in self.character:
            self.character.append(" ")
        self.reverse = "arabic" in dict_path.name

    def decode(self, preds: np.ndarray) -> list[tuple[str, float]]:
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        results: list[tuple[str, float]] = []
        for batch_index in range(len(preds_idx)):
            selection = np.ones(len(preds_idx[batch_index]), dtype=bool)
            selection[1:] = preds_idx[batch_index][1:] != preds_idx[batch_index][:-1]
            selection &= preds_idx[batch_index] != 0
            chars = [self.character[int(idx)] for idx in preds_idx[batch_index][selection]]
            text = "".join(chars)
            if self.reverse:
                text = self._pred_reverse(text)
            confidence = (
                float(np.mean(preds_prob[batch_index][selection]))
                if np.any(selection)
                else 0.0
            )
            results.append((text, confidence))
        return results

    @staticmethod
    def _pred_reverse(text: str) -> str:
        chunks: list[str] = []
        current = ""
        for char in text:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", char)):
                if current:
                    chunks.append(current)
                chunks.append(char)
                current = ""
            else:
                current += char
        if current:
            chunks.append(current)
        return "".join(chunks[::-1])
