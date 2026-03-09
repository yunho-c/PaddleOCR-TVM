import numpy as np

from paddleocr_tvm.geometry import get_rotate_crop_image, sorted_boxes


def test_sorted_boxes_orders_by_row_then_column() -> None:
    boxes = np.array(
        [
            [[50, 50], [70, 50], [70, 70], [50, 70]],
            [[10, 10], [30, 10], [30, 30], [10, 30]],
            [[40, 12], [60, 12], [60, 32], [40, 32]],
        ],
        dtype=np.float32,
    )
    ordered = sorted_boxes(boxes)
    assert ordered[0][0][0] == 10
    assert ordered[1][0][0] == 40
    assert ordered[2][0][0] == 50


def test_get_rotate_crop_image_returns_non_empty_crop() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[10:30, 10:30] = 255
    points = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
    crop = get_rotate_crop_image(image, points)
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0
    assert crop.mean() > 0
