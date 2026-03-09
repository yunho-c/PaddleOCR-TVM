import numpy as np

from paddleocr_tvm.preprocess import det_resize_for_test, prepare_rec_batch


def test_det_resize_for_test_rounds_to_multiple_of_32() -> None:
    image = np.zeros((333, 777, 3), dtype=np.uint8)
    resized, shape = det_resize_for_test(image)
    assert resized.shape[0] % 32 == 0
    assert resized.shape[1] % 32 == 0
    assert shape.shape == (4,)


def test_prepare_rec_batch_sorts_and_pads() -> None:
    image_a = np.zeros((48, 120, 3), dtype=np.uint8)
    image_b = np.zeros((48, 240, 3), dtype=np.uint8)
    batch, indices, ratios, max_wh_ratio = prepare_rec_batch([image_b, image_a], (3, 48, 320))
    assert batch.shape[0] == 2
    assert indices == [1, 0]
    assert ratios[0] < ratios[1]
    assert max_wh_ratio >= ratios[1]
