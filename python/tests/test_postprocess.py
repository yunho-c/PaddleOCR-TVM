from pathlib import Path

import numpy as np

from paddleocr_tvm.postprocess import CTCLabelDecoder


def test_ctc_decoder_collapses_duplicates_and_blanks(tmp_path: Path) -> None:
    dict_path = tmp_path / "dict.txt"
    dict_path.write_text("a\nb\n", encoding="utf-8")
    decoder = CTCLabelDecoder(dict_path, use_space_char=False)
    preds = np.array(
        [
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.1, 0.7, 0.2],
                [0.9, 0.1, 0.0],
                [0.1, 0.2, 0.7],
            ]
        ],
        dtype=np.float32,
    )
    decoded = decoder.decode(preds)
    assert decoded == [("ab", 0.75)]
