from pathlib import Path

import yaml

from paddleocr_tvm.artifacts import load_character_dict
from paddleocr_tvm.postprocess import CTCLabelDecoder


def test_load_character_dict_reads_inference_yaml(tmp_path: Path) -> None:
    inference_dir = tmp_path / "infer"
    inference_dir.mkdir()
    (inference_dir / "inference.yml").write_text(
        yaml.safe_dump({"PostProcess": {"character_dict": ["\u3000", "A", "B"]}}),
        encoding="utf-8",
    )

    assert load_character_dict(inference_dir) == ["\u3000", "A", "B"]


def test_ctc_decoder_supports_inline_character_dict() -> None:
    decoder = CTCLabelDecoder(["\u3000", "A", "B"])
    assert decoder.character == ["blank", "\u3000", "A", "B", " "]


def test_ctc_decoder_preserves_ideographic_space_from_file(tmp_path: Path) -> None:
    dict_path = tmp_path / "dict.txt"
    dict_path.write_text("\u3000\nA\nB\n", encoding="utf-8")

    decoder = CTCLabelDecoder(dict_path)

    assert decoder.character == ["blank", "\u3000", "A", "B", " "]
