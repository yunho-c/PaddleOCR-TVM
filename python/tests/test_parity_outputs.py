from pathlib import Path

from PIL import Image

from paddleocr_tvm.parity import save_parity_visualization, write_parity_summary


def test_write_parity_summary_creates_json(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "summary.json"
    write_parity_summary({"images": 1, "records": []}, output_path)
    assert output_path.exists()
    assert '"images": 1' in output_path.read_text(encoding="utf-8")


def test_save_parity_visualization_creates_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(image_path)
    output_path = tmp_path / "viz" / "sample.png"
    lines = [
        {
            "points": [[10, 10], [100, 10], [100, 30], [10, 30]],
            "text": "hello",
            "score": 0.95,
        }
    ]

    save_parity_visualization(
        image_path,
        paddle_lines=lines,
        tvm_lines=lines,
        output_path=output_path,
    )

    assert output_path.exists()
