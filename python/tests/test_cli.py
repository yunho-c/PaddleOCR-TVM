from paddleocr_tvm.cli import main


def test_main_prints_ready_message(capsys) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "scaffold is ready" in captured.out


def test_main_prints_version(capsys) -> None:
    assert main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "0.1.0"
