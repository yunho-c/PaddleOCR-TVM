from paddleocr_tvm.cli import build_parser, main


def test_main_prints_version(capsys) -> None:
    assert main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "0.1.0"


def test_parser_knows_mobile_subcommands() -> None:
    parser = build_parser()
    namespace = parser.parse_args(["prepare-mobile"])
    assert namespace.command == "prepare-mobile"


def test_parity_parser_accepts_output_options(tmp_path) -> None:
    parser = build_parser()
    namespace = parser.parse_args(
        [
            "parity-mobile",
            "--images",
            str(tmp_path),
            "--output-json",
            str(tmp_path / "summary.json"),
            "--visualizations-dir",
            str(tmp_path / "viz"),
        ]
    )
    assert namespace.command == "parity-mobile"
    assert namespace.output_json == tmp_path / "summary.json"
    assert namespace.visualizations_dir == tmp_path / "viz"
