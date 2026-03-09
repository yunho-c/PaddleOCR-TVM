from paddleocr_tvm.cli import build_parser, main


def test_main_prints_version(capsys) -> None:
    assert main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "0.1.0"


def test_parser_knows_mobile_subcommands() -> None:
    parser = build_parser()
    namespace = parser.parse_args(["prepare-mobile"])
    assert namespace.command == "prepare-mobile"
