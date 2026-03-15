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


def test_benchmark_parser_accepts_backend_and_output_options(tmp_path) -> None:
    parser = build_parser()
    namespace = parser.parse_args(
        [
            "benchmark-mobile",
            "--images",
            str(tmp_path),
            "--backend",
            "paddle",
            "--backend",
            "tvm-llvm",
            "--output-json",
            str(tmp_path / "bench.json"),
            "--output-csv",
            str(tmp_path / "bench.csv"),
            "--warmup",
            "2",
            "--repeat",
            "7",
        ]
    )
    assert namespace.command == "benchmark-mobile"
    assert namespace.backends == ["paddle", "tvm-llvm"]
    assert namespace.output_json == tmp_path / "bench.json"
    assert namespace.output_csv == tmp_path / "bench.csv"
    assert namespace.warmup == 2
    assert namespace.repeat == 7
