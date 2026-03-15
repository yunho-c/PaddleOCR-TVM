from pathlib import Path

from paddleocr_tvm.benchmark import write_benchmark_csv, write_benchmark_summary


def _summary() -> dict[str, object]:
    return {
        "dataset": {"images": 1, "recognition_batches": 1, "recognition_crops": 2},
        "warmup": 1,
        "repeat": 2,
        "backends": [
            {
                "backend": "tvm-llvm",
                "kind": "tvm",
                "target": "llvm",
                "device": "cpu",
                "detector": {
                    "calls": 2,
                    "logical_items": 2,
                    "total_ms": 10.0,
                    "mean_ms": 5.0,
                    "median_ms": 5.0,
                    "p90_ms": 5.0,
                    "p95_ms": 5.0,
                    "min_ms": 5.0,
                    "max_ms": 5.0,
                    "throughput_calls_per_s": 200.0,
                    "throughput_items_per_s": 200.0,
                },
                "recognizer": {
                    "calls": 2,
                    "logical_items": 4,
                    "total_ms": 8.0,
                    "mean_ms": 4.0,
                    "median_ms": 4.0,
                    "p90_ms": 4.0,
                    "p95_ms": 4.0,
                    "min_ms": 4.0,
                    "max_ms": 4.0,
                    "throughput_calls_per_s": 250.0,
                    "throughput_items_per_s": 500.0,
                },
                "end_to_end": {
                    "calls": 2,
                    "logical_items": 2,
                    "total_ms": 20.0,
                    "mean_ms": 10.0,
                    "median_ms": 10.0,
                    "p90_ms": 10.0,
                    "p95_ms": 10.0,
                    "min_ms": 10.0,
                    "max_ms": 10.0,
                    "throughput_calls_per_s": 100.0,
                    "throughput_items_per_s": 100.0,
                },
            }
        ],
    }


def test_write_benchmark_summary_creates_json(tmp_path: Path) -> None:
    output_path = tmp_path / "bench" / "summary.json"
    write_benchmark_summary(_summary(), output_path)
    assert output_path.exists()
    assert '"backend": "tvm-llvm"' in output_path.read_text(encoding="utf-8")


def test_write_benchmark_csv_creates_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "bench" / "summary.csv"
    write_benchmark_csv(_summary(), output_path)
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "backend,kind,target,device,scope" in content
    assert "tvm-llvm,tvm,llvm,cpu,detector" in content
