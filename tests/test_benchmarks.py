"""
Check that the benchmark scripts in `benchmarks/` run, using small problem
sizes so that the tests are fast.
"""
import importlib.util
import json
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"


def _load_benchmark(name):
    spec = importlib.util.spec_from_file_location(
        f"benchmark_{name}", BENCHMARKS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cropping_benchmark_save_and_compare(tmp_path):
    benchmark = _load_benchmark("cropping")
    fp_results = tmp_path / "results.json"
    args = ["--n-source", "2000", "5000", "--n-lam", "10", "--margin", "5.0"]

    assert benchmark.main(args + ["--repeats", "2", "--save", str(fp_results)]) == 0
    results = json.loads(fp_results.read_text())
    assert set(results["runs"]) == {"2000", "5000"}
    for run in results["runs"].values():
        assert len(run["timings"]) == 2
        assert 0 < run["n_kept"] < 5000

    # comparing against the same version keeps the same points
    assert benchmark.main(args + ["--compare-to", str(fp_results)]) == 0

    # a reference with different kept points is reported as a mismatch
    results["runs"]["2000"]["kept_grid_index"] = results["runs"]["2000"][
        "kept_grid_index"
    ][:-1]
    fp_results.write_text(json.dumps(results))
    assert benchmark.main(args + ["--compare-to", str(fp_results)]) == 1


def test_cropping_benchmark_max_slowdown(tmp_path):
    benchmark = _load_benchmark("cropping")
    fp_results = tmp_path / "results.json"
    args = ["--n-source", "2000", "--n-lam", "10", "--margin", "5.0"]
    assert benchmark.main(args + ["--save", str(fp_results)]) == 0
    results = json.loads(fp_results.read_text())

    # a reference that was much faster makes this run count as a slowdown
    results["runs"]["2000"]["median"] /= 100.0
    fp_results.write_text(json.dumps(results))
    compare = args + ["--compare-to", str(fp_results)]
    assert benchmark.main(compare + ["--max-slowdown-pct", "5"]) == 1
    # without a limit only the kept points are checked
    assert benchmark.main(compare) == 0

    # a reference that was much slower is within any limit
    results["runs"]["2000"]["median"] *= 100.0 * 100.0
    fp_results.write_text(json.dumps(results))
    assert benchmark.main(compare + ["--max-slowdown-pct", "5"]) == 0
