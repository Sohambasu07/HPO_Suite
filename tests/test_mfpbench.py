from __future__ import annotations

import argparse
from pathlib import Path

import mfpbench


def test_mfpbench(data_dir: Path,
                  benchmark_name: str,
                  task_id: str,
                  fidelity: int | None = None) -> None:

    if ("lcbench" in benchmark_name or benchmark_name == "jahs"):
        bench = mfpbench.get(benchmark_name, task_id=task_id, datadir=data_dir)
    else:
        bench = mfpbench.get(benchmark_name, datadir=data_dir)

    if isinstance(bench, mfpbench.tabular.TabularBenchmark):
        pass

    else:
        bench.sample()
        fid_range = bench.fidelity_range
        list(range(fid_range[0], fid_range[1] + 1, fid_range[2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the MF Prior Bench library")
    parser.add_argument("--data_dir", type=str, default=Path("./data"))
    parser.add_argument("--benchmark_suite", type=str, default="lcbench-tabular")
    parser.add_argument("--benchmark_name", type=str, default=None)
    parser.add_argument("--task_id", type=str, default=None)
    parser.add_argument("--fidelity", type=int, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir / args.benchmark_suite

    test_mfpbench(data_dir, args.benchmark_name, args.task_id, args.fidelity)