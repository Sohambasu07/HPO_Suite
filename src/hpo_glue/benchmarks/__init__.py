from __future__ import annotations

from typing import TYPE_CHECKING

from hpo_glue.benchmarks.mfp_bench import mfpbench_benchmarks
from hpo_glue.benchmarks.pymoo import pymoo_benchmarks

if TYPE_CHECKING:
    from hpo_glue.benchmark import BenchmarkDescription

BENCHMARKS: dict[str, BenchmarkDescription] = {}
for desc in mfpbench_benchmarks():
    BENCHMARKS[desc.name] = desc
for desc in pymoo_benchmarks():
    BENCHMARKS[desc.name] = desc

__all__ = [
    "BENCHMARKS",
]
