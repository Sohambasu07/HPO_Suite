from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hpo_glue.benchmark import BenchmarkDescription


BENCHMARKS: dict[str, BenchmarkDescription] = {}

logger = logging.getLogger(__name__)


try:
    from hpo_glue.benchmarks.mfp_bench import mfpbench_benchmarks

    for desc in mfpbench_benchmarks():
        BENCHMARKS[desc.name] = desc

except ImportError:
    logger.info("Could not import mfpbench, skipping benchmarks from mfpbench")


def get_benchmark(name: str) -> BenchmarkDescription:
    """Get a benchmark by name."""
    return BENCHMARKS[name]


__all__ = [
    "get_benchmark",
    "BENCHMARKS",
]
