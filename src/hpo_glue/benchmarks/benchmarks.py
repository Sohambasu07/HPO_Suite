from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hpo_glue.benchmark import Benchmark, BenchmarkFactory


BENCHMARK_FACTORIES: dict[str, BenchmarkFactory] = {}

logger = logging.getLogger(__name__)


try:
    from hpo_glue.benchmarks.mfp_bench import _get_benchmark_factories

    BENCHMARK_FACTORIES.update({b.unique_name: b for b in _get_benchmark_factories()})
except ImportError as e:
    raise e
    logger.info("Could not import mfpbench, skipping benchmarks from mfpbench")


def get_benchmark_factory(name: str) -> BenchmarkFactory:
    """Entry point of the repo to get a benchmark."""
    factory = BENCHMARK_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown benchmark {name}")

    return factory


def get_benchmark(name: str, **kwargs: Any) -> Benchmark:
    """Get a benchmark by name."""
    factory = get_benchmark_factory(name)
    return factory(**kwargs)


def all_benchmarks() -> dict[str, BenchmarkFactory]:
    """List all available benchmarks."""
    return dict(BENCHMARK_FACTORIES)


__all__ = [
    "BENCHMARK_FACTORIES",
    "get_benchmark",
    "get_benchmark_factory",
    "benchmarks",
]
