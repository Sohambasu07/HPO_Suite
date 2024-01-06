from __future__ import annotations

from typing import Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from hpo_glue.glu import Benchmark
from benchmarks.mfp_bench import lcbench_tabular, yahpo_surrogate_benchmark

BENCHMARK_FACTORIES = {
        "lcbench-tabular": lcbench_tabular,
        "yahpo": yahpo_surrogate_benchmark,
    }

def get_benchmark(name: str, **kwargs) -> Benchmark:
    """Entry point of the repo to get a benchmark."""
    # This will probably look like a big if statement
    # or do some introspection on Benchmark.__subclasses__
    
    factory = BENCHMARK_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown benchmark {name}")

    return factory(**kwargs)

def get_all_benchmarks() -> Dict[str, Benchmark]:
    """Get all benchmarks."""

    return BENCHMARK_FACTORIES