from typing import Dict, Any
from hpo_glue.glu import Benchmark
from lcbench import LCBenchTabular, LCBenchSetup, lcbench_tabular

def get_benchmark(name: str, **kwargs) -> Benchmark:
    """Entry point of the repo to get a benchmark."""
    # This will probably look like a big if statement
    # or do some introspection on Benchmark.__subclasses__
    BENCHMARK_FACTORIES = {
        "lcbench-tabular": lcbench_tabular,
        # "yahpo": yahpo_surrogate_benchmark,
    }
    factory = BENCHMARK_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown benchmark {name}")

    return factory(**kwargs)