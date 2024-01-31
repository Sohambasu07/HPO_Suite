"""Script to interface with the MF Prior Bench library"""

import mfpbench
from pathlib import Path
from typing import TYPE_CHECKING

from hpo_glue.glu import TabularBenchmark, SurrogateBenchmark, Query

def lcbench_tabular(
        task_id: str, 
        datadir: Path | str
) -> TabularBenchmark:
    
    if isinstance(datadir, str):
        datadir = Path(datadir)

    datadir = datadir / "lcbench-tabular"
    table = mfpbench.get("lcbench_tabular", task_id=task_id, datadir=datadir)
    table_for_task = table.table
    return TabularBenchmark(
        name = f"lcbench_tabular_{task_id}",
        table = table_for_task,
        id_key = "id",  # Key in the table to uniquely identify tasks
        config_keys = table.config_keys,  # Keys in the table that correspond to configs
        result_keys = table.result_keys,  # Key in the table that corresponds to the result
        fidelity_keys = table.fidelity_key,  # Key in the table that corresponds to the fidelity (e.g. "epoch")
        time_budget = "time",  # Time budget key
        default_objective = "test_cross_entropy",  # Default objective to optimize
        minimize_default = True,  # Whether the default objective should be minimized
    )

def yahpo_surrogate_benchmark(
        benchmark_name: str, 
        task_id: str, 
        datadir: Path
) -> SurrogateBenchmark:
    
    if isinstance(datadir, str):
        datadir = Path(datadir)

    datadir = datadir / "yahpo"
    bench = mfpbench.get(benchmark_name, task_id=task_id, datadir=datadir)
    fid_range = bench.fidelity_range
    fidelity_space = list(range(fid_range[0], fid_range[1] + 1, fid_range[2]))
    return SurrogateBenchmark(
        name="yahpo" + "_" + benchmark_name + "_" + task_id,
        config_space = bench.space,
        fidelity_space = fidelity_space,
        result_keys = dir(bench.Result),
        fidelity_keys = bench.fidelity_name,
        query_function = yahpo_query_function,
        benchmark = bench,
        time_budget = "time",
        default_objective = "test_cross_entropy", # Default objective to optimize
        minimize_default = True, # Whether the default objective should be minimized
    )

def yahpo_query_function(benchmark, query: Query):
    q = benchmark.query(query.config.values, at=query.fidelity)
    return q.dict()