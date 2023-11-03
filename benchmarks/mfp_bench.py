"""Script to interface with the MF Prior Bench library"""

import mfpbench
from pathlib import Path
import pandas as pd

from hpo_glue.glu import TabularBenchmark
from hpo_glue.glu import Query, Result

def lcbench_tabular(task_id: str, datadir: Path) -> TabularBenchmark:

    data_dir = datadir
    table = mfpbench.get("lcbench_tabular", task_id=task_id, datadir=data_dir)
    table_for_task = table.table
    return TabularBenchmark(
        name=f"lcbench_tabular_{task_id}",
        table=table_for_task,
        id_key="id",  # Key in the table to uniquely identify tasks
        config_keys=list(table_for_task.keys().values[7:]),  # Keys in the table that correspond to configs
        result_keys=list(table_for_task.keys().values[:7]),  # Key in the table that corresponds to the result
        fidelity_key="epoch",  # Key in the table that corresponds to the fidelity (e.g. "epoch")
    )

# def query_mfpbench(table: pd.DataFrame, query: Query) -> Result:
#     #Query the benchmark for a result
#     result = 
#     return Result(query, result