"""Script to perform HPO using Random Search on LCBench Tabular Benchmark"""

import logging
import os
import pandas as pd
from hpo_glue.glu import *
from optimizers.random_search import RandomSearch
from pathlib import Path
from benchmarks.benchmarks import get_benchmark
import argparse


def rs_lcbench(n_trials = 1, 
               task_id = "adult", 
               datadir = Path("./data")
               ) -> GLUEReport:
    """Perform HPO using Random Search on LCBench Tabular Benchmark"""


    print("==========================================================")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("\nRunning Random Search on LCBench Tabular Benchmark "
                "from MF Prior Bench\n")
    
    # Get the benchmark
    benchmark = get_benchmark(name = "lcbench-tabular", 
                              task_id = task_id, 
                              datadir = datadir / "lcbench-tabular")
    
    # Get the optimizer
    optimizer = RandomSearch

    # Run the optimizer on the benchmark using GLUE
    glu_report = GLUE.run(optimizer, benchmark, n_trials, is_tabular=True)
    
    # Report the results
    logger.info("Random Search on LCBench Tabular Benchmark complete \n")
    return glu_report
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform HPO using Random "
                                     "Search on LCBench Tabular Benchmark")
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--task_id", type=str, default="adult")
    parser.add_argument("--datadir", type=str, default=Path("./data"))
    args = parser.parse_args()

    report = rs_lcbench(args.n_trials, args.task_id, args.datadir)

    print(report.optimizer_name)
    print(report.benchmark_name)

    for rep in report.history:
        print("Query Config id: ", rep.query.config.id)
        print("Query Config: ", rep.query.config.values)
        print("Query Fidelity: ", rep.query.fidelity)
        print("Result: ", rep.result)
