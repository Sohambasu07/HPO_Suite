"""Script to perform HPO using Random Search on LCBench Tabular Benchmark"""

import logging
from hpo_glue.glu import GLUE, GLUEReport
from optimizers.random_search import RandomSearch
from pathlib import Path
from benchmarks.benchmarks import get_benchmark
import argparse


def rs_lcbench(n_trials = 1, 
               task_id = "adult", 
               datadir = Path("./data"),
               save_dir = Path("./results")
               ) -> GLUEReport:
    """Perform HPO using Random Search on LCBench Tabular Benchmark"""


    print("==========================================================")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running Random Search on LCBench Tabular Benchmark "
                "from MF Prior Bench\n")
    
    # Get the benchmark
    benchmark = get_benchmark(name = "lcbench-tabular", 
                              task_id = task_id, 
                              datadir = datadir / "lcbench-tabular")
    
    # Get the optimizer
    optimizer = RandomSearch

    # Run the optimizer on the benchmark using GLUE
    glu_report = GLUE.run(optimizer, 
                          benchmark, 
                          n_trials,
                          save_dir = save_dir)
    
    # Report the results
    logger.info("Random Search on LCBench Tabular Benchmark complete \n")
    return glu_report
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform HPO using Random "
                                     "Search on LCBench Tabular Benchmark")
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--task_id", type=str, default="adult")
    parser.add_argument("--datadir", type=str, default=Path("./data"))
    parser.add_argument("--save_dir", type=str, default=Path("./results"))
    args = parser.parse_args()

    report = rs_lcbench(args.n_trials, 
                        args.task_id, 
                        args.datadir, 
                        args.save_dir)
