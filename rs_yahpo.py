"""Script to perform HPO using Random Search on LCBench Tabular Benchmark"""

import logging
from hpo_glue.glu import GLUE, GLUEReport
from optimizers.random_search import RandomSearch
from pathlib import Path
from benchmarks.benchmarks import get_benchmark
import argparse


def rs_yahpo(n_trials: int, 
             benchmark_name: str, 
             task_id: str,
             seed: int, 
             datadir: Path, 
             save_dir = Path,
             ) -> GLUEReport:
    """Perform Random Search and query LCBench from Yahpo-Gym using mfpbench"""


    print("==========================================================")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running Random Search on LCBench from Yahpo-Gym using"
                "MF Prior Bench\n")
    
    # Get the benchmark
    benchmark = get_benchmark(name = "yahpo",
                              benchmark_name = benchmark_name, 
                              task_id = task_id, 
                              datadir = datadir)
    
    # Get the optimizer
    optimizer = RandomSearch

    # Run the optimizer on the benchmark using GLUE
    glu_report = GLUE.run(optimizer, 
                          benchmark, 
                          n_trials,
                          save_dir,
                          seed)
    
    # Report the results
    logger.info("Random Search on Yahpo-Gym LCBench complete \n")
    return glu_report
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform HPO using Random "
                                     "Search on Yahpo-Gym LCBench Benchmark")
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--benchmark_name", type=str, default="lcbench")
    parser.add_argument("--task_id", type=str, default="167184")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--datadir", type=str, default=Path("./data"))
    parser.add_argument("--save_dir", type=str, default=Path("./results"))
    args = parser.parse_args()

    report = rs_yahpo(args.n_trials, 
                      args.benchmark_name,
                      args.task_id,
                      args.seed, 
                      args.datadir, 
                      args.save_dir)