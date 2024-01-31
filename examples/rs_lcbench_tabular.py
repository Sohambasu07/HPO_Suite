"""Script to perform HPO using Random Search on LCBench Tabular Benchmark"""

import logging
from hpo_glue.glu import GLUE, GLUEReport, ProblemStatement
from optimizers.random_search import RandomSearch
from pathlib import Path
from benchmarks.benchmarks import get_benchmark
import argparse


def rs_lcbench(budget_type: str,
               budget: int, 
               task_id,
               seed,
               datadir,
               save_dir
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
                              datadir = datadir)
    
    # Get the optimizer
    optimizer = RandomSearch

    ps = ProblemStatement(
        name = "Random_Search_on_LCBench_Tabular",
        config_space = benchmark.config_space,
        fidelity_space = benchmark.fidelity_space,
        result_keys = "test_cross_entropy",
        budget_type = budget_type,
        budget = budget,
        minimize = True,
    )

    # Run the optimizer on the benchmark using GLUE
    glu_report = GLUE.run(problem_statement=ps,
                          optimizer=optimizer, 
                          benchmark=benchmark,
                          save_dir=save_dir,
                          seed=seed)
    
    # Report the results
    logger.info("Random Search on LCBench Tabular Benchmark complete \n")
    return glu_report
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform HPO using Random "
                                     "Search on LCBench Tabular Benchmark")
    
    parser.add_argument("--budget_type", type=str, default="n_trials")
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--task_id", type=str, default="adult")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--datadir", type=str, default=Path("./data"))
    parser.add_argument("--save_dir", type=str, default=Path("./results"))
    args = parser.parse_args()

    report = rs_lcbench(args.budget_type,
                        args.budget, 
                        args.task_id,
                        args.seed, 
                        args.datadir, 
                        args.save_dir)
