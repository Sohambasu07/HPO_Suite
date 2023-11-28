import logging
from pathlib import Path
import argparse

from hpo_glue.glu import ProblemStatement, GLUE, GLUEReport
from hpo_glue.utils import plot_incumbents
from optimizers.smac import SMAC_Optimizer
from optimizers.random_search import RandomSearch
from optimizers.optuna import OptunaOptimizer
from benchmarks.benchmarks import get_benchmark


def run_exps(budget_type: str,
             budget: int,
             seed: int, 
             datadir: Path, 
             save_dir = Path,
             ) -> GLUEReport:
    """Perform GLUE experiments"""


    print("==========================================================")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running GLUE Experiments")
    
    # Get the benchmark
    tabular_benchmark = get_benchmark(name = "lcbench-tabular",
                              task_id = "adult", 
                              datadir = datadir)
    
    surrogate_benchmark = get_benchmark(name = "yahpo",
                              benchmark_name = "lcbench",
                              task_id = "167184", 
                              datadir = datadir)

    problems = [
        # ProblemStatement(
        # name = "Tabular-BBO",
        # config_space = tabular_benchmark.config_space,
        # fidelity_space = None,
        # result_keys = "test_cross_entropy",
        # budget_type = budget_type,
        # budget = budget,
        # fidelity_keys = "epoch",
        # minimize = True,
        # ),
        # ProblemStatement(
        # name = "Tabular-MultiFidelity",
        # config_space = tabular_benchmark.config_space,
        # fidelity_space = tabular_benchmark.fidelity_space,
        # result_keys = "test_cross_entropy",
        # budget_type = budget_type,
        # budget = budget,
        # fidelity_keys = "epoch",
        # minimize = True,
        # ),
        ProblemStatement(
        name = "Surrogate-BBO",
        config_space = surrogate_benchmark.config_space,
        fidelity_space = None,
        result_keys = "test_cross_entropy",
        budget_type = budget_type,
        budget = budget,
        fidelity_keys = "epoch",
        minimize = True,
        ),
        # ProblemStatement(
        # name = "Surrogate-MultiFidelity",
        # config_space = surrogate_benchmark.config_space,
        # fidelity_space = surrogate_benchmark.fidelity_space,
        # result_keys = "test_cross_entropy",
        # budget_type = budget_type,
        # budget = budget,
        # fidelity_keys = "epoch",
        # minimize = True,
        # )        
    ]

    GLUE.root = Path("./")
    seed = None

    glu_report = []

    for problem in problems:
        # Run the optimizer on the benchmark using GLUE

        # if "Tabular" in problem.name:
        #     bench = tabular_benchmark
        #     optimizer = RandomSearch
        # else:
        #     bench = surrogate_benchmark
        #     optimizer = SMAC_Optimizer
        
        bench = surrogate_benchmark
        optimizer = OptunaOptimizer

        glu_report.append(GLUE.run(problem,
                            optimizer, 
                            bench, 
                            save_dir,
                            seed))
    
    for report in glu_report:
        plot_incumbents(report, save_dir, "test_cross_entropy", 0, report.problem_statement.budget)

    # Report the results
    logger.info("GLUE Experiments complete \n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GLUE Experiments")
    parser.add_argument("--budget_type", type=str, default="n_trials")
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--datadir", type=str, default=Path("./data"))
    parser.add_argument("--save_dir", type=str, default=Path("./results"))
    args = parser.parse_args()

    if isinstance(args.datadir, str):
        args.datadir = Path(args.datadir)
    if isinstance(args.save_dir, str):
        args.save_dir = Path(args.save_dir)

    run_exps(args.budget_type,
             args.budget,
             args.seed, 
             args.datadir, 
             args.save_dir)
