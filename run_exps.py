import logging
from pathlib import Path
import argparse
import inspect

from hpo_glue.glu import ProblemStatement, Problem, Run, Experiment, GLUE, GLUEReport, History
from hpo_glue.utils import plot_incumbents
from optimizers.smac import SMAC_Optimizer, SMAC_BO, SMAC_Hyperband
from optimizers.random_search import RandomSearch
from optimizers.optuna import OptunaOptimizer
from benchmarks.benchmarks import get_benchmark


def run_exps(budget_type: str,
             budget: int,
             seed: int | None, 
             datadir: Path, 
             save_dir = Path,
             num_workers = 1
             ) -> GLUEReport:
    """Perform GLUE experiments"""


    print("==========================================================")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running GLUE Experiments")

    # Set GLUE Root path

    GLUE.root = Path("./")
    
    # Get the benchmarks
    
    benchmarks = [
        get_benchmark(name = "lcbench-tabular",
                      task_id = "adult", 
                      datadir = datadir),
        get_benchmark(name = "yahpo",
                      benchmark_name = "lcbench",
                      task_id = "167184", 
                      datadir = datadir)
    ]

    optimizers = [
        RandomSearch,
        SMAC_Optimizer,
        SMAC_BO,
        SMAC_Hyperband
    ]

    optimizer_kwargs = {
        "RandomSearch": {},
        "SMAC_Optimizer": {},
        "SMAC_BO": {"xi": 0.01},
        "SMAC_Hyperband": {"eta": 3}
        }

    problems = []

    # Getting valid ProblemStatements

    for benchmark in benchmarks:
        for optimizer in optimizers:
            if GLUE.sanity_checks(
                optimizer = optimizer,
                benchmark = benchmark
            ):
                problem_statement = ProblemStatement(
                    optimizer = optimizer,
                    benchmark = benchmark
                )

                problem = Problem(
                    problem_statement = problem_statement,
                    objectives = benchmark.default_objective,
                    minimize = benchmark.minimize_default
                )

                if isinstance(benchmark.fidelity_keys, list):
                    problem.fidelities = benchmark.fidelity_keys[0]
                else:
                    problem.fidelities = benchmark.fidelity_keys


                problems.append(problem)

    # Creating a Run
                
    run = Run(
        budget_type = budget_type,
        budget = budget,
        seed = seed,
        problems = problems
    )

    # Creating an Experiment

    exp = Experiment(
        run = run,
        n_workers = num_workers
    )

    # Running the Experiment

    GLUE.experiment(
        experiment = exp,
        save_dir = save_dir,
        optimizer_kwargs = optimizer_kwargs
    )
    
    # for report in glu_report:
    #     plot_incumbents(report, save_dir, "test_cross_entropy", 0, report.problem_statement.budget)

    # Report the results
    logger.info("GLUE Experiments complete \n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GLUE Experiments")
    parser.add_argument("--budget_type", 
                        type=str,
                        help="Budget types available: n_trials, fidelity_budget, time_budget",
                        default="n_trials")
    
    parser.add_argument("--budget", 
                        type=int, 
                        default=25)
    
    parser.add_argument("--seed", 
                        type=int, 
                        default=None)
    
    parser.add_argument("--num_workers", 
                        type=int, 
                        default=1)
    
    parser.add_argument("--datadir", 
                        type=str, 
                        default=Path("./data"))
    
    parser.add_argument("--save_dir", 
                        type=str, 
                        default=Path("./results"))
    args = parser.parse_args()

    if isinstance(args.datadir, str):
        args.datadir = Path(args.datadir)
    if isinstance(args.save_dir, str):
        args.save_dir = Path(args.save_dir)

    run_exps(args.budget_type,
             args.budget,
             args.seed, 
             args.datadir, 
             args.save_dir,
             args.num_workers)
