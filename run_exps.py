import logging
from pathlib import Path
import argparse
import yaml

from hpo_glue.glu import ProblemStatement, Problem, Run, Experiment, GLUE, GLUEReport
from hpo_glue.utils import plot_incumbents
from optimizers.smac import SMAC_Optimizer, SMAC_BO, SMAC_Hyperband
from optimizers.random_search import RandomSearch
from optimizers.optuna import OptunaOptimizer
from benchmarks.benchmarks import get_benchmark


def run_exps(budget_type: str,
             budget: int,
             seed: int | None, 
             datadir: Path, 
             exp_config: Path,
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

    problems = []

    # Getting valid ProblemStatements

    with open(exp_config, "r") as f:
        config = yaml.safe_load(f)

    for benchmark in benchmarks:
        for instance in config["optimizer_instances"]:
            problem_statement = ProblemStatement(
                optimizer = eval(config["optimizer_instances"][instance]["optimizer"]),
                benchmark = benchmark,
                hyperparameters = config["optimizer_instances"][instance]["hyperparameters"],
            )
            problem = Problem(
                problem_statement = problem_statement,
                objectives = benchmark.default_objective,
                minimize = benchmark.minimize_default,
                fidelities = benchmark.fidelity_keys # defaults to fidelity_keys[0] in case of a list
            )
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
        runs = [run],
        n_workers = num_workers
    )

    # Running the Experiment

    GLUE.experiment(
        experiment = exp,
        save_dir = save_dir,
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

    parser.add_argument("--exp_config",
                        type=str,
                        default=Path("./configs/exp_configs.yaml"))
    
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
             args.exp_config,
             args.save_dir,
             args.num_workers)
