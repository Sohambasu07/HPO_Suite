from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import yaml

from hpo_glue.benchmarks.benchmarks import get_benchmark
from hpo_glue.glu import GLUE, Experiment, Problem, ProblemReport


def run_exps(
    budget_type: Literal["n_trials", "fidelity_budget", "time_budget"],
    budget: int,
    seed: int | list[int] | None,
    exp_name: str,
    datadir: Path,
    exp_config: Path,
    save_dir: Path,
    num_workers: int = 1,
) -> ProblemReport:
    """Perform GLUE experiments."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running GLUE Experiments")

    # Get the benchmarks
    benchmarks = [
        get_benchmark(
            name="lcbench-tabular",
            task_id="adult",
            datadir=datadir,
        ),
        get_benchmark(
            name="yahpo",
            benchmark_name="lcbench",
            task_id="167184",
            datadir=datadir,
        ),
        get_benchmark(
            name="yahpo",
            benchmark_name="lcbench",
            task_id="3945",
            datadir=datadir,
        ),
        get_benchmark(
            name="yahpo",
            benchmark_name="lcbench",
            task_id="189908",
            datadir=datadir,
        ),
    ]

    problems = []

    # Getting valid ProblemStatements

    with exp_config.open("r") as f:
        config = yaml.safe_load(f)

    seed_list = seed if isinstance(seed, list) else [seed]
    for benchmark in benchmarks:
        for instance in config["optimizer_instances"]:
            for _seed in seed_list:
                Problem(
                    benchmark=benchmark,
                    seed=_seed,
                    budget_type=budget_type,
                    budget=budget,
                    optimizer=eval(config["optimizer_instances"][instance]["optimizer"]),
                    optimizer_hyperparameters=config["optimizer_instances"][instance][
                        "hyperparameters"
                    ],
                    objective=benchmark.default_objective,
                    minimize=benchmark.minimize_default,
                    fidelity_key=benchmark.fidelity_keys,  # defaults to fidelity_keys[0] in case of a list
                )

    # Creating an Experiment
    exp = Experiment(
        name=exp_name,
        problems=problems,
        n_workers=num_workers,
    )

    # Running the Experiment

    glue = GLUE()
    exp_dir = glue.experiment(
        experiment=exp,
        save_dirname=save_dir,
        root_dir=Path(),
    )

    # for report in glu_report:
    #     plot_incumbents(report, save_dir, "test_cross_entropy", 0, report.problem_statement.budget)

    # Report the results
    logger.info("GLUE Experiments complete \n")
    logger.info(f"Results saved at {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GLUE Experiments")
    parser.add_argument("--exp_name", type=str, help="Name of the experiment", default="test")

    parser.add_argument(
        "--budget_type",
        type=str,
        help="Budget types available: n_trials, fidelity_budget, time_budget",
        default="n_trials",
    )

    parser.add_argument("--budget", type=int, default=25)

    parser.add_argument("--seeds", nargs="+", type=int, default=None)

    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--datadir", type=str, default=Path("./data"))

    parser.add_argument("--exp_config", type=str, default=Path("./configs/exp_configs.yaml"))

    parser.add_argument("--save_dir", type=str, default=Path("./results"))
    args = parser.parse_args()

    if isinstance(args.datadir, str):
        args.datadir = Path(args.datadir)
    if isinstance(args.save_dir, str):
        args.save_dir = Path(args.save_dir)

    run_exps(
        budget_type=args.budget_type,
        budget=args.budget,
        seed=args.seeds,
        exp_name=args.exp_name,
        datadir=args.datadir,
        exp_config=args.exp_config,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
    )
