from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from hpo_glue.benchmarks import BENCHMARKS
from hpo_glue.optimizers.dehb import DEHB_Optimizer
from hpo_glue.optimizers.smac import SMAC_Hyperband
from hpo_glue.run import Run

logging.basicConfig(level=logging.DEBUG)

smac_logger = logging.getLogger("smac")
smac_logger.setLevel(logging.ERROR)

THIS_FILE = Path(__file__).absolute().resolve()


def experiments(expdir: Path, num_seeds: int) -> list[Run]:
    return Run.generate(
        expdir=expdir,
        optimizers=[
            DEHB_Optimizer,
            # SMAC_Hyperband,
            # (SMAC_Hyperband, {"eta": 2}),
        ],
        benchmarks=[
            BENCHMARKS["mfh3_good"],
            # BENCHMARKS["mfh6_good"],
        ],
        # seeds=[1],
        num_seeds=num_seeds,
        budget=50,
        objectives=1,
        fidelities=1,
        on_error="ignore",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuations", action="store_true")
    parser.add_argument("--precision", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_seeds", type=int, default=1)
    subparsers = parser.add_subparsers(dest="command")

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("--expdir", type=Path, default="hpo-glue-output")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--expdir", type=Path, default="hpo-glue-output")
    run_parser.add_argument("--isolated", action="store_true")

    yaml_launch_parser = subparsers.add_parser("from-yaml")
    yaml_launch_parser.add_argument("--path", type=Path, required=True)

    args = parser.parse_args()
    glue_path = Path.cwd()
    match args.command:
        case "setup":
            expdir = Path(args.expdir)
            for run in experiments(expdir, num_seeds=args.num_seeds):
                run.create_env(hpo_glue=f"-e {Path.cwd()}")
                run.write_yaml()
        case "run":
            expdir = Path(args.expdir)

            if args.isolated:
                for run in experiments(expdir, num_seeds=args.num_seeds):
                    script = str(THIS_FILE)
                    subprocess.run(
                        [run.venv.python, script, "from-yaml", "--path", str(run.run_yaml_path)],  # noqa: S603
                        check=True,
                    )
            else:
                for run in experiments(expdir, num_seeds=args.num_seeds):
                    run.run(
                        overwrite=args.overwrite, 
                        progress_bar=True,
                        continuations=args.continuations,
                        precision=args.precision
                    )

        case "from-yaml":
            exp = Run.from_yaml(args.path)
            exp.run(overwrite=True, progress_bar=False)
        case _:
            parser.print_help()
            sys.exit(1)


"""
experiment_report = experiment.run(overwrite=True, n_jobs=4)
for _problem, report in experiment_report.groupby_problem():
    _df = report.df(incumbent_trajectory=True)

    print(_problem.name)  # noqa: T201
    print(_df)  # noqa: T201
"""
