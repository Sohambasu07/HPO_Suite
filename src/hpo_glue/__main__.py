import argparse
from pathlib import Path

from hpo_glue.study import create_study


def glue_study(  # noqa: D103
    optimizers: str,
    benchmarks: str,
    seeds: list,
    num_seeds: int,
    budget: int,
    precision: int,
    exp_name: str,
    results_dir: Path,
    overwrite: bool = False,  # noqa: FBT001, FBT002
    continuations: bool = False,  # noqa: FBT001, FBT002
    exec_type: str = "sequential",
    group_by: str = None,
):
    study = create_study(
        results_dir=results_dir,
        name=exp_name,
    )
    study.optimize(
        optimizers=optimizers,
        benchmarks=benchmarks,
        seeds=seeds,
        num_seeds=num_seeds,
        budget=budget,
        precision=precision,
        overwrite=overwrite,
        continuations=continuations,
        exec_type=exec_type,
        group_by=group_by,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", "-e",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--results_dir", "-r",
        type=Path,
        default="hpo-glue-output",
        help="Results directory",
    )
    parser.add_argument(
        "--optimizers", "-o",
        nargs="+",
        type=str,
        required=True,
        help="Optimizer to use",
    )
    parser.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        type=str,
        required=True,
        help="Benchmark to use",
    )
    parser.add_argument(
        "--seeds", "-s",
        nargs="+",
        type=int,
        default=None,
        help="Seed(s) to use",
    )
    parser.add_argument(
        "--num_seeds", "-n",
        type=int,
        default=1,
        help="Number of seeds to be generated. "
        "Only used if seeds is not provided",
    )
    parser.add_argument(
        "--budget", "-bgt",
        type=int,
        default=50,
        help="Budget to use",
    )
    parser.add_argument(
        "--overwrite", "-ow",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--continuations", "-c",
        action="store_true",
        help="Use continuations",
    )
    parser.add_argument(
        "--precision", "-p",
        type=int,
        help="Precision to use",
    )
    parser.add_argument(
        "--exec_type", "-x",
        type=str,
        default="dump",
        choices=["sequential", "parallel", "dump"],
        help="Execution type",
    )
    parser.add_argument(
        "--group_by", "-g",
        type=str,
        default=None,
        choices=["optimizer", "benchmark", "seed", "memory"],
        help="Runs dump group by\n"
        "Only used if exec_type is dump"
    )
    args = parser.parse_args()

    glue_study(
        optimizers=args.optimizers,
        benchmarks=args.benchmarks,
        seeds=args.seeds,
        num_seeds=args.num_seeds,
        budget=args.budget,
        precision=args.precision,
        exp_name=args.exp_name,
        results_dir = args.results_dir,
        overwrite=args.overwrite,
        continuations=args.continuations,
        exec_type=args.exec_type,
        group_by=args.group_by,
    )

