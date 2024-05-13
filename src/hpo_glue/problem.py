from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace


class ProblemState(str, Enum):
    pending = "pending"
    running = "running"
    crashed = "crashed"
    finished = "finished"


@dataclass
class Problem:
    """A problem to optimize over."""

    objective: str | list[str]
    """The key(s) in the result that we want to consider as the objective value

    * str -> single objective
    * list[str] -> multi-objective
    """

    # TODO(soham): For testing purposes. Will be replaced by Metrics inside Benchmark object
    minimize: bool | list[bool]
    """Whether to minimize or maximize the objective value. One per objective"""

    fidelity_key: str | list[str] | None
    """The key(s) in the result that we want to consider as the fidelity

    * str -> single fidelity parameter
    * list[str] -> many fidelity parameters
    * None -> no fidelity
    """

    seed: int | None
    """The seed to use."""

    budget: int | float
    """The budget to run the optimizer for"""

    budget_type: Literal["n_trials", "time_budget", "fidelity_budget"]
    """The type of budget to use for the optimizer"""

    optimizer: type[Optimizer]
    """The optimizer to use for this problem statement"""

    benchmark: Callable[[], Benchmark]
    """The benchmark to use for this problem statement"""

    optimizer_hyperparameters: dict[str, Any]
    """The hyperparameters to use for the optimizer"""

    root_dir: Path = field(default=Path("optimizer_cache"))
    """Path for optimizer output"""

    name: str = field(init=False)
    """The name of the problem statement. This is used to identify the problem statement
    in the results and in the filesystem"""

    is_tabular: bool = field(init=False)
    """Whether the benchmark is tabular"""

    is_multiobjective: bool = field(init=False)
    """Whether the problem has multiple objectives"""

    is_multifidelity: bool = field(init=False)
    """Whether the problem has a fidelity parameter"""

    is_manyfidelity: bool = field(init=False)
    """Whether the problem has many fidelities"""

    config_space: list[Config] | ConfigurationSpace = field(init=False)
    """The configuration space to optimize over for the benchmark"""

    fidelity_space: list[int] | list[float] | None = field(init=False)
    """The fidelity space for the benchmark"""

    relative_optimizer_path: Path = field(init=False)
    """The unique path for this problem statement"""

    started_flag_path: Path = field(init=False)
    """The path to the started flag"""

    crashed_flag_path: Path = field(init=False)
    """The path to the crashed flag"""

    success_flag_path: Path = field(init=False)
    """The path to the finished flag"""

    cache_dir: Path = field(init=False)
    """The path to the optimizer cache"""

    def __post_init__(self):
        if isinstance(self.fidelity_key, list):
            raise NotImplementedError("Many fidelities not yet supported")

        if isinstance(self.objective, list):
            raise NotImplementedError("Multiobjective not yet supported")

        if self.budget_type not in ("n_trials", "time_budget", "fidelity_budget"):
            raise ValueError(
                "Invalid budget type!"
                f" '{self.budget_type}' not in ('n_trials', 'time_budget', 'fidelity_budget')",
            )

        if self.budget_type == "fidelity_budget" and self.fidelity_key is None:
            raise ValueError("Fidelity budget specified but no fidelities in proble!")

        if self.budget_type == "time_budget" and self.benchmark.time_budget is None:
            raise ValueError("Time budget specified but no time budget in benchmark!")

        self.config_space = self.benchmark.config_space
        self.fidelity_space = self.benchmark.fidelity_space

        self.is_tabular = isinstance(self.benchmark, TabularBenchmark)
        self.is_multifidelity = isinstance(self.fidelity_key, str)
        self.is_manyfidelity = isinstance(self.fidelity_key, list)
        self.is_multiobjective = isinstance(self.objective, list)

        self.cache_dir = self.root_dir / self.relative_optimizer_path
        self.started_flag_path = self.root_dir / self.relative_optimizer_path / "started.flag"
        self.crashed_flag_path = self.root_dir / self.relative_optimizer_path / "crashed.flag"
        self.success_flag_path = self.root_dir / self.relative_optimizer_path / "success.flag"

        objective_str = (
            self.objective if isinstance(self.objective, str) else ",".join(self.objective)
        )
        hp_str = (
            ",".join(f"{k}={v}" for k, v in self.optimizer_hyperparameters.items())
            if any(self.optimizer_hyperparameters)
            else "default"
        )
        name_parts = {
            "benchmark": self.benchmark.name,
            "optimizer": self.optimizer.name,
            "seed": self.seed,
            "budget_type": self.budget_type,
            "budget": self.budget,
            "objective": objective_str,
            "hp_kwargs": hp_str,
        }
        self.name = "_".join(f"{k}={v}" for k, v in name_parts.items())
        self.path = Path(*[f"{k}={v}" for k, v in name_parts.items()])

        if self.is_tabular and not self.optimizer.supports_tabular:
            raise ValueError(
                f"{self.optimizer.name} does not support tabular benchmarks! "
                f"{self.optimizer.name} and {self.benchmark.name} are not compatible.",
            )

        if self.is_multifidelity and not self.optimizer.supports_multifidelity:
            raise ValueError(
                f"{self.optimizer.name} does not support multi-fidelity but the problem"
                f" was specified with multi fidelity!\n{self}"
            )

        if self.is_manyfidelity and not self.optimizer.supports_manyfidelity:
            raise ValueError(
                f"{self.optimizer.name} does not support many-fidelity but the problem"
                f" was specified with many fidelities!\n{self}"
            )

        if self.is_multiobjective and not self.optimizer.supports_multiobjective:
            raise ValueError(
                f"{self.optimizer.name} does not support multi-objective but the problem"
                f" was specified with multi objectives!\n{self}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the Problem as a dictionary."""
        return asdict(self)

    def state(self) -> ProblemState:
        """Return the state of the problem."""
        if self.crashed_flag_path.exists():
            return ProblemState.crashed

        if self.success_flag_path.exists():
            return ProblemState.finished

        if self.started_flag_path.exists():
            return ProblemState.running

        return ProblemState.pending

    def run(self, *, overwrite: bool = False) -> Problem.Report:
        """Run the problem."""
        if overwrite:
            logger.info(f"Overwriting {self.name} as `overwrite=True` was set.")
            if self.cache_dir.exists():
                try:
                    shutil.rmtree(self.cache_dir)
                except Exception as e:
                    logger.exception(e)
                    logger.error(f"Error deleting {self.cache_dir}: {e}")
        elif self.cache_dir.exists():
            raise FileExistsError(
                f"Optimizer cache already exists at {self.cache_dir}."
                " Set `overwrite=True` to overwrite.",
            )

        from hpo_glue._run import run_problem

        return run_problem.run(problem=self)

    @dataclass
    class Report:
        """The report of a GLUE run."""

        problem: Problem
        history: History
        path: Path = field(init=False)

        def __post_init__(self):
            self.path = self.problem.path

        def df(self) -> pd.DataFrame:
            """Return the history as a pandas DataFrame."""
            # Fill in the fields of the problem as constant columns
            return self.history.df().assign(
                **{f"problem.{k}": v for k, v in asdict(self.problem).items()}
            )

        @classmethod
        def from_path(cls, path: Path) -> Problem.Report:
            """Load a GLUEReport from a path."""
            df = pd.read_parquet(path)  # noqa: PD901
            problem_cols = [col for col in df.columns if col.startswith("problem.")]
            other_cols = [col for col in df.columns if not col.startswith("problem.")]

            problem_df = df[problem_cols].drop_duplicates()
            if len(problem_df) != 1:
                raise ValueError(
                    f"Expected exactly one problem in {path}, got {len(problem_df)}"
                    f"\n{problem_df}"
                )
            problem_dict = problem_df.to_series().to_dict()

            # TODO(eddiebergman): This won't fix optimizer hyperparameters or list types...
            problem = Problem(**{k.replace("problem.", ""): v for k, v in problem_dict.items()})
            history = History.from_df(df[other_cols])
            return cls(problem=problem, history=history)
