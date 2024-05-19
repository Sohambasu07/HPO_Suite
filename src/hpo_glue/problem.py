from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from hpo_glue.benchmark import BenchmarkDescription, SurrogateBenchmark, TabularBenchmark
from hpo_glue.history import History

if TYPE_CHECKING:
    from hpo_glue.budget import BudgetType
    from hpo_glue.fidelity import Fidelity
    from hpo_glue.measure import Measure
    from hpo_glue.optimizer import Optimizer

logger = logging.getLogger(__name__)


class ProblemState(str, Enum):
    """The state of a problem."""

    pending = "pending"
    running = "running"
    crashed = "crashed"
    finished = "finished"


@dataclass
class Problem:
    """A problem to optimize over."""

    objective: tuple[str, Measure] | Mapping[str, Measure]
    """The metrics to optimize for this problem, with a specific order.

    If only one metric is specified, this is considered single objective and
    not multiobjective.
    """

    fidelity: tuple[str, Fidelity] | Mapping[str, Fidelity] | None
    """Fidelities to use from the Benchmark.

    When `None`, the problem is considered a black-box problem with no fidelity.

    When a single fidelity is specified, the problem is considered a _multi-fidelity_ problem.

    When many fidelities are specified, the problem is considered a _many-fidelity_ problem.
    """

    cost: tuple[str, Measure] | Mapping[str, Measure] | None
    """The cost metric to use for this problem.

    When `None`, the problem is considered a black-box problem with no cost.

    When a single cost is specified, the problem is considered a _cost-sensitive_ problem.

    When many costs are specified, the problem is considered a _multi-cost_ problem.
    """

    benchmark: BenchmarkDescription
    """The benchmark to use for this problem statement"""

    seed: int
    """The seed to use."""

    budget: BudgetType
    """The type of budget to use for the optimizer."""

    optimizer: type[Optimizer]
    """The optimizer to use for this problem statement"""

    optimizer_hyperparameters: dict[str, Any] = field(default_factory=dict)
    """The hyperparameters to use for the optimizer"""

    optimizer_output_dir: Path = field(default=Path("optimizer_cache"))
    """Path for optimizer output"""

    name: str = field(init=False)
    """The name of the problem statement.

    This is used to identify the problem statement in the results and in the filesystem.
    """

    is_tabular: bool = field(init=False)
    """Whether the benchmark is tabular"""

    is_multiobjective: bool = field(init=False)
    """Whether the problem has multiple objectives"""

    is_multifidelity: bool = field(init=False)
    """Whether the problem has a fidelity parameter"""

    is_manyfidelity: bool = field(init=False)
    """Whether the problem has many fidelities"""

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
        self.is_tabular: bool
        match self.benchmark:
            case TabularBenchmark():
                self.is_tabular = True
            case SurrogateBenchmark():
                self.is_tabular = False
            case _:
                raise TypeError("Benchmark must be a TabularBenchmark or SurrogateBenchmark")

        self.is_manyfidelity: bool
        self.is_multifidelity: bool
        match self.fidelity:
            case None:
                self.is_multifidelity = False
                self.is_manyfidelity = False
            case tuple():
                self.is_multifidelity = True
                self.is_manyfidelity = False
            case Mapping():
                if len(self.fidelity) == 1:
                    raise ValueError("Single fidelity should be a tuple, not a mapping")

                self.is_multifidelity = False
                self.is_manyfidelity = True
            case _:
                raise TypeError("Fidelity must be a tuple (name, fidelity) or a mapping")

        self.is_multiobjective: bool
        objective_str: str
        match self.objective:
            case tuple():
                self.is_multiobjective = False
                objective_str = self.objective[0]
            case Mapping():
                if len(self.objective) == 1:
                    raise ValueError("Single objective should be a tuple, not a mapping")

                self.is_multiobjective = True
                objective_str = ",".join(self.objective.keys())
            case _:
                raise TypeError("Objective must be a tuple (name, measure) or a mapping")

        hp_str = (
            ",".join(f"{k}={v}" for k, v in self.optimizer_hyperparameters.items())
            if any(self.optimizer_hyperparameters)
            else "default"
        )
        name_parts = {
            "benchmark": self.benchmark.name,
            "optimizer": self.optimizer.name,
            "seed": self.seed,
            "budget": self.budget.path_str,
            "objective": objective_str,
            "hp_kwargs": hp_str,
        }
        self.name = "_".join(f"{k}={v}" for k, v in name_parts.items())
        self.path = Path(*[f"{k}={v}" for k, v in name_parts.items()])

        self.cache_dir = self.optimizer_output_dir / self.relative_optimizer_path
        self.started_flag_path = (
            self.optimizer_output_dir / self.relative_optimizer_path / "started.flag"
        )
        self.crashed_flag_path = (
            self.optimizer_output_dir / self.relative_optimizer_path / "crashed.flag"
        )
        self.success_flag_path = (
            self.optimizer_output_dir / self.relative_optimizer_path / "success.flag"
        )

        self.optimizer.support.check_opt_support(who=self.optimizer.name, problem=self)

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

    @dataclass(kw_only=True)
    class Support:
        """The support of an optimizer for a problem."""

        objectives: tuple[Literal["single", "many"], ...]
        fidelities: tuple[Literal[None, "single", "many"], ...]
        cost_awareness: tuple[Literal[None, "single", "many"], ...]
        tabular: bool

        def check_opt_support(self, who: str, *, problem: Problem) -> None:
            """Check if the problem is supported by the support."""
            match problem.fidelity:
                case None if None not in self.fidelities:
                    raise ValueError(
                        f"Optimizer {who} does not support having no fidelties for {problem.name}!"
                    )
                case tuple() if "single" not in self.fidelities:
                    raise ValueError(
                        f"Optimizer {who} does not support multi-fidelty for {problem.name}!"
                    )
                case Mapping() if "many" not in self.fidelities:
                    raise ValueError(
                        f"Optimizer {who} does not support many-fidelty for {problem.name}!"
                    )

            match problem.objective:
                case tuple() if "single" not in self.objectives:
                    raise ValueError(
                        f"Optimizer {who} does not support single-objective for {problem.name}!"
                    )
                case Mapping() if "many" not in self.objectives:
                    raise ValueError(
                        f"Optimizer {who} does not support multi-objective for {problem.name}!"
                    )

            match problem.cost:
                case None if None not in self.cost_awareness:
                    raise ValueError(
                        f"Optimizer {who} does not support having no cost for {problem.name}!"
                    )
                case tuple() if "single" not in self.cost_awareness:
                    raise ValueError(
                        f"Optimizer {who} does not support single-cost for {problem.name}!"
                    )
                case Mapping() if "many" not in self.cost_awareness:
                    raise ValueError(
                        f"Optimizer {who} does not support multi-cost for {problem.name}!"
                    )

            match problem.is_tabular:
                case True if not self.tabular:
                    raise ValueError(
                        f"Optimizer {who} does not support tabular benchmarks for {problem.name}!"
                    )

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
