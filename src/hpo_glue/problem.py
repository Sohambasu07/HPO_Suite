from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias
from typing_extensions import Self

import pandas as pd

from hpo_glue.benchmark import BenchmarkDescription, SurrogateBenchmark, TabularBenchmark
from hpo_glue.history import History

if TYPE_CHECKING:
    from hpo_glue.fidelity import Fidelity
    from hpo_glue.measure import Measure
    from hpo_glue.optimizer import Optimizer
    from hpo_glue.result import Result

logger = logging.getLogger(__name__)


class ProblemState(str, Enum):
    """The state of a problem."""

    pending = "pending"
    running = "running"
    crashed = "crashed"
    finished = "finished"


@dataclass
class TrialBudget:
    """A budget for the number of trials to run."""

    budget: int | float
    """Total amount of budget allowed for the optimizer for this problem.

    How this is interpreted is depending on fidelity type.

    If the problem **does not** include a fidelity, then this is assumed
    to be a black-box problem, and each fully complete trial counts as
    1 towards the budget.

    If the problem **does** include a **single** fidelity, then the fidelity
    at which the trial was evaluated is taken as a fraction of the full fidelity
    and added to the used budget. For example, 40 epochs of a fidelity that
    maxes out at 100 epochs would count as 0.4 towards the budget.

    If the problem **does** include **many** fidelities, then the fraction as calculated
    for a single fidelity is applied to all fidelities, and then summed, normalized by
    the total number of fidelities. For example, 40 epochs of a fidelity that maxes out
    at 100 epochs and data percentage of 0.6 of a fidelity that maxes out at 1.0 would
    equate to (0.4 + 0.6) / 2 = 0.5 towards the budget.
    """

    used_budget: float = 0.0

    def calculate_used_budget(self, *, result: Result, problem: Problem) -> float:
        """Calculate the used budget for a given result.

        Args:
            result: The result of the trial.
            problem: The original problem statement.

        Returns:
            The amount of budget used for this result.
        """
        match problem.fidelity:
            case None:
                return 1
            case (_, fidelity_desc):
                assert isinstance(result.fidelity, int | float)
                return fidelity_desc.normalize(result.fidelity)
            case Mapping():
                assert problem.benchmark.fidelities is not None
                assert isinstance(result.fidelity, dict)

                normed_fidelities = []
                n_fidelities = len(result.fidelity)
                for k, v in result.fidelity.items():
                    fidelity_desc = problem.benchmark.fidelities[k]
                    norm_fidelity = fidelity_desc.normalize(v)
                    normed_fidelities.append(norm_fidelity)

                return sum(normed_fidelities) / n_fidelities
            case _:
                raise TypeError("Fidelity must be None, str, or list[str]")

    def update(self, *, result: Result, problem: Problem) -> None:
        """Update the budget with the result of a trial.

        Args:
            result: The result of the trial.
            problem: The original problem statement.
        """
        self.used_budget += self.calculate_used_budget(result=result, problem=problem)

    def should_stop(self) -> bool:
        """Check if the budget has been used up."""
        return self.used_budget >= self.budget

    @property
    def path_str(self) -> str:
        """Return a string representation of the budget."""
        clsname = self.__class__.__name__
        return f"{clsname}={self.budget}"

    def clone(self) -> Self:
        """Return a clone of the budget."""
        return replace(self)


@dataclass(kw_only=True)
class CostBudget:
    """A budget for the cost of the trials to run."""

    budget: int | float

    def __post_init__(self):
        raise NotImplementedError("Cost budgets not yet supported")

    def update(self, *, result: Result, problem: Problem) -> None:
        """Update the budget with the result of a trial.

        Args:
            result: The result of the trial.
            problem: The original problem statement.
        """
        raise NotImplementedError("Cost budgets not yet supported")

    def should_stop(self) -> bool:
        """Check if the budget has been used up."""
        raise NotImplementedError("Cost budgets not yet supported")

    def calculate_used_budget(self, *, result: Result, problem: Problem) -> float:
        """Calculate the used budget for a given result.

        Args:
            result: The result of the trial.
            problem: The original problem statement.

        Returns:
            The amount of budget used for this result.
        """
        raise NotImplementedError("Cost budgets not yet supported")

    @property
    def path_str(self) -> str:
        """Return a string representation of the budget."""
        clsname = self.__class__.__name__
        return f"{clsname}={self.budget}"

    def clone(self) -> Self:
        """Return a clone of the budget."""
        return replace(self)


BudgetType: TypeAlias = TrialBudget | CostBudget


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
                if self.optimizer.tabular_support is False:
                    raise ValueError(
                        f"{self.optimizer.name} does not support tabular benchmarks! "
                        f"{self.optimizer.name} and {self.benchmark.name} are not compatible.",
                    )
            case SurrogateBenchmark():
                self.is_tabular = False
            case _:
                raise TypeError("Benchmark must be a TabularBenchmark or SurrogateBenchmark")

        self.is_manyfidelity: bool
        self.is_multifidelity: bool

        match self.fidelity:
            case None:
                match self.optimizer.fidelity_support:
                    case Support(required=True):
                        raise ValueError(
                            f"{self.optimizer.name} requires a fidelity but the problem"
                            f" was specified without a fidelity!\n{self}"
                        )
                    case _:
                        self.is_multifidelity = False
                        self.is_manyfidelity = False
            case tuple():
                if self.optimizer.fidelity_support == Support.NO():
                    raise ValueError(
                        f"{self.optimizer.name} does not support fidelities but the problem"
                        f" was specified with a fidelity!\n{self}"
                    )
                if not self.optimizer.supports_multifidelity:
                    raise ValueError(
                        f"{self.optimizer.name} does not support multi-fidelity but the problem"
                        f" was specified with multi fidelity!\n{self}"
                    )

                self.is_multifidelity = True
                self.is_manyfidelity = False
            case Mapping():
                if len(self.fidelity) == 1:
                    raise ValueError("Single fidelity should be a tuple, not a mapping")

                if not self.optimizer.supports_manyfidelity:
                    raise ValueError(
                        f"{self.optimizer.name} does not support many-fidelity but the problem"
                        f" was specified with many fidelities!\n{self}"
                    )

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

                if not self.optimizer.supports_multiobjective:
                    raise ValueError(
                        f"{self.optimizer.name} does not support multi-objective but the problem"
                        f" was specified with multiple objectives!\n{self}"
                    )

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
        fidelities: Literal["single", "single_required", "many", "many_required"] | Literal[False]
        objectives: Literal["single", "many"]
        cost_awareness: (
            Literal["single", "single_required", "many", "many_required"] | Literal[False]
        )
        tabular: bool

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
