from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import pandas as pd

from hpo_glue.budget import CostBudget, TrialBudget
from hpo_glue.history import History
from hpo_glue.optimizer import Optimizer

if TYPE_CHECKING:
    from hpo_glue.benchmark import BenchmarkDescription
    from hpo_glue.budget import BudgetType
    from hpo_glue.fidelity import Fidelity
    from hpo_glue.measure import Measure

logger = logging.getLogger(__name__)

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]


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

    optimizer_hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    """The hyperparameters to use for the optimizer"""

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

    supports_trajectory: bool = field(init=False)
    """Whether the problem setup allows for trajectories to be queried."""

    def __post_init__(self):
        self.is_tabular = self.benchmark.is_tabular
        self.is_manyfidelity: bool
        self.is_multifidelity: bool
        self.supports_trajectory: bool
        match self.fidelity:
            case None:
                self.is_multifidelity = False
                self.is_manyfidelity = False
                self.supports_trajectory = False
            case (_name, _fidelity):
                self.is_multifidelity = True
                self.is_manyfidelity = False
                if _fidelity.supports_continuation:
                    self.supports_trajectory = True
                else:
                    self.supports_trajectory = False

            case Mapping():
                if len(self.fidelity) == 1:
                    raise ValueError("Single fidelity should be a tuple, not a mapping")

                self.is_multifidelity = False
                self.is_manyfidelity = True
                self.supports_trajectory = False
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
        self._objective_str = objective_str
        self.name = "_".join(f"{k}={v}" for k, v in name_parts.items())
        self.optimizer.support.check_opt_support(who=self.optimizer.name, problem=self)

    def as_dict(self) -> dict[str, Any]:
        """Return the Problem as a dictionary."""
        return asdict(self)

    def run(
        self,
        *,
        overwrite: bool = False,
        expdir: Path | str = "hpo-glue-output",
    ) -> Problem.Report:
        """Run the problem.

        Args:
            overwrite: Whether to overwrite the results if they already exist.
            expdir: The directory to store the results in.
        """
        from hpo_glue._run import run_problem

        return run_problem(problem=self, expdir=expdir, overwrite=overwrite)

    @classmethod
    def generate(
        cls,
        optimizers: (
            type[Optimizer]
            | OptWithHps
            | list[type[Optimizer]]
            | list[OptWithHps | type[Optimizer]]
        ),
        benchmarks: BenchmarkDescription | Iterable[BenchmarkDescription],
        *,
        budget: BudgetType | int | float,
        seeds: int | Iterable[int],
        fidelities: int = 0,
        objectives: int = 1,
        costs: int = 0,
        multi_objective_generation: Literal["mix_metric_cost", "metric_only"] = "mix_metric_cost",
        on_error: Literal["warn", "raise", "ignore"] = "warn",
    ) -> Iterator[Problem]:
        """Generate a set of problems for the given optimizer and benchmark.

        If there is some incompatibility between the optimizer, the benchmark and the requested
        amount of objectives, fidelities or costs, a ValueError will be raised.

        Args:
            optimizer: The optimizer class to generate problems for.
                Can provide a single optimizer or a list of optimizers.
                If you wish to provide hyperparameters for the optimizer, provide a tuple with the
                optimizer.
            benchmark: The benchmark to generate problems for.
                Can provide a single benchmark or a list of benchmarks.
            budget: The budget to use for the problems. Budget defaults to a n_trials budget
                where when multifidelty is enabled, fractional budget can be used and 1 is
                equivalent a full fidelity trial.
            seeds: The seed or seeds to use for the problems.
            fidelities: The number of fidelities to generate problems for.
            objectives: The number of objectives to generate problems for.
            costs: The number of costs to generate problems for.
            multi_objective_generation: The method to generate multiple objectives.
            on_error: The method to handle errors.

                * "warn": Log a warning and continue.
                * "raise": Raise an error.
                * "ignore": Ignore the error and continue.
        """
        from hpo_glue._problem_generators import _generate_problem_set

        yield from _generate_problem_set(
            optimizers=optimizers,
            benchmarks=benchmarks,
            budget=budget,
            seeds=seeds,
            fidelities=fidelities,
            objectives=objectives,
            costs=costs,
            multi_objective_generation=multi_objective_generation,
            on_error=on_error,
        )

    def series(self) -> pd.Series:
        """Return the Problem as a pandas Series."""
        parts = {}
        parts["name"] = self.name

        match self.objective:
            case tuple():
                parts["objective"] = self.objective[0]
            case Mapping():
                parts["objective"] = ",".join(self.objective.keys())
            case _:
                raise TypeError("Objective must be a tuple (name, measure) or a mapping")

        match self.fidelity:
            case None:
                parts["fidelity"] = None
            case (name, _):
                parts["fidelity"] = name
            case Mapping():
                parts["fidelity"] = ",".join(self.fidelity.keys())

        match self.cost:
            case None:
                parts["cost"] = None
            case (name, _):
                parts["cost"] = name
            case Mapping():
                parts["cost"] = ",".join(self.cost.keys())

        parts["benchmark"] = self.benchmark.name
        parts["seed"] = self.seed
        match self.budget:
            case TrialBudget(budget):
                parts["budget.kind"] = "TrialBudget"
                parts["budget.value"] = budget
            case CostBudget(budget):
                parts["budget.kind"] = "CostBudget"
                parts["budget.value"] = budget
            case _:
                raise NotImplementedError(f"Unknown budget type {self.budget}")

        parts["opt.name"] = self.optimizer.name

        for k, v in self.optimizer_hyperparameters.items():
            parts[f"opt.hp.{k}"] = v

        return pd.Series({f"problem.{k}": v for k, v in parts.items()})

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

        def df(self) -> pd.DataFrame:
            """Return the history as a pandas DataFrame."""
            # Fill in the fields of the problem as constant columns
            _df = self.history.df()
            problem_series = self.problem.series()
            for k, v in problem_series.items():
                _df[k] = v

            return _df.convert_dtypes()

        @classmethod
        def from_df(cls, df: pd.DataFrame, problem: Problem) -> Problem.Report:
            return cls(problem=problem, history=History.from_df(df))

        def save(self, path: Path) -> None:
            """Save the GLUEReport to a path."""
            self.df().to_parquet(path, index=False)

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
