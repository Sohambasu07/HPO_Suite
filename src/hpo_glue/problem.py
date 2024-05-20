from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import pandas as pd

from hpo_glue.budget import CostBudget, TrialBudget
from hpo_glue.config import Config
from hpo_glue.constants import DEFAULT_RELATIVE_EXP_DIR
from hpo_glue.fidelity import Fidelity, ListFidelity, RangeFidelity
from hpo_glue.measure import Measure
from hpo_glue.optimizer import Optimizer
from hpo_glue.query import Query
from hpo_glue.result import Result
from hpo_glue.utils import reduce_dtypes

if TYPE_CHECKING:
    from hpo_glue.benchmark import BenchmarkDescription
    from hpo_glue.budget import BudgetType

logger = logging.getLogger(__name__)

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]


def _try_delete_if_exists(path: Path) -> None:
    if path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        except Exception as e:
            logger.exception(e)
            logger.error(f"Error deleting {path}: {e}")


@dataclass(kw_only=True)
class Problem:
    """A problem to optimize over."""

    # NOTE: These are mainly for consumers who need to interact beyond forward facing API
    Config: TypeAlias = Config
    Query: TypeAlias = Query
    Result: TypeAlias = Result
    Measure: TypeAlias = Measure
    TrialBudget: TypeAlias = TrialBudget
    CostBudget: TypeAlias = CostBudget
    RangeFidelity: TypeAlias = RangeFidelity
    ListFidelity: TypeAlias = ListFidelity

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
    """The cost metric to use for this proble.

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

    expdir: Path = field(default=DEFAULT_RELATIVE_EXP_DIR)
    """Default directory to use for experiments."""

    working_dir: Path = field(init=False)
    """The working directory for this problem."""

    df_path: Path = field(init=False)
    """Path to the dataframe of the problem."""

    error_file: Path = field(init=False)
    """Path to the error file of the problem."""

    running_flag: Path = field(init=False)
    """Path to the file flag inidicating this problem is running."""

    complete_flag: Path = field(init=False)
    """Path to the flag inidicating this problem is complete."""

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
        if isinstance(self.expdir, str):
            self.expdir = Path(self.expdir)

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
        self.df_path = self.expdir / "dfs" / f"{self.name}.parquet"

        self.working_dir = self.expdir / self.name
        self.error_file = self.working_dir / "error.txt"
        self.running_flag = self.working_dir / "running.flag"
        self.complete_flag = self.working_dir / "complete.flag"

        self.optimizer.support.check_opt_support(who=self.optimizer.name, problem=self)

    class State(str, Enum):
        """The state of a problem."""

        PENDING = "PENDING"
        RUNNING = "RUNNING"
        CRASHED = "CRASHED"
        COMPLETE = "COMPLETE"

        @classmethod
        def collect(
            cls,
            state: str | Problem.State | bool | Iterable[Problem.State | str],
        ) -> list[Problem.State]:
            """Collect state requested."""
            match state:
                case True:
                    return list(cls)
                case False:
                    return []
                case Problem.State():
                    return [state]
                case str():
                    return [cls(state)]
                case _:
                    return [cls(s) if isinstance(s, str) else s for s in state]

    def state(self) -> Problem.State:
        """Return the state of the problem.

        Args:
            expdir: The directory to check for the problem.
        """
        if self.complete_flag.exists():
            return Problem.State.COMPLETE

        if self.error_file.exists():
            return Problem.State.CRASHED

        if self.running_flag.exists():
            return Problem.State.RUNNING

        return Problem.State.PENDING

    def set_state(
        self,
        state: Problem.State,
        *,
        df: pd.DataFrame | None = None,
        err_tb: tuple[Exception, str] | None = None,
    ) -> None:
        """Set the problem to running.

        Args:
            state: The state to set the problem to.
            df: Optional dataframe to save if setting to [`Problem.State.COMPLETE`].
            err_tb: Optional error traceback to save if setting to [`Problem.State.CRASHED`].
        """
        match state:
            case Problem.State.PENDING:
                # NOTE: Working dir takes care of flags
                for _file in (self.working_dir, self.df_path):
                    _try_delete_if_exists(_file)
            case Problem.State.RUNNING:
                for _file in (self.df_path, self.complete_flag, self.error_file, self.error_file):
                    _try_delete_if_exists(_file)

                self.working_dir.mkdir(parents=True, exist_ok=True)
                self.df_path.parent.mkdir(parents=True, exist_ok=True)
                self.running_flag.touch()
            case Problem.State.CRASHED:
                for _file in (self.complete_flag, self.running_flag, self.df_path):
                    _try_delete_if_exists(_file)

                with self.error_file.open("w") as f:
                    if err_tb is None:
                        f.write("None")
                    else:
                        exc, tb = err_tb
                        f.write(f"{tb}\n{exc}")

            case Problem.State.COMPLETE:
                self.complete_flag.touch()

                if df is not None:
                    df.convert_dtypes().to_parquet(self.df_path)

                for flag in (self.error_file, self.running_flag):
                    _try_delete_if_exists(flag)
            case _:
                raise ValueError(f"Unknown state {state}")

    def as_dict(self) -> dict[str, Any]:
        """Return the Problem as a dictionary."""
        return asdict(self)

    def run(
        self,
        *,
        on_error: Literal["raise", "continue"] = "raise",
        overwrite: Problem.State | str | Sequence[Problem.State | str] | bool = False,
    ) -> Problem.Report:
        """Run the problem.

        Args:
            on_error: How to handle errors. In any case, the error will be written
                into the [`working_dir`][hpo_glue.problem.Problem.working_dir]
                of the problem.

                * If "raise", raise an error.
                * If "continue", log the error and continue.

            overwrite: What to overwrite.

                * If a single value, overwrites problem in that state,
                * If a list of states, overwrites any problem in one of those
                 states.
                * If `True`, overwrite problems in all states.
                * If `False`, don't overwrite any problems.
        """
        from hpo_glue._run import _run_problem

        return _run_problem(problem=self, on_error=on_error, overwrite=overwrite)

    @classmethod
    def generate(  # noqa: PLR0913
        cls,
        optimizers: (
            type[Optimizer]
            | OptWithHps
            | list[type[Optimizer]]
            | list[OptWithHps | type[Optimizer]]
        ),
        benchmarks: BenchmarkDescription | Iterable[BenchmarkDescription],
        *,
        expdir: Path | str = DEFAULT_RELATIVE_EXP_DIR,
        budget: BudgetType | int,
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
            optimizers: The optimizer class to generate problems for.
                Can provide a single optimizer or a list of optimizers.
                If you wish to provide hyperparameters for the optimizer, provide a tuple with the
                optimizer.
            benchmarks: The benchmark to generate problems for.
                Can provide a single benchmark or a list of benchmarks.
            expdir: Which directory to store experiment results into.
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
            expdir=expdir,
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
        results: list[Result]

        def df(self) -> pd.DataFrame:  # noqa: C901, PLR0915
            """Return the history as a pandas DataFrame.

            Returns:
                The history as a pandas DataFrame.
            """

            def _encode_result(_r: Result) -> dict[str, Any]:
                _rparts: dict[str, Any] = {
                    "config.id": _r.config.config_id,
                    "query.id": _r.query.query_id,
                    "result.budget_cost": _r.budget_cost,
                }
                match _r.query.fidelity:
                    case None:
                        _rparts["query.fidelity.count"] = 0
                    case (name, val):
                        _rparts["query.fidelity.count"] = 1
                        _rparts["query.fidelity.1.name"] = name
                        _rparts["query.fidelity.1.value"] = val
                    case Mapping():
                        _rparts["query.fidelity.count"] = len(_r.query.fidelity)
                        for i, (k, v) in enumerate(_r.query.fidelity.items(), start=1):
                            _rparts[f"query.fidelity.{i}.name"] = k
                            _rparts[f"query.fidelity.{i}.value"] = v

                match self.problem.objective:
                    case (_name, _measure):
                        _rparts["result.objective.1.value"] = _r.values[_name]
                    case Mapping():
                        for i, name in enumerate(self.problem.objective, start=1):
                            _rparts[f"result.cost.{i}.value"] = _r.values[name]

                match self.problem.fidelity:
                    case None:
                        pass
                    case (name, _):
                        assert isinstance(_r.fidelity, tuple)
                        _rparts["result.fidelity.1.value"] = _r.fidelity[1]
                    case Mapping():
                        assert isinstance(_r.fidelity, Mapping)
                        for i, name in enumerate(self.problem.fidelity, start=1):
                            _rparts[f"result.fidelity.{i}.value"] = _r.fidelity[name]

                match self.problem.cost:
                    case None:
                        pass
                    case (name, _):
                        _rparts["result.cost.1.value"] = _r.values[name]
                    case Mapping():
                        for i, name in enumerate(self.problem.cost, start=1):
                            _rparts[f"result.fidelity.{i}.value"] = _r.values[name]

                return _rparts

            parts = {}
            problem = self.problem
            parts["problem.name"] = problem.name

            match problem.objective:
                case (name, measure):
                    parts["problem.objective.count"] = 1
                    parts["problem.objective.1.name"] = name
                    parts["problem.objective.1.minimize"] = measure.minimize
                    parts["problem.objective.1.min"] = measure.bounds[0]
                    parts["problem.objective.1.max"] = measure.bounds[1]
                case Mapping():
                    list(problem.objective)
                    parts["problem.objective.count"] = len(problem.objective)
                    for i, (k, v) in enumerate(problem.objective.items(), start=1):
                        parts[f"problem.objective.{i}.name"] = k
                        parts[f"problem.objective.{i}.minimize"] = v.minimize
                        parts[f"problem.objective.{i}.min"] = v.bounds[0]
                        parts[f"problem.objective.{i}.max"] = v.bounds[1]
                case _:
                    raise TypeError("Objective must be a tuple (name, measure) or a mapping")

            match problem.fidelity:
                case None:
                    parts["problem.fidelity.count"] = 0
                case (name, fid):
                    parts["problem.fidelity.count"] = 1
                    parts["problem.fidelity.1.name"] = name
                    parts["problem.fidelity.1.min"] = fid.min
                    parts["problem.fideltiy.1.max"] = fid.max
                case Mapping():
                    list(problem.fidelity)
                    parts["problem.fidelity.count"] = len(problem.fidelity)
                    for i, (k, v) in enumerate(problem.fidelity.items(), start=1):
                        parts[f"problem.fidelity.{i}.name"] = k
                        parts[f"problem.fidelity.{i}.min"] = v.min
                        parts[f"problem.fidelity.{i}.max"] = v.max
                case _:
                    raise TypeError("Must be a tuple (name, fidelitiy) or a mapping")

            match problem.cost:
                case None:
                    parts["problem.cost.count"] = 0
                case (name, cost):
                    parts["problem.cost.count"] = 1
                    parts["problem.cost.1.name"] = name
                    parts["problem.cost.1.minimize"] = cost.minimize
                    parts["problem.cost.1.min"] = cost.bounds[0]
                    parts["problem.cost.1.max"] = cost.bounds[1]
                case Mapping():
                    list(problem.cost)
                    parts["problem.cost.count"] = len(problem.cost)
                    for i, (k, v) in enumerate(problem.cost.items(), start=1):
                        parts[f"problem.cost.{i}.name"] = k
                        parts[f"problem.cost.{i}.minimize"] = v.minimize
                        parts[f"problem.cost.{i}.min"] = v.bounds[0]
                        parts[f"problem.cost.{i}.max"] = v.bounds[1]

            _df = pd.DataFrame.from_records([_encode_result(r) for r in self.results])
            for k, v in parts.items():
                _df[k] = v

            _df["problem.benchmark"] = problem.benchmark.name
            _df["problem.seed"] = problem.seed
            match problem.budget:
                case TrialBudget(total):
                    _df["problem.budget.kind"] = "TrialBudget"
                    _df["problem.budget.total"] = total
                case CostBudget(total):
                    _df["problem.budget.kind"] = "CostBudget"
                    _df["problem.budget.total"] = total
                case _:
                    raise NotImplementedError(f"Unknown budget type {problem.budget}")

            _df["problem.opt.name"] = problem.optimizer.name

            if len(problem.optimizer_hyperparameters) > 0:
                for k, v in problem.optimizer_hyperparameters.items():
                    _df[f"problem.opt.hp.{k}"] = v

                _df["problem.opt.hp_str"] = ",".join(
                    f"{k}={v}" for k, v in problem.optimizer_hyperparameters.items()
                )
            else:
                _df["problem.opt.hp_str"] = "default"

            _df = _df.convert_dtypes()
            cat_cols = [
                c
                for c in _df.select_dtypes(include="string").columns
                if c not in ("config.id", "query.id")
            ]
            _df[cat_cols] = _df[cat_cols].astype("category")
            return reduce_dtypes(_df, reduce_int=True, reduce_float=True)

        @classmethod
        def from_df(cls, df: pd.DataFrame, problem: Problem) -> Problem.Report:  # noqa: C901, PLR0915
            """Load a GLUEReport from a pandas DataFrame.

            Args:
                df: The dataframe to load from. Will subselect rows
                    that match the problem name.
                problem: The problem definition that generated the runs.
            """

            def _row_to_result(series: pd.Series) -> Result:
                _row = series.to_dict()
                _result_values: dict[str, Any] = {}
                match problem.objective:
                    case (name, _):
                        assert int(_row["problem.objective.count"]) == 1
                        assert str(_row["problem.objective.1.name"]) == name
                        _result_values[name] = _row["result.objective.1.value"]
                    case Mapping():
                        assert int(_row["problem.objective.count"]) == len(problem.objective)
                        for i, k in enumerate(problem.objective, start=1):
                            assert str(_row[f"problem.objective.{i}.name"]) == k
                            _result_values[k] = _row[f"result.objective.{i}.value"]
                    case _:
                        raise TypeError("Objective must be a tuple (name, measure) or a mapping")

                match problem.fidelity:
                    case None:
                        assert int(_row["problem.fidelity.count"]) == 0
                        _result_fidelity = None
                    case (name, fid):
                        assert int(_row["problem.fidelity.count"]) == 1
                        assert str(_row["problem.fidelity.1.name"]) == name
                        _result_fidelity = (name, fid.kind(_row["result.fidelity.1.value"]))
                    case Mapping():
                        assert int(_row["problem.fidelity.count"]) == len(problem.fidelity)
                        _result_fidelity = {}
                        for i, (name, fid) in enumerate(problem.fidelity.items(), start=1):
                            assert str(_row[f"problem.fidelity.{i}.name"]) == name
                            _result_fidelity[name] = fid.kind(_row[f"result.fidelity.{i}.value"])
                    case _:
                        raise TypeError("Must be a tuple (name, fidelitiy) or a mapping")

                match problem.cost:
                    case None:
                        assert int(_row["problem.cost.count"]) == 0
                    case (name, _):
                        assert int(_row["problem.cost.count"]) == 1
                        assert str(_row["problem.cost.1.name"]) == name
                        _result_values[name] = _row["result.cost.1.value"]
                    case Mapping():
                        assert int(_row["problem.cost.count"]) == len(problem.cost)
                        for i, (name, _) in enumerate(problem.cost.items(), start=1):
                            assert str(_row[f"problem.cost.{i}.name"]) == name
                            _result_values[name] = _row[f"result.cost.{i}.value"]
                    case _:
                        raise TypeError("Must be a tuple (name, fidelitiy) or a mapping")

                _query_f_count = int(_row["query.fidelity.count"])
                _query_fidelity: None | tuple[str, int | float] | dict[str, int | float]
                match _query_f_count:
                    case 0:
                        _query_fidelity = None
                    case 1:
                        _name = str(_row["query.fidelity.1.name"])
                        assert problem.benchmark.fidelities is not None
                        _fid = problem.benchmark.fidelities[_name]
                        _val = _fid.kind(_row["query.fidelity.1.value"])
                        _query_fidelity = (_name, _val)
                    case _:
                        _query_fidelity = {}
                        for i in range(1, _query_f_count + 1):
                            _name = str(_row[f"query.fidelity.{i}.name"])
                            assert problem.benchmark.fidelities is not None
                            _fid = problem.benchmark.fidelities[_name]
                            _query_fidelity[_name] = _fid.kind(_row[f"query.fidelity.{i}.value"])

                return Result(
                    query=Query(
                        config=Config(config_id=str(_row["config.id"]), values=None),
                        optimizer_info=None,
                        request_trajectory=False,
                        fidelity=_query_fidelity,
                    ),
                    budget_cost=float(_row["result.budget_cost"]),
                    values=_result_values,
                    fidelity=_result_fidelity,
                    trajectory=None,
                )

            this_problem = df["problem.name"] == problem.name
            problem_columns = [c for c in df.columns if c.startswith("problem.")]
            problem_duplicate_rows = df[this_problem].drop_duplicates(subset=problem_columns)
            if len(problem_duplicate_rows) > 1:
                raise ValueError(
                    f"Multiple problem rows found for the provided df for problem '{problem.name}'"
                    f"\n{problem_duplicate_rows}"
                )

            results = [_row_to_result(row) for _, row in df[this_problem].iterrows()]
            return cls(problem=problem, results=results)

        def save(self, path: Path) -> None:
            """Save the GLUEReport to a path."""
            self.df().to_parquet(path, index=False)

        @classmethod
        def from_path(cls, path: Path, problem: Problem) -> Problem.Report:
            """Load a GLUEReport from a path."""
            df = pd.read_parquet(path)  # noqa: PD901
            return cls.from_df(df, problem=problem)
