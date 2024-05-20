from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

import pandas as pd

from hpo_glue.config import Config
from hpo_glue.constants import DEFAULT_RELATIVE_EXP_DIR
from hpo_glue.optimizer import Optimizer
from hpo_glue.result import Result

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from hpo_glue.budget import BudgetType
    from hpo_glue.fidelity import Fidelity
    from hpo_glue.measure import Measure
    from hpo_glue.problem import Problem
    from hpo_glue.query import Query

    class TrajectoryF(Protocol):
        def __call__(
            self,
            *,
            query: Query,
            frm: int | float | None = None,
            to: int | float | None = None,
        ) -> pd.DataFrame: ...


logger = logging.getLogger(__name__)

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]


@dataclass(kw_only=True)
class BenchmarkDescription:
    """Describes a benchmark without loading it all in."""

    name: str
    """Unique name of the benchmark."""

    load: Callable[[BenchmarkDescription], Benchmark]
    """Function to load the benchmark."""

    metrics: Mapping[str, Measure]
    """All the metrics that the benchmark supports."""

    test_metrics: Mapping[str, Measure] | None = None
    """All the test metrics that the benchmark supports."""

    costs: Mapping[str, Measure] | None = None
    """All the costs that the benchmark supports."""

    fidelities: Mapping[str, Fidelity] | None = None
    """All the fidelities that the benchmark supports."""

    has_conditionals: bool = False
    """Whether the benchmark has conditionals."""

    is_tabular: bool = False
    """Whether the benchmark is tabular."""

    def generate_problems(
        self,
        optimizers: (
            type[Optimizer]
            | OptWithHps
            | list[type[Optimizer]]
            | list[OptWithHps | type[Optimizer]]
        ),
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
            optimizers=optimizers,
            benchmarks=self,
            budget=budget,
            seeds=seeds,
            expdir=expdir,
            fidelities=fidelities,
            objectives=objectives,
            costs=costs,
            multi_objective_generation=multi_objective_generation,
            on_error=on_error,
        )


@dataclass(kw_only=True)
class SurrogateBenchmark:
    """Defines the interface for a surrogate benchmark."""

    desc: BenchmarkDescription
    """The description of the benchmark."""

    config_space: ConfigurationSpace
    """ The configuration space for the benchmark """

    benchmark: Any
    """The wrapped benchmark object."""

    query: Callable[[Query], Result]
    """The query function for the benchmark."""

    trajectory_f: TrajectoryF | None = None
    """The trajectory function for the benchmark, if one exists.

    This function should return a DataFrame with the trajectory of the query up
    to the given fidelity. The index should be the fidelity parameter with the
    columns as the values.

    ```
    def __call__(
        self,
        *,
        query: Query,
        frm: int | float | None = None,
        to: int | float | None = None,
    ) -> pd.DataFrame:
        ...
    ```

    If not provided, the query will be called repeatedly to generate this.
    """

    def trajectory(
        self,
        *,
        query: Query,
        frm: int | float | None = None,
        to: int | float | None = None,
    ) -> pd.DataFrame:
        if self.trajectory_f is not None:
            return self.trajectory_f(query=query, frm=frm, to=to)

        assert isinstance(query.fidelity, tuple)
        assert self.desc.fidelities is not None

        fid_name, fid_value = query.fidelity
        fid = self.desc.fidelities[fid_name]
        frm = frm if frm is not None else fid.min
        to = to if to is not None else fid_value

        index: list[int] | list[float] = []
        results: list[Result] = []
        for val in iter(fid):
            if val < frm:
                continue

            if val > to:
                break

            index.append(val)
            result = self.query(query.with_fidelity((fid_name, val)))
            results.append(result)

        # Return in trajectory format
        # fid_name    **results
        # 0         | . | . | ...
        # 1         | . | . | ...
        # ...
        return pd.DataFrame.from_records(
            [result.values for result in results],
            index=pd.Index(index, name=fid_name),
        )


class TabularBenchmark:
    """Defines the interface for a tabular benchmark."""

    desc: BenchmarkDescription
    """The description of the benchmark."""

    table: pd.DataFrame
    """ The table holding all information """

    id_key: str
    """The key in the table that we want to use as the id."""

    config_space: list[Config]
    """ All possible configs for the benchmark """

    config_keys: list[str]
    """The keys in the table to use as the config keys."""

    result_keys: list[str]
    """The keys in the table to use as the result keys.

    This is inferred from the `desc=`.
    """

    def __init__(
        self,
        *,
        desc: BenchmarkDescription,
        table: pd.DataFrame,
        id_key: str,
        config_keys: list[str],
    ) -> None:
        """Create a tabular benchmark.

        The result and fidelity keys will be inferred from the `desc=`.

        Args:
            desc: The description of the benchmark.
            table: The table holding all information.
            id_key: The key in the table that we want to use as the id.
            config_keys: The keys in the table that we want to use as the config.
        """
        # Make sure we work with a clean slate, no issue with index.
        table = table.reset_index()

        for key in config_keys:
            if key not in table.columns:
                raise KeyError(
                    f"Config key '{key}' not in columns {table.columns}."
                    "This is most likely from a misspecified BecnhmarkDescription for "
                    f"{desc.name}.",
                )

        result_keys = [
            *desc.metrics.keys(),
            *(desc.test_metrics.keys() if desc.test_metrics else []),
            *(desc.costs.keys() if desc.costs else []),
        ]
        for key in result_keys:
            if key not in table.columns:
                raise KeyError(
                    f"Result key '{key}' not in columns {table.columns}."
                    "This is most likely from a misspecified BecnhmarkDescription for "
                    f"{desc.name}.",
                )

        match desc.fidelities:
            case None:
                fidelity_keys = None
            case Mapping():
                fidelity_keys = list(desc.fidelities.keys())
                for key in fidelity_keys:
                    if key not in table.columns:
                        raise KeyError(
                            f"Fidelity key '{key}' not in columns {table.columns}."
                            "This is most likely from a misspecified BecnhmarkDescription for "
                            f"{desc.name}.",
                        )
            case _:
                raise TypeError(f"{desc.fidelities=} not supported")

        # Make sure that the column `id` only exist if it's the `id_key`
        if "id" in table.columns and id_key != "id":
            raise ValueError(
                f"Can't have `id` in the columns if it's not the {id_key=}."
                " Please drop it or rename it.",
            )

        # Remap their id column to `id`
        table = table.rename(columns={id_key: "id"})

        # We will create a multi-index for the table, done by the if and
        # the remaining fidelity keys
        _fid_cols = [] if fidelity_keys is None else fidelity_keys

        # Drop all the columns that are not relevant
        relevant_cols: list[str] = ["id", *_fid_cols, *result_keys, *config_keys]
        table = table[relevant_cols]  # type: ignore
        table = table.set_index(["id", *_fid_cols]).sort_index()

        # We now have the following table
        #
        #     id    fidelity | **metrics, **config_values
        #     0         0    |
        #               1    |
        #               2    |
        #     1         0    |
        #               1    |
        #               2    |
        #   ...
        self.table = table
        self.id_key = id_key
        self.desc = desc
        self.config_keys = config_keys
        self.result_keys = result_keys
        self.fidelity_keys = fidelity_keys
        self.config_space = [
            Config(config_id=str(i), values=config)  # enforcing str for id
            for i, config in enumerate(
                self.table[config_keys]
                .drop_duplicates()
                .sort_values(by=config_keys)  # Sorting to ensure table config order is consistent
                .to_dict(orient="records"),
            )
        ]

    def query(self, query: Query) -> Result:
        """Query the benchmark for a result."""
        # NOTE(eddiebergman):
        # Some annoying logic here to basically be able to handle partially specified fidelties,
        # even if it does not match the order of what the table has. In the case where a fidelity
        # is not specified, we select ALL (slice(None)) for that fidelity. Later, we will just take
        # the last row then. Important that we go in the order of `self.fidelity_keys`
        ALL = slice(None)
        fidelity_order = self.table.index.names[1:]

        match query.fidelity:
            case None:
                slices = {col: ALL for col in fidelity_order}
            case (key, value):
                assert self.fidelity_keys is not None
                slices = {col: (value if key == col else ALL) for col in fidelity_order}
            case Mapping():
                assert self.fidelity_keys is not None
                slices = {col: query.fidelity.get(col, ALL) for col in fidelity_order}
            case _:
                raise TypeError(f"type of {query.fidelity=} ({type(query.fidelity)}) supported")

        result = self.table.loc[(query.config_id, *slices.values())]
        row: pd.Series
        match result:
            case pd.Series():
                # If it's a series, a row was uniquely specified, meaning that all
                # of the fidelity values were specified.
                retrieved_results = result[self.result_keys]
                assert isinstance(retrieved_results, pd.Series)
                unspecified_fids = {}
            case pd.DataFrame():
                # If it's a DataFrame, we have multiple rows, we take the last one,
                # under the assumption that:
                # 1. Larger fidelity values are what is requested.
                # 2. The table is sorted by fidelity values.
                retrieved_results = result[self.result_keys]

                # Get the non-specified fidelity values
                # We have to keep it as a dataframe using `[-1:]`
                # for the moment so we can get the correct fidelity names and values.
                row = result.iloc[-1:]
                assert isinstance(row, pd.DataFrame)
                assert len(row.index) == 1
                unspecified_fids = dict(zip(row.index.names, row.index, strict=True))

                retrieved_results = retrieved_results.iloc[-1]
            case _:
                raise TypeError(f"type of {result=} ({type(result)}) not supported")

        match query.fidelity:
            case None:
                fidelities_retrieved = unspecified_fids
            case (key, value):
                fidelities_retrieved = {**unspecified_fids, key: value}
            case Mapping():
                fidelities_retrieved = {**unspecified_fids, **query.fidelity}

        return Result(
            query=query,
            values=retrieved_results.to_dict(),
            fidelity=fidelities_retrieved,
        )

    def trajectory(
        self,
        *,
        query: Query,
        frm: int | float | None = None,
        to: int | float | None = None,
    ) -> pd.DataFrame:
        """Query the benchmark for a result."""
        assert isinstance(query.fidelity, tuple)
        fid_name, fid_value = query.fidelity
        if self.fidelity_keys is None:
            raise ValueError("No fidelities to query for this benchmark!")

        if self.fidelity_keys != [fid_name]:
            raise NotImplementedError(
                f"Can't get trajectory for {fid_name=} when more than one"
                f" fidelity {self.fidelity_keys=}!",
            )

        assert self.desc.fidelities is not None
        frm = frm if frm is not None else self.desc.fidelities[fid_name].min
        to = to if to is not None else fid_value

        # Return in trajectory format
        # fid_name    **results
        # 0         | . | . | ...
        # 1         | . | . | ...
        # ...
        return self.table[self.result_keys].loc[query.config_id, frm:to].droplevel(0).sort_index()


# NOTE(eddiebergman): Not using a base class as we really don't expect to need
# more than just these two types of benchmarks.
Benchmark: TypeAlias = TabularBenchmark | SurrogateBenchmark
