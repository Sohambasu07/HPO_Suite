from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from hpo_glue.config import Config
from hpo_glue.result import Result

if TYPE_CHECKING:
    import pandas as pd
    from ConfigSpace import ConfigurationSpace

    from hpo_glue.query import Query

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkFactory:
    f: Callable[..., Benchmark]
    unique_name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    has_conditionals: bool = False
    supports_multifidelity: bool = False
    supports_multiobjective: bool = False
    supports_manyfidelity: bool = False
    is_tabular: bool = False

    def __call__(self, **kwargs: Any) -> Benchmark:
        return self.f(**{**self.kwargs, **kwargs})


class SurrogateBenchmark:
    """Defines the interface for a surrogate benchmark."""

    name: str
    """ The name of the benchmark """

    config_space: ConfigurationSpace
    """ The configuration space for the benchmark """

    result_keys: str | list[str]
    """The key(s) in the benchmark that we want to consider as the results """

    default_objective: str
    """ Default objective to optimize """

    default_fidelity: str | None
    """Default fidelity to use."""

    minimize_default: bool
    """ Whether the default objective should be minimized """

    fidelity_space: list[int] | list[float] | None
    """All possible fidelities for the benchmark """

    fidelity_keys: str | list[str] | None
    """The key(s) in the benchmark that we want to consider as the fidelity """

    query_function: Callable[[Query], Result]
    """ The function to query the benchmark """

    benchmark: Any
    """ The actual benchmark object """

    time_budget: str | None
    """ Time budget support:
            str: time budget key
            None: time budget not supported
    """

    def __init__(  # noqa: D107, PLR0913
        self,
        *,
        name: str,
        config_space: ConfigurationSpace,
        result_keys: list[str],
        default_objective: str,
        minimize_default: bool,
        query_function: Callable[[Query], Result],
        benchmark: Any,
        fidelity_keys: str | list[str] | None = None,
        fidelity_space: list[int] | list[float] | None = None,
        time_budget: str | None = None,
    ) -> None:
        self.name = name
        self.config_space = config_space
        self.result_keys = result_keys
        self.default_objective = default_objective
        self.minimize_default = minimize_default
        self.fidelity_space = fidelity_space
        self.fidelity_keys = fidelity_keys
        self.query_function = query_function
        self.benchmark = benchmark
        self.time_budget = time_budget

    def query(self, query: Query) -> Result:
        """Query the benchmark for a result.

        Args:
            query: The query to the benchmark

        Returns:
            The result of the query
        """
        return self.query_function(query)


class TabularBenchmark:
    """Defines the interface for a tabular benchmark."""

    name: str
    """ The name of the benchmark """

    table: pd.DataFrame
    """ The table holding all information """

    id_key: str
    """ The key in the table that we want to use as the id """

    config_space: list[Config]
    """ All possible configs for the benchmark """

    result_keys: str | list[str]
    """The key(s) in the benchmark that we want to consider as the results """

    default_objective: str
    """Default objective to optimize """

    default_fidelity: str | None
    """Default fidelity to use."""

    minimize_default: bool
    """ Whether the default objective should be minimized """

    fidelity_space: list[int] | list[float] | None
    """ All possible fidelities for the benchmark """

    fidelity_keys: str | list[str] | None
    """The key(s) in the benchmark that we want to consider as the fidelity """

    time_budget: str | None
    """ Time budget support, None indicates not supported."""

    def __init__(  # noqa: C901, D107, PLR0913
        self,
        name: str,
        table: pd.DataFrame,
        *,
        id_key: str,
        config_keys: list[str],
        result_keys: list[str],
        default_objective: str,
        minimize_default: bool,
        fidelity_keys: str | list[str] | None = None,
        remove_constants: bool = False,
        time_budget: str | None = None,
    ) -> None:
        # Make sure we work with a clean slate, no issue with index.
        table = table.reset_index()

        # Make sure all the keys they specified exist
        if id_key not in table.columns:
            raise ValueError(f"'{id_key=}' not in columns {table.columns}")

        if fidelity_keys not in table.columns:
            raise ValueError(f"'{fidelity_keys=}' not in columns {table.columns}")

        if not all(key in table.columns for key in result_keys):
            raise ValueError(f"{result_keys=} not in columns {table.columns}")

        if not all(key in table.columns for key in config_keys):
            raise ValueError(f"{config_keys=} not in columns {table.columns}")

        # Make sure that the column `id` only exist if it's the `id_key`
        if "id" in table.columns and id_key != "id":
            raise ValueError(
                f"Can't have `id` in the columns if it's not the {id_key=}."
                " Please drop it or rename it.",
            )

        # Remove constants from the table
        if remove_constants:

            def is_constant(_s: pd.Series) -> bool:
                _arr = _s.to_numpy()
                return bool((_arr == _arr[0]).all())

            constant_cols = [
                col
                for col in table.columns
                if is_constant(table[col])  # type: ignore
            ]
            table = table.drop(columns=constant_cols)  # type: ignore
            config_keys = [k for k in config_keys if k not in constant_cols]

        # Remap their id column to `id`
        table = table.rename(columns={id_key: "id"})

        # Index the table
        match fidelity_keys:
            case None:
                index_cols: list[str] = ["id"]
            case str():
                index_cols = ["id", fidelity_keys]
            case list():
                index_cols = ["id", *fidelity_keys]

        # Drop all the columns that are not relevant
        relevant_cols: list[str] = [
            *index_cols,
            *result_keys,
            *config_keys,
        ]
        table = table[relevant_cols]  # type: ignore
        table = table.set_index(index_cols).sort_index()

        # We now have the following table
        #
        #     id    fidelity | **metric, **config_values
        #     0         0    |
        #               1    |
        #               2    |
        #     1         0    |
        #               1    |
        #               2    |
        #   ...

        self.name = name
        self.table = table
        self.id_key = id_key
        self.fidelity_keys = fidelity_keys
        self.config_keys = sorted(config_keys)
        self.result_keys = sorted(result_keys)
        self.default_objective = default_objective
        self.minimize_default = minimize_default
        self.time_budget = time_budget

        # A list of all possible configs
        self.config_space = [
            Config(id=str(i), values=config)  # enforcing str for id
            for i, config in enumerate(
                self.table[self.config_keys]
                .drop_duplicates()
                # Sorting is important to make sure it's always consistent
                .sort_values(by=self.config_keys)
                .to_dict(orient="records"),
            )
        ]

        match self.fidelity_keys:
            case None:
                self.fidelity_space = None
                self.fidelity_range = None
            case str():
                # Make sure we have equidistance fidelities for all configs
                fidelity_values = self.table.index.get_level_values(self.fidelity_keys)
                fidelity_counts = fidelity_values.value_counts()
                if not (fidelity_counts == fidelity_counts.iloc[0]).all():
                    raise ValueError(f"{self.fidelity_keys=} not uniform. \n{fidelity_counts}")

                sorted_fids = sorted(fidelity_values.unique())
                start = sorted_fids[0]
                end = sorted_fids[-1]

                # Here we get all the unique configs
                #     id    fidelity | **metric, **config_values
                #     0         0    |
                #     1         0    |
                #   ...
                self.fidelity_space = sorted_fids
                self.fidelity_range = (start, end)
            case list():
                raise NotImplementedError("Many fidelities not yet supported")
            case _:
                raise TypeError(f"type of {self.fidelity_keys=} not supported")

    def query(self, query: Query) -> Result:
        """Query the benchmark for a result."""
        match query.fidelity:
            case None:
                result = self.table.loc[query.config_id]
            case int() | float():
                result = self.table.loc[(query.config_id, query.fidelity)]
            case list():
                raise NotImplementedError("Many fidelities not yet supported")
            case _:
                raise TypeError(f"type of {query.fidelity=} not supported")

        return Result(
            query=query,
            result=result.get(self.result_keys).to_dict(),
        )


# NOTE(eddiebergman): Not using a base class as we really don't expect to need
# more than just these two types of benchmarks.
Benchmark: TypeAlias = TabularBenchmark | SurrogateBenchmark
