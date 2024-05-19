from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from hpo_glue.query import Query

if TYPE_CHECKING:
    from hpo_glue.config import Config


@dataclass(kw_only=True)
class Result:
    """The result of a query from a benchmark."""

    query: Query
    """The query that generated this result"""

    fidelity: tuple[str, int | float] | Mapping[str, int | float] | None
    """What fidelity the result is at, usually this will be the same as the query fidelity,
    unless the benchmark has multiple fidelities.
    """

    values: dict[str, Any]
    """Everything returned by the benchmark for a given query at the fideltiy."""

    trajectory: pd.DataFrame | None = None
    """If given, the trajectory of the query up to the given fidelity.

    This will only provided if:
    * The problem says it should be provided.
    * There is only a single fidelity parameter.

    It will be multi-indexed, with the multi indexing consiting of the
    config id and the fidelity.
    """

    previous_result: Result | None = None
    """The previous result of a query."""

    @property
    def config_id(self) -> str:
        """The id of the config."""
        return self.query.config_id

    @property
    def query_id(self) -> str:
        """The id of the config."""
        return self.query.query_id

    @property
    def config(self) -> Config:
        """The config."""
        return self.query.config

    @property
    def query_fidelity(self) -> tuple[str, int | float] | Mapping[str, int | float] | None:
        """The fidelity of the query."""
        return self.query.fidelity

    @property
    def config_values(self) -> dict[str, Any]:
        """The fidelity."""
        return self.query.config.values

    def series(self) -> pd.Series:
        """Return the result as a pandas Series."""
        d = {
            **self.query.series(),
            **{f"result.value.{k}": v for k, v in self.values.items()},
        }
        match self.fidelity:
            case None:
                return pd.Series(d)
            case (name, value):
                return pd.Series({**d, f"result.fidelity.{name}": value})
            case Mapping():
                return pd.Series(
                    {**d, **{f"result.fidelity.{k}": v for k, v in self.fidelity.items()}}
                )
            case _:
                raise ValueError(f"Unexpected fidelity type {self.fidelity}")

    @classmethod
    def from_series(cls, series: pd.Series) -> Result:
        """Create a Result from a pandas Series."""
        return Result(
            query=Query.from_series(series.filter(regex=r"query\.|config\.")),
            values=series.filter(like="result.value.")
            .rename(lambda x: x[len("result.value.") :])
            .to_dict(),
            fidelity=series.filter(like="result.fidelity.")
            .rename(lambda x: x[len("result.fidelity.") :])
            .to_dict(),
        )
