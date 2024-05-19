from __future__ import annotations

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

    values: dict[str, Any]
    """Everything returned by the benchmark for a given query."""

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
    def fidelity(self) -> int | float | dict[str, int | float] | None:
        """The fidelity."""
        return self.query.fidelity

    @property
    def config_values(self) -> dict[str, Any]:
        """The fidelity."""
        return self.query.config.values

    def series(self) -> pd.Series:
        """Return the result as a pandas Series."""
        return pd.Series(
            {
                **self.query.series(),
                **{f"result.{k}": v for k, v in self.values.items()},
            }
        )

    @classmethod
    def from_series(cls, series: pd.Series) -> Result:
        """Create a Result from a pandas Series."""
        return Result(
            query=Query.from_series(series.filter(like="query.")),
            values=series.filter(like="result.").to_dict(),
        )
