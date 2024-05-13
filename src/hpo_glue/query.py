from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from hpo_glue.config import Config


@dataclass(frozen=True)
class Query:
    """A query to a benchmark."""

    config: Config
    """ The config to evaluate """

    fidelity: int | float | None
    """ What fidelity to evaluate at """

    @property
    def config_id(self) -> str:
        """The id of the config."""
        return self.config.id

    @property
    def config_values(self) -> dict[str, Any]:
        """The fidelity."""
        return self.config.values

    def series(self) -> pd.Series:
        """Return the query as a pandas Series."""
        return pd.Series(
            {
                **self.config.series(),
                "query.fidelity": self.fidelity,
            }
        )

    @classmethod
    def from_series(cls, series: pd.Series) -> Query:
        """Create a Query from a pandas Series."""
        return Query(
            config=Config.from_series(series),
            fidelity=series["query.fidelity"],  # type: ignore
        )
