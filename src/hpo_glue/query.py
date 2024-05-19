from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from hpo_glue.config import Config


@dataclass(frozen=True)
class Query:
    """A query to a benchmark."""

    config: Config
    """ The config to evaluate """

    fidelity: tuple[str, int | float] | Mapping[str, int | float] | None
    """ What fidelity to evaluate at """

    optimizer_info: Any | None = None
    """Any optimizer specific info required across ask and tell"""

    @property
    def config_id(self) -> str:
        """The id of the config."""
        return self.config.id

    @property
    def query_id(self) -> str:
        """The id of the query."""
        match self.fidelity:
            case None:
                return self.config.id
            case (name, value):
                return f"{self.config.id}-{name}={value}"
            case Mapping():
                return f"{self.config.id}-{'-'.join(f'{k}={v}' for k, v in self.fidelity.items())}"
            case _:
                raise NotImplementedError("Unexpected fidelity type")

    @property
    def config_values(self) -> dict[str, Any]:
        """The fidelity."""
        return self.config.values

    def series(self) -> pd.Series:
        """Return the query as a pandas Series."""
        match self.fidelity:
            case None:
                return self.config.series()
            case (name, value):
                return pd.Series(
                    {
                        **self.config.series(),
                        f"query.fidelity.{name}": value,
                    }
                )
            case Mapping():
                return pd.Series(
                    {
                        **self.config.series(),
                        **{f"query.fidelity.{k}": v for k, v in self.fidelity.items()},
                    }
                )
            case _:
                raise NotImplementedError("Unexpected fidelity type")

    @classmethod
    def from_series(cls, series: pd.Series) -> Query:
        """Create a Query from a pandas Series."""
        fid: tuple[str, int | float] | Mapping[str, int | float] | None
        filtered = series.filter(like="query.fidelity.").to_dict()
        match len(filtered):
            case 0:
                fid = None
            case 1:
                k, v = next(iter(filtered.items()))
                fid = (k.split("query.fidelity.")[1], v)
            case _:
                fid = {k.split("query.fidelity.")[1]: v for k, v in filtered.items()}

        # NOTE(eddiebergman): We unlikely have the optimizer info in the series
        return Query(config=Config.from_series(series), fidelity=fid, optimizer_info=None)
