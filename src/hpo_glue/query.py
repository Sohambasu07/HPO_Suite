from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from hpo_glue.config import Config


@dataclass(kw_only=True)
class Query:
    """A query to a benchmark."""

    config: Config
    """ The config to evaluate """

    fidelity: tuple[str, int | float] | Mapping[str, int | float] | None
    """ What fidelity to evaluate at """

    optimizer_info: Any | None = None
    """Any optimizer specific info required across ask and tell"""

    request_trajectory: bool | tuple[int, int] | tuple[float | float] = False
    """Whether the optimizer requires a trajectory curve for multi-fidelity optimization.

    If a specific range is requested, then a tuple can be provided.
    """

    query_id: str = field(init=False)
    """The id of the query.

    This includes information about the config id and fidelity.
    """

    def __post_init__(self) -> None:
        match self.fidelity:
            case None:
                self.query_id = self.config.id
            case (name, value):
                self.query_id = f"{self.config.id}-{name}={value}"
            case Mapping():
                self.query_id = (
                    f"{self.config.id}-{'-'.join(f'{k}={v}' for k, v in self.fidelity.items())}"
                )
            case _:
                raise NotImplementedError("Unexpected fidelity type")

        if self.request_trajectory:
            match self.fidelity:
                case None:
                    raise ValueError("Learning curve requested but no fidelity provided")
                case tuple():
                    pass
                case Mapping():
                    raise ValueError(
                        "Learning curve requested but more than a single fidelity provided"
                    )

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
        match self.fidelity:
            case None:
                return self.config.series()
            case (name, value):
                return pd.Series(
                    {
                        **self.config.series(),
                        "query.id": self.query_id,
                        f"query.fidelity.{name}": value,
                    }
                )
            case Mapping():
                return pd.Series(
                    {
                        **self.config.series(),
                        "query.id": self.query_id,
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

    def with_fidelity(
        self,
        fidelity: tuple[str, int | float] | Mapping[str, int | float] | None,
    ) -> Query:
        """Create a new query with a different fidelity."""
        return replace(self, fidelity=fidelity)
