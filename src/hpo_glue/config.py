from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Config(Mapping[str, Any]):
    """A configuration to evaluate."""

    id: str
    """Some unique identifier"""

    values: dict[str, Any]
    """ The actual config values to evaluate """

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def series(self) -> pd.Series:
        """Return the config as a pandas Series."""
        return pd.Series(
            {
                **{f"config.{k}": v for k, v in self.values.items()},
                "config.id": self.id,
            }
        )

    @classmethod
    def from_series(cls, series: pd.Series) -> Config:
        """Create a Config from a pandas Series."""
        return Config(
            id=series["config.id"],  # type: ignore
            values={k.split("config.")[1]: v for k, v in series.filter(like="config.").items()},  # type: ignore
        )
