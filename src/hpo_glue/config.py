from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math


@dataclass
class Config:
    """A configuration to evaluate."""

    config_id: str
    """Some unique identifier"""

    values: dict[str, Any] | None
    """The actual config values to evaluate.

    In the case this config was deserialized, it will likely be `None`.
    """

    def to_tuple(self) -> tuple:
        return tuple(self.values.values()) #TODO: Set precision here itself, default to 12

    def set_precision(self, precision: int | None = None) -> None:
        if precision is None:
            return
        for key, value in self.values.items():
            if isinstance(value, float):
                self.values[key] = math.floor(value * 10 ** precision) / 10 ** precision #TODO: Use np.round