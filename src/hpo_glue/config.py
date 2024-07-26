from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class Config:
    """A configuration to evaluate."""

    config_id: str
    """Some unique identifier"""

    values: dict[str, Any] | None
    """The actual config values to evaluate.

    In the case this config was deserialized, it will likely be `None`.
    """

    def to_tuple(self, precision) -> tuple:
        return tuple(
            self.set_precision(
                self.values, 
                precision
            ).values()
        )

    def set_precision(self, values: dict, precision: int = 12) -> None:
        for key, value in values.items():
            if isinstance(value, float):
                self.values[key] = np.round(value, precision)