from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    """A configuration to evaluate."""

    config_id: str
    """Some unique identifier"""

    values: dict[str, Any] | None
    """The actual config values to evaluate.

    In the case this config was deserialized, it will likely be `None`.
    """
