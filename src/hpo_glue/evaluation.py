from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hpo_glue.result import Result


@dataclass(kw_only=True)
class Evaluation:
    """The evaluation of a benchmark."""

    ask_duration: float
    """The time it took to ask the optimizer."""

    tell_duration: float
    """The time it took to tell the optimizer the result."""

    result: Result
    """The id of the config."""

    continued_from: Result | None
    """The previous result of a query."""

    def series(self) -> pd.Series:
        """Return the result as a pandas Series."""
        return pd.Series(
            {
                **self.query.series(),
                **{f"result.{k}": v for k, v in self.result.items()},
            }
        )

    @classmethod
    def from_series(cls, series: pd.Series) -> Result:
        """Create a Result from a pandas Series."""
        return Result(
            query=Query.from_series(series.filter(like="query.")),
            full_results=series.filter(like="result.").to_dict(),
        )
