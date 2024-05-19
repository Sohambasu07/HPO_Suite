from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from hpo_glue.evaluation import Evaluation


@dataclass
class History:
    """Collects the history of a run."""

    evaluations: list[Evaluation] = field(default_factory=list)

    def add(self, result: Evaluation) -> None:
        """Add a result to the history."""
        self.evaluations.append(result)

    def df(self) -> pd.DataFrame:
        """Return the history as a pandas DataFrame."""
        return pd.concat([res.series() for res in self.evaluations], axis=1).T.convert_dtypes()

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> History:
        return History([Evaluation.from_series(row) for _, row in df.iterrows()])

    def _save(
        self,
        report: pd.DataFrame,
        runsave_dir: Path,
        benchmark_name: str,
        optimizer_name: str,
        optimizer_hyperparameters: dict[str, Any],
        seed: int | None,
    ) -> None:
        """Save the history of the run and along with some metadata."""
        _optimizer_hyperparameters = (
            optimizer_hyperparameters if any(optimizer_hyperparameters) else ""
        )
        filename = f"{benchmark_name}_{optimizer_name}_{_optimizer_hyperparameters}"
        filesave_dir = runsave_dir / benchmark_name / optimizer_name / str(seed)
        filesave_dir.mkdir(parents=True, exist_ok=True)
        report.convert_dtypes().to_parquet(
            filesave_dir / f"report_{filename}.parquet",
            index=False,
        )
