from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hpo_glue.problem import Problem


@dataclass
class Experiment:
    """An experiment to run."""

    name: str
    """The name of the Experiment"""

    path: Path
    """The path to the Experiment"""

    problems: list[Problem]
    """The list of Runs inside the Experiment"""

    @dataclass
    class Report:
        """The report of an Experiment run."""

        experiment: Experiment
        """The Experiment that was run"""

        reports: list[Problem.Report]
        """The list of Problem Reports inside the Experiment"""
