from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hpo_glue.experiment import Experiment
    from hpo_glue.problem import Problem

logger = logging.getLogger(__name__)


class GLUE:
    """The main class for running GLUE experiments."""

    def run_experiment(self, exp: Experiment, root_dir: Path) -> Iterator[Problem.Report]:
        """Runs an experiment, returning a report."""
        # Creating current Experiment directory

        # Running the experiment using GLUE.run()
        total_problem_count = len(exp.problems)
        exp_dir = root_dir / exp.name

        for i, problem in enumerate(exp.problems, start=1):
            logger.info(f"Executing Run: {problem.name} ({i}/{total_problem_count})\n")
            report = self.run(problem=problem, root_dir=exp_dir)
            yield report
