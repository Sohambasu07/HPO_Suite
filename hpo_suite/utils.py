from __future__ import annotations

from typing import Any

from hpo_glue import Problem
from hpo_glue.budgets import CostBudget, TrialBudget


class GlueWrapperFunctions:
    """A collection of wrapper functions around certain hpo_glue methods."""

    @staticmethod
    def problem_to_dict(problem: Problem) -> dict[str, Any]:
        """Convert a Problem instance to a dictionary."""
        return problem.to_dict()

    @staticmethod
    def problem_from_dict(data: dict[str, Any]) -> Problem:
        """Convert a dictionary to a Problem instance."""
        from hpo_suite.benchmarks import BENCHMARKS

        return Problem.from_dict(
            data=data,
            benchmarks_dict=BENCHMARKS
        )

