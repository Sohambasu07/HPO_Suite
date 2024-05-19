from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeAlias
from typing_extensions import Self

if TYPE_CHECKING:
    from hpo_glue.problem import Problem
    from hpo_glue.result import Result


@dataclass
class TrialBudget:
    """A budget for the number of trials to run."""

    budget: int | float
    """Total amount of budget allowed for the optimizer for this problem.

    How this is interpreted is depending on fidelity type.

    If the problem **does not** include a fidelity, then this is assumed
    to be a black-box problem, and each fully complete trial counts as
    1 towards the budget.

    If the problem **does** include a **single** fidelity, then the fidelity
    at which the trial was evaluated is taken as a fraction of the full fidelity
    and added to the used budget. For example, 40 epochs of a fidelity that
    maxes out at 100 epochs would count as 0.4 towards the budget.

    If the problem **does** include **many** fidelities, then the fraction as calculated
    for a single fidelity is applied to all fidelities, and then summed, normalized by
    the total number of fidelities. For example, 40 epochs of a fidelity that maxes out
    at 100 epochs and data percentage of 0.6 of a fidelity that maxes out at 1.0 would
    equate to (0.4 + 0.6) / 2 = 0.5 towards the budget.
    """

    used_budget: float = 0.0

    def calculate_used_budget(self, *, result: Result, problem: Problem) -> float:
        """Calculate the used budget for a given result.

        Args:
            result: The result of the trial.
            problem: The original problem statement.

        Returns:
            The amount of budget used for this result.
        """
        match problem.fidelity:
            case None:
                return 1
            case (_, fidelity_desc):
                assert isinstance(result.fidelity, int | float)
                return fidelity_desc.normalize(result.fidelity)
            case Mapping():
                assert problem.benchmark.fidelities is not None
                assert isinstance(result.fidelity, dict)

                normed_fidelities = []
                n_fidelities = len(result.fidelity)
                for k, v in result.fidelity.items():
                    fidelity_desc = problem.benchmark.fidelities[k]
                    norm_fidelity = fidelity_desc.normalize(v)
                    normed_fidelities.append(norm_fidelity)

                return sum(normed_fidelities) / n_fidelities
            case _:
                raise TypeError("Fidelity must be None, str, or list[str]")

    def update(self, *, result: Result, problem: Problem) -> None:
        """Update the budget with the result of a trial.

        Args:
            result: The result of the trial.
            problem: The original problem statement.
        """
        self.used_budget += self.calculate_used_budget(result=result, problem=problem)

    def should_stop(self) -> bool:
        """Check if the budget has been used up."""
        return self.used_budget >= self.budget

    @property
    def path_str(self) -> str:
        """Return a string representation of the budget."""
        clsname = self.__class__.__name__
        return f"{clsname}={self.budget}"

    def clone(self) -> Self:
        """Return a clone of the budget."""
        return replace(self)


@dataclass(kw_only=True)
class CostBudget:
    """A budget for the cost of the trials to run."""

    budget: int | float

    def __post_init__(self):
        raise NotImplementedError("Cost budgets not yet supported")

    def update(self, *, result: Result, problem: Problem) -> None:
        """Update the budget with the result of a trial.

        Args:
            result: The result of the trial.
            problem: The original problem statement.
        """
        raise NotImplementedError("Cost budgets not yet supported")

    def should_stop(self) -> bool:
        """Check if the budget has been used up."""
        raise NotImplementedError("Cost budgets not yet supported")

    def calculate_used_budget(self, *, result: Result, problem: Problem) -> float:
        """Calculate the used budget for a given result.

        Args:
            result: The result of the trial.
            problem: The original problem statement.

        Returns:
            The amount of budget used for this result.
        """
        raise NotImplementedError("Cost budgets not yet supported")

    @property
    def path_str(self) -> str:
        """Return a string representation of the budget."""
        clsname = self.__class__.__name__
        return f"{clsname}={self.budget}"

    def clone(self) -> Self:
        """Return a clone of the budget."""
        return replace(self)


BudgetType: TypeAlias = TrialBudget | CostBudget
