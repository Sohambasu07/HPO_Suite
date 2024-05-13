from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hpo_glue.problem import Problem
    from hpo_glue.query import Query
    from hpo_glue.result import Result


class Optimizer(ABC):
    """Defines the common interface for Optimizers."""

    name: ClassVar[str]
    """The name of the optimizer"""

    supports_multiobjective: ClassVar[bool] = False
    """Whether the optimizer supports multi-objective"""

    supports_multifidelity: ClassVar[bool] = False
    """Whether the optimizer supports multi-fidelity"""

    supports_manyfidelity: ClassVar[bool] = False
    """Whether the optimizer supports many fidelities"""

    supports_tabular: ClassVar[bool] = False
    """Whether the optimizer supports tabular benchmarks"""

    @abstractmethod
    def __init__(
        self,
        *,
        problem: Problem,
        working_directory: Path,
        **optimizer_kwargs: Any,
    ) -> None:
        """Initialize the optimizer.

        Args:
            problem: The problem to optimize over
            working_directory: The directory to save the optimizer's state
            optimizer_kwargs: Any additional hyperparameters for the optimizer
        """

    @abstractmethod
    def ask(self) -> Query:
        """Ask the optimizer for a new config to evaluate."""

    @abstractmethod
    def tell(self, result: Result) -> None:
        """Tell the optimizer the result of the query."""
