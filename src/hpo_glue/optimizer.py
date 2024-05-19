from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from hpo_glue.config import Config
    from hpo_glue.problem import Problem
    from hpo_glue.query import Query
    from hpo_glue.result import Result


class Optimizer(ABC):
    """Defines the common interface for Optimizers."""

    name: ClassVar[str]
    """The name of the optimizer"""

    support: ClassVar[Problem.Support]
    """What kind of problems the optimizer supports"""

    minimize_only: ClassVar[bool]
    """Whether the optimizer only supports minimization, in which
    case the report results to the optimizer will be negated for it.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        problem: Problem,
        config_space: list[Config] | ConfigurationSpace,
        working_directory: Path,
        **optimizer_kwargs: Any,
    ) -> None:
        """Initialize the optimizer.

        Args:
            problem: The problem to optimize over
            config_space: The configuration space to optimize over
            working_directory: The directory to save the optimizer's state
            optimizer_kwargs: Any additional hyperparameters for the optimizer
        """

    @abstractmethod
    def ask(self) -> Query:
        """Ask the optimizer for a new config to evaluate."""

    @abstractmethod
    def tell(self, result: Result) -> None:
        """Tell the optimizer the result of the query."""
