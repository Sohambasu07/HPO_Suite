from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import ConfigurationSpace

from hpo_glue.config import Config
from hpo_glue.optimizer import Optimizer
from hpo_glue.query import Query

if TYPE_CHECKING:
    from hpo_glue.problem import Problem
    from hpo_glue.result import Result

if TYPE_CHECKING:
    from hpo_glue.problem import Problem
    from hpo_glue.result import Result


class RandomSearch(Optimizer):
    """Random Search Optimizer."""

    name = "RandomSearch"
    supports_multifidelity = True
    supports_manyfidelity = True
    supports_tabular = True
    supports_multiobjective = True

    def __init__(
        self,
        *,
        problem: Problem,
        working_directory: Path,  # noqa: ARG002
    ):
        """Create a Random Search Optimizer instance for a given problem statement."""
        self.problem = problem
        self._counter = 0
        self.rng = np.random.default_rng(problem.seed)

        match problem.config_space:
            case ConfigurationSpace():
                self.config_space = copy.deepcopy(problem.config_space)
                if problem.seed is None:
                    _config_space_seed = int(self.rng.integers(2**31 - 1))
                    self.config_space.seed(_config_space_seed)
                else:
                    self.config_space.seed(problem.seed)
            case list():
                self.config_space = problem.config_space
            case _:
                raise TypeError("Config space must be a ConfigSpace or a list of Configs")

    def ask(self) -> Query:
        """Ask the optimizer for a new config to evaluate."""
        self._counter += 1
        # Randomly sampling from fidelity space for multifidelity
        match self.problem.fidelity_space:
            case None:
                fidelity = None
            case list():
                fidelity = self.problem.fidelity_space[-1]
            case dict():
                fidelity = {k: v[-1] for k, v in self.problem.fidelity_space.items()}
            case _:
                raise TypeError("Fidelity space must be a list of elements or None")

        # We are dealing with a tabular benchmark
        match self.config_space:
            case ConfigurationSpace():
                config = Config(
                    id=str(self._counter),
                    values=self.config_space.sample_configuration().get_dictionary(),
                )
            case list():
                index = self.rng.integers(len(self.config_space))
                config = self.config_space[int(index)]
            case _:
                raise TypeError("Config space must be a ConfigSpace or a list of Configs")

        return Query(config=config, fidelity=fidelity)

    def tell(self, result: Result) -> None:
        """Tell the optimizer the result of the query."""
        # NOTE(eddiebergman): Random search does nothing with the result
