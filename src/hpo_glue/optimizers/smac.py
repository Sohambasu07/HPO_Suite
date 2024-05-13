from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from smac import (
    BlackBoxFacade as BOFacade,
    HyperbandFacade as HBFacade,
    Scenario,
)
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runhistory.enumerations import StatusType

from hpo_glue.config import Config
from hpo_glue.optimizer import Optimizer
from hpo_glue.query import Query

if TYPE_CHECKING:
    from smac.facade import AbstractFacade

    from hpo_glue.problem import Problem
    from hpo_glue.result import Result


class SMAC_Optimizer(Optimizer):
    """Default SMAC Optimizer.

    # TODO: Document me
    """

    name = "SMAC-base"
    supports_manyfidelity = False
    supports_tabular = False

    def __init__(
        self,
        *,
        problem: Problem,
        working_directory: Path,
        optimizer: AbstractFacade,
    ):
        """Create a SMAC Optimizer instance for a given problem statement.

        Args:
            problem: Problem statement.
            working_directory: Working directory to store SMAC run.
            optimizer: SMAC optimizer instance.
        """
        self.problem = problem
        self.working_directory = working_directory
        self.optimizer = optimizer
        self._trial_lookup: dict[str, TrialInfo] = {}

    def ask(self) -> Query:
        """Ask SMAC for a new config to evaluate."""
        smac_info = self.optimizer.ask()
        config = smac_info.config
        fidelity = smac_info.budget
        instance = smac_info.instance
        seed = smac_info.seed
        config_id = self.optimizer.intensifier.runhistory.config_ids[config]
        raw_config = config.get_dictionary()

        name = f"{config_id=}_{seed=}_{fidelity=}_{instance=}"
        self._trial_lookup[name] = smac_info

        if isinstance(fidelity, float):
            fidelity = round(fidelity)

        return Query(
            config=Config(id=name, values=raw_config),
            fidelity=fidelity,
        )

    def tell(self, result: Result) -> None:
        """Tell SMAC the result of the query."""
        # TODO(eddiebergman): Normalize costs if we have a metric definition.
        objectives = self.problem.objective
        minimize = self.problem.minimize
        match objectives:
            case str():
                _cost = result.result[objectives]
                assert isinstance(minimize, bool)
                if self.problem.minimize is False:
                    _cost = -_cost
            case list():
                assert isinstance(minimize, list)
                _cost = [result.result[o] for o in self.problem.objective]
                _cost = [-c if self.problem.minimize else c for c in _cost]
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        original_info = self._trial_lookup.pop(result.config.id)
        self.optimizer.tell(
            original_info,
            TrialValue(
                cost=_cost,
                time=0.0,
                starttime=0.0,
                endtime=0.0,
                status=StatusType.SUCCESS,
                additional_info={},
            ),
            save=True,
        )


class SMAC_BO(SMAC_Optimizer):
    """SMAC Bayesian Optimization."""

    name = "SMAC_BO"
    supports_manyfidelity = False
    supports_tabular = False
    supports_multifidelity = False  # TODO(eddiebergman): Verify this
    supports_multiobjective = True

    def __init__(
        self,
        *,
        problem: Problem,
        working_directory: Path,
        xi: float = 0.0,
    ):
        """Create a SMAC BO Optimizer instance for a given problem statement.

        Args:
            problem: Problem statement.
            working_directory: Working directory to store SMAC run.
            seed: Random seed. Defaults to None.
            xi: Exploration-exploitation trade-off parameter. Defaults to 0.0.
        """
        if isinstance(problem.config_space, list):
            raise ValueError("SMAC does not support tabular benchmarks!")

        working_directory.mkdir(parents=True, exist_ok=True)

        if problem.fidelity_space is not None:
            min_budget = problem.fidelity_space[0]
            max_budget = problem.fidelity_space[-1]
        else:
            min_budget = None
            max_budget = None

        if problem.seed is None:
            seed = int(np.random.default_rng().integers(2**31 - 1))
        else:
            seed = problem.seed

        facade = BOFacade
        scenario = Scenario(
            configspace=problem.config_space,
            deterministic=True,
            objectives=problem.objective,
            n_trials=0,
            seed=seed,
            output_directory=working_directory / "smac-output",
            min_budget=min_budget,
            max_budget=max_budget,
        )
        super().__init__(
            problem=problem,
            working_directory=working_directory,
            optimizer=facade(
                scenario=scenario,
                target_function=lambda *_: None,
                intensifier=facade.get_intensifier(scenario),
                acquisition_function=facade.get_acquisition_function(scenario, xi=xi),
                overwrite=True,
            ),
        )


class SMAC_Hyperband(SMAC_Optimizer):
    """SMAC Hyperband."""

    name = "SMAC-Hyperband"
    supports_manyfidelity = False
    supports_tabular = False
    supports_multifidelity = True
    supports_multiobjective = True  # TODO(eddiebergman): Verify this

    def __init__(
        self,
        *,
        problem: Problem,
        working_directory: Path,
        eta: int = 3,
    ):
        """Create a SMAC Hyperband Optimizer instance for a given problem statement.

        Args:
            problem: Problem statement.
            working_directory: Working directory to store SMAC run.
            seed: Random seed. Defaults to None.
            eta: Hyperband eta parameter. Defaults to 3.
        """
        if isinstance(problem.config_space, list):
            raise ValueError("SMAC does not support tabular benchmarks!")

        if problem.fidelity_space is None:
            raise ValueError("SMAC Hyperband requires a fidelity space!")

        working_directory.mkdir(parents=True, exist_ok=True)
        if problem.fidelity_space is not None:
            min_budget = problem.fidelity_space[0]
            max_budget = problem.fidelity_space[-1]
        else:
            min_budget = None
            max_budget = None

        if problem.seed is None:
            seed = int(np.random.default_rng().integers(2**31 - 1))
        else:
            seed = problem.seed

        facade = HBFacade
        scenario = Scenario(
            configspace=problem.config_space,
            deterministic=True,
            objectives=problem.objective,
            n_trials=0,
            seed=seed,
            output_directory=working_directory / "smac-output",
            min_budget=min_budget,
            max_budget=max_budget,
        )
        super().__init__(
            problem=problem,
            working_directory=working_directory,
            optimizer=facade(
                scenario=scenario,
                target_function=lambda *_: None,
                intensifier=facade.get_intensifier(scenario, eta=eta),
                overwrite=True,
            ),
        )
