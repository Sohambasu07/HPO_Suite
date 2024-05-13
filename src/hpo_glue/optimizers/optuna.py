from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import Any, override

import ConfigSpace as CS  # noqa: N817
import optuna
from optuna.distributions import (
    CategoricalDistribution as Cat,
    FloatDistribution as Float,
    IntDistribution as Int,
)
from optuna.samplers import NSGAIISampler, TPESampler

from hpo_glue.config import Config
from hpo_glue.optimizer import Optimizer
from hpo_glue.query import Query

if TYPE_CHECKING:
    from hpo_glue.problem import Problem
    from hpo_glue.result import Result

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OptunaOptimizer(Optimizer):
    """The Optuna Optimizer.

    # TODO: Document me
    """

    name = "Optuna"
    supports_multifidelity = False
    supports_manyfidelity = False
    supports_tabular = False
    supports_multiobjective = True

    def __init__(self, *, problem: Problem, working_directory: Path, **kwargs: Any):
        """Create an Optuna Optimizer instance for a given problem statement."""
        if isinstance(problem.config_space, list):
            raise NotImplementedError("# TODO: Tabular not yet implemented for Optuna!")

        match problem.objective:
            case str():
                direction = "minimize" if problem.minimize else "maximize"
                directions = None
                sampler = TPESampler(seed=problem.seed, **kwargs)
            case list():
                assert isinstance(problem.minimize, list)
                direction = None
                directions = ["minimize" if mini else "maximize" for mini in problem.minimize]
                sampler = NSGAIISampler(seed=problem.seed, **kwargs)
            case _:
                raise ValueError("Objective must be a string or a list of strings!")

        self.problem = problem
        self.working_directory = working_directory
        self.optimizer = optuna.create_study(
            sampler=sampler,
            storage=None,
            pruner=None,  # TODO(eddiebergman): Figure out how to use this for MF
            study_name=self.problem.name,
            load_if_exists=False,
            direction=direction,
            directions=directions,
        )
        self._distributions = _configspace_to_optuna_distributions(problem.config_space)
        self._trial_lookup: dict[str, optuna.trial.Trial] = {}

    @override
    def ask(self) -> Query:
        trial = self.optimizer.ask(self._distributions)
        # TODO(eddiebergman): Not sure if just using
        # trial.number is enough in MF setting
        name = f"trial_{trial.number}"
        self._trial_lookup[name] = trial
        return Query(
            config=Config(id=name, values=trial.params),
            fidelity=None,
        )

    @override
    def tell(self, result: Result) -> None:
        original_trial = self._trial_lookup.pop(result.config.id)
        match self.problem.objective:
            case str():
                _values = result.result[self.problem.objective]
            case list():
                _values = [result.result[o] for o in self.problem.objective]
            case _:
                raise TypeError("Objective must be a string or a list of strings!")

        self.optimizer.tell(
            trial=original_trial,
            values=_values,
            state=optuna.trial.TrialState.COMPLETE,
            skip_if_finished=False,
        )


def _configspace_to_optuna_distributions(config_space: CS.ConfigurationSpace) -> dict:  # noqa: C901
    if not isinstance(config_space, CS.ConfigurationSpace):
        raise ValueError("Need search space of type ConfigSpace.ConfigurationSpace}")

    if len(config_space.get_conditions()) > 0:
        raise NotImplementedError("Conditions are not yet supported!")

    if len(config_space.get_forbiddens()) > 0:
        raise NotImplementedError("Forbiddens are not yet supported!")

    optuna_space = {}
    for hp in config_space.get_hyperparameters():
        if isinstance(hp, CS.UniformIntegerHyperparameter):
            optuna_space[hp.name] = Int(hp.lower, hp.upper, log=hp.log)
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            optuna_space[hp.name] = Float(hp.lower, hp.upper, log=hp.log)
        elif isinstance(hp, CS.CategoricalHyperparameter):
            if hp.weights is not None:
                raise NotImplementedError("Weights on categoricals are not yet supported!")
            optuna_space[hp.name] = Cat(hp.choices)
        elif isinstance(hp, CS.Constant):
            optuna_space[hp.name] = Cat([hp.value])
        elif isinstance(hp, CS.OrdinalHyperparameter):
            raise NotImplementedError("Ordinal hyperparameters are not yet supported!")
        else:
            raise ValueError("Unrecognized type of hyperparameter in ConfigSpace!")

    return optuna_space
