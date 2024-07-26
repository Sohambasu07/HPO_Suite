from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import Any, override

import ConfigSpace as CS  # noqa: N817

from hpo_glue.config import Config
from hpo_glue.optimizer import Optimizer
from hpo_glue.problem import Problem
from hpo_glue.query import Query
import nevergrad as ng

if TYPE_CHECKING:
    from nevergrad.optimization.base import (
        ConfiguredOptimizer as ConfNGOptimizer,
        Optimizer as NGOptimizer,
    )
    from nevergrad.parametrization import parameter

    from hpo_glue.result import Result

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NevergradOptimizer(Optimizer):
    """The Nevergrad Optimizer.

    # TODO: Document me
    """

    name = "Nevergrad"
    support = Problem.Support(
        fidelities=(None,),  # TODO: Implement fidelity support
        objectives=("single", "many"),
        cost_awareness=(None,),
        tabular=False,
    )

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        config_space: list[Config] | CS.ConfigurationSpace,
        **kwargs: Any,
    ) -> None:
        """Create a Nevergrad Optimizer instance for a given problem statement."""

        self._parametrization: dict[str, parameter.Parameter]
        match config_space:
            case CS.ConfigurationSpace():
                self._parametrization = _configspace_to_nevergrad_space(config_space)
            case list():
                raise NotImplementedError("# TODO: Tabular not yet implemented for Nevergrad!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        self.optimizer: NGOptimizer | ConfNGOptimizer
        match problem.objective:
            case (_, objective):
                self.optimizer = optuna.create_study(
                    sampler=TPESampler(seed=seed, **kwargs),
                    storage=None,
                    pruner=None,  # TODO(eddiebergman): Figure out how to use this for MF
                    study_name=f"{problem.name}-{seed}",
                    load_if_exists=False,
                    direction="minimize" if objective.minimize else "maximize",
                )
            case Mapping():
                self.optimizer = optuna.create_study(
                    sampler=NSGAIISampler(seed=seed, **kwargs),
                    storage=None,
                    pruner=None,  # TODO(eddiebergman): Figure out how to use this for MF
                    study_name=f"{problem.name}-{seed}",
                    load_if_exists=False,
                    directions=[
                        "minimize" if obj.minimize else "maximize"
                        for obj in problem.objective.values()
                    ],
                )
            case _:
                raise ValueError("Objective must be a string or a list of strings!")

        self.problem = problem
        self.working_directory = working_directory
        self._trial_lookup: dict[str, optuna.trial.Trial] = {}

    @override
    def ask(self) -> Query:
        match self.problem.fidelity:
            case None:
                trial = self.optimizer.ask(self._distributions)
                name = f"trial_{trial.number}"
                return Query(
                    config=Config(config_id=name, values=trial.params),
                    fidelity=None,
                    optimizer_info=trial,
                )
            case tuple():
                # TODO(eddiebergman): Not sure if just using
                # trial.number is enough in MF setting
                raise NotImplementedError("# TODO: Fidelity-aware not yet implemented for Optuna!")
            case Mapping():
                raise NotImplementedError("# TODO: Fidelity-aware not yet implemented for Optuna!")
            case _:
                raise TypeError("Fidelity must be None or a tuple!")

    @override
    def tell(self, result: Result) -> None:
        match self.problem.objective:
            case (name, _):
                _values = result.values[name]
            case Mapping():
                _values = [result.values[key] for key in self.problem.objective]
            case _:
                raise TypeError("Objective must be a string or a list of strings!")

        match self.problem.cost:
            case None:
                pass
            case tuple():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for Optuna!")
            case Mapping():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for Optuna!")
            case _:
                raise TypeError("Cost must be None or a mapping!")

        assert isinstance(result.query.optimizer_info, optuna.trial.Trial)
        self.optimizer.tell(
            trial=result.query.optimizer_info,
            values=_values,
            state=optuna.trial.TrialState.COMPLETE,
            skip_if_finished=False,
        )


def _configspace_to_nevergrad_space(
    config_space: CS.ConfigurationSpace,
) -> dict[str, ng.p.Instrumentation]:

    if len(config_space.get_conditions()) > 0:
        raise NotImplementedError("Conditions are not yet supported!")

    if len(config_space.get_forbiddens()) > 0:
        raise NotImplementedError("Forbiddens are not yet supported!")

    ng_space = ng.p.Dict()
    for hp in config_space.get_hyperparameters():
        match hp:
            case CS.UniformIntegerHyperparameter():
                if hp.log:
                    ng_space[hp.name] = ng.p.Log(lower=hp.lower, upper=hp.upper).set_integer_casting()
                else:
                    ng_space[hp.name] = ng.p.Scalar(lower=hp.lower, upper=hp.upper).set_integer_casting()
            case CS.UniformFloatHyperparameter():
                if hp.log:
                    ng_space[hp.name] = ng.p.Log(lower=hp.lower, upper=hp.upper)
                else:
                    ng_space[hp.name] = ng.p.Scalar(lower=hp.lower, upper=hp.upper)
            case CS.CategoricalHyperparameter():
                if hp.weights is not None:
                    raise NotImplementedError("Weights on categoricals are not yet supported!")
                ng_space[hp.name] = ng.p.Choice(hp.choices)
            case CS.Constant():
                ng_space[hp.name] = ng.p.Choice([hp.value])
            case CS.OrdinalHyperparameter():
                ng_space[hp.name] = ng.p.TransitionChoice(hp.sequence)
            case _:
                raise ValueError("Unrecognized type of hyperparameter in ConfigSpace!")

    return ng_space
