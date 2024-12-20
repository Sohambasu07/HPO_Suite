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

opt_list = sorted(ng.optimizers.registry.keys())
ext_opts = {
    "Hyperopt": ng.optimization.optimizerlib.NGOpt13,
    "CMA-ES": ng.optimization.optimizerlib.ParametrizedCMA,
    "bayes_opt": ng.optimization.optimizerlib.ParametrizedBO,
    "DE": ng.families.DifferentialEvolution,
    "EvolutionStrategy": ng.families.EvolutionStrategy,
}


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
            
        self.problem = problem
        self.working_directory = working_directory
            
        optimizer_name = kwargs.get("optimizer", "NGOpt10")
        if kwargs["optimizer"] not in ext_opts or kwargs["optimizer"] not in opt_list:
            raise ValueError(f"Unknown optimizer: {kwargs['optimizer']}")

        self.optimizer: NGOptimizer | ConfNGOptimizer
        match problem.objective:
            case (_, objective):
                if optimizer_name in ext_opts:
                    if optimizer_name == "Hyperopt":
                        ng_opt = ext_opts["Hyperopt"]
                    else:
                        ng_opt = ext_opts[optimizer_name]()
                    ng_opt = ng_opt(
                        parametrization=self._parametrization,
                        budget=self.problem.budget, # TODO: Check this
                        num_workers=1,
                    )
                else:
                    ng_opt = opt_list[optimizer_name](
                        parametrization=self._parametrization,
                        budget=self.problem.budget, # TODO: Check this
                        num_workers=1,
                    )
            case Mapping():
                raise NotImplementedError("# TODO: Multi-objective not yet implemented for Nevergrad!")
            case _:
                raise ValueError("Objective must be a string or a list of strings!")

        self.history: dict[str, tuple[ng.p.Parameter, float | list[float] | None]] = {}
        self.counter = 0

    @override
    def ask(self) -> Query:
        match self.problem.fidelity:
            case None:
                config: parameter.Parameter = self.optimizer.ask()
                name = f"{self.counter}_{config.value}_{self.nevergrad_cfg.seed}"
                self.history[name] = (config, None)
                self.counter += 1
                return Query(
                    config=Config(
                        config_id=name,
                        values=config.value),
                        # allow_inactive_with_values=True,
                    fidelity=None,
                    optimizer_info=config,
                )
            case tuple():
                # TODO(eddiebergman): Not sure if just using
                # trial.number is enough in MF setting
                raise NotImplementedError("# TODO: Fidelity-aware not available in Nevergrad!")
            case Mapping():
                raise NotImplementedError("# TODO: Fidelity-aware not available in Nevergad!")
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
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for Nevergrad!")
            case Mapping():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for Nevergrad!")
            case _:
                raise TypeError("Cost must be None or a mapping!")

        assert isinstance(result.query.optimizer_info, parameter.Parameter)
        self.optimizer.tell(
            result.query.optimizer_info,
            _values,
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
