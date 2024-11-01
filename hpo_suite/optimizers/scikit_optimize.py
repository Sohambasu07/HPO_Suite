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

if TYPE_CHECKING:
    from skopt.space.space import Space
    from hpo_glue.result import Result

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


base_estimators = ["GP", "RF", "ET", "GBRT"]
acq_funcs = ["LCB", "EI", "PI", "gp_hedge"]
acq_optimizers = ["sampling", "lbfgs", "auto"]

class SkoptOptimizer(Optimizer):
    """The Scikit_Optimize Optimizer.

    # TODO: Document me
    """

    name = "Scikit_Optimize"
    support = Problem.Support(
        fidelities=(None,),  # NOTE: Skopt does not support multi-fidelity optimization
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
        """Create an Skopt Optimizer instance for a given problem statement."""
        import skopt

        self._space: list[Space]
        match config_space:
            case CS.ConfigurationSpace():
                self._space = _configspace_to_skopt_space(config_space)
            case list():
                raise NotImplementedError("# TODO: Tabular not yet implemented for Scikit_Optimize!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")
            
        base_estimator = kwargs.get("base_estimator", "GP")
        assert base_estimator in base_estimators, f"base_estimator must be one of {base_estimators}!"

        acq_func = kwargs.get("acq_func", "gp_hedge")
        assert acq_func in acq_funcs, f"acq_func must be one of {acq_funcs}!"

        acq_optimizer = kwargs.get("acq_optimizer", "auto")
        assert acq_optimizer in acq_optimizers, f"acq_optimizer must be one of {acq_optimizers}!"

        self.optimizer: skopt.optimizer.Optimizer
        match problem.objective:
            case (_, objective):
                self.optimizer = skopt.optimizer.Optimizer(
                    dimensions=self._space,
                    base_estimator=base_estimator,
                    acq_func=acq_func,
                    acq_optimizer=acq_optimizer,
                    random_state=seed,
                    n_initial_points=5
                )
            case Mapping():
                raise NotImplementedError("Multiobjective not supported by Scikit_Optimize!")
            case _:
                raise ValueError("Objective must be a string or a list of strings!")

        self.problem = problem
        self.working_directory = working_directory
        self.trial_counter = 0

    @override
    def ask(self) -> Query:
        match self.problem.fidelity:
            case None:
                config = self.optimizer.ask()
                config_values = {hp.name: value for hp, value in zip(self.configspace.get_hyperparameters(), config, strict=False)}
                assert list(config_values.keys()) == list(self.configspace.get_hyperparameter_names())
                assert list(config_values.keys()) == [hp.name for hp in self.skopt_space]
                name = f"trial_{self.trial_counter}"
                self.trial_counter += 1
                return Query(
                    config=Config(config_id=name, values=config_values),
                    fidelity=None,
                    optimizer_info=None,
                )
            case tuple():
                raise NotImplementedError("# TODO: Fidelity-aware not yet implemented for Scikit_Optimize!")
            case Mapping():
                raise NotImplementedError("# TODO: Fidelity-aware not yet implemented for Scikit_Optimize!")
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
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for Scikit_Optimize!")
            case Mapping():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for Scikit_Optimize!")
            case _:
                raise TypeError("Cost must be None or a mapping!")

        _ = self.solver.tell(
            list(result.query.config.values), 
            _values
        )


def _configspace_to_skopt_space(
    config_space: CS.ConfigurationSpace,
) -> dict[str, Space]:
    import numpy as np
    from skopt.space.space import Categorical, Integer, Real

    if len(config_space.get_conditions()) > 0:
        raise NotImplementedError("Conditions are not yet supported!")

    if len(config_space.get_forbiddens()) > 0:
        raise NotImplementedError("Forbiddens are not yet supported!")

    skopt_space: list[float] = []
    for hp in config_space.get_hyperparameters():
        match hp:
            case CS.UniformIntegerHyperparameter() if hp.log:
                skopt_space.append(Integer(hp.lower, hp.upper, name=hp.name, log=hp.log))
            case CS.UniformIntegerHyperparameter():
                skopt_space.append(Integer(hp.lower, hp.upper, name=hp.name))
            case CS.UniformFloatHyperparameter() if hp.log:
                skopt_space.append(Real(hp.lower, hp.upper, name=hp.name, log=hp.log))
            case CS.UniformFloatHyperparameter():
                skopt_space.append(Real(hp.lower, hp.upper, name=hp.name))
            case CS.CategoricalHyperparameter() if hp.weights is not None:
                weights = np.asarray(hp.weights) / np.sum(hp.weights)
                skopt_space.append(Categorical(hp.choices, name=hp.name, prior=weights))
            case CS.CategoricalHyperparameter():
                skopt_space.append(Categorical(hp.choices, name=hp.name))
            case CS.Constant():
                skopt_space.append(Categorical([hp.value], name=hp.name))
            case CS.OrdinalHyperparameter():
                skopt_space.append(Categorical(list(hp.sequence), name=hp.name))
            case _:
                raise ValueError(f"Unrecognized type of hyperparameter in ConfigSpace: {hp.__class__.__name__}!")

    return skopt_space
