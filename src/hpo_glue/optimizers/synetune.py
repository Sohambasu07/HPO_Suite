from __future__ import annotations

# https://syne-tune.readthedocs.io/en/latest/examples.html#ask-tell-interface
# https://syne-tune.readthedocs.io/en/latest/examples.html#ask-tell-interface-for-hyperband
import datetime
from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import ConfigSpace as CS  # noqa: N817
from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import Domain, choice, lograndint, loguniform, ordinal, randint, uniform
from syne_tune.optimizer.baselines import BayesianOptimization

from hpo_glue.config import Config
from hpo_glue.optimizer import Optimizer
from hpo_glue.problem import Problem
from hpo_glue.query import Query

if TYPE_CHECKING:
    from syne_tune.optimizer.scheduler import TrialScheduler

    from hpo_glue.result import Result


class SyneTuneOptimizer(Optimizer):
    """Base class for SyneTune Optimizers."""

    name = "SyneTune-base"

    @abstractmethod
    def __init__(
        self,
        *,
        problem: Problem,
        working_directory: Path,
        optimizer: TrialScheduler,
    ):
        """Create a SyneTune Optimizer instance for a given problem statement."""
        working_directory.mkdir(parents=True, exist_ok=True)
        self.problem = problem
        self.optimizer = optimizer
        self._counter = 0

    def ask(self) -> Query:
        """Get a configuration from the optimizer."""
        self._counter += 1
        trial_suggestion = self.optimizer.suggest(self._counter)
        assert trial_suggestion is not None
        assert trial_suggestion.config is not None
        name = str(self._counter)
        trial = Trial(
            trial_id=self._counter,
            config=trial_suggestion.config,
            creation_time=datetime.datetime.now(),
        )

        # TODO: How to get the fidelity??
        return Query(
            config=Config(id=name, values=trial.config),
            fidelity=None,
            optimizer_info=trial,
        )

    def tell(self, result: Result) -> None:
        """Update the SyneTune Optimizer with the result of a query."""
        match self.problem.objective:
            case Mapping():
                results_obj_dict = {
                    key: result.full_results[key]
                    for key in result.full_results
                    if key in self.problem.objective
                }
            case (metric_name, _):
                results_obj_dict = {metric_name: result.full_results[metric_name]}
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        self.optimizer.on_trial_complete(
            trial=result.query.optimizer_info,  # type: ignore
            result=results_obj_dict,
        )


class SyneTuneBO(SyneTuneOptimizer):
    """SyneTune Bayesian Optimization."""

    name = "SyneTune-BO"
    support = Problem.Support(
        fidelities=False,
        objectives="single",
        cost_awareness=False,
        tabular=False,
    )

    def __init__(
        self,
        *,
        problem: Problem,
        working_directory: Path,
        config_space: CS.ConfigurationSpace | list[Config],
        **kwargs: Any,
    ):
        """Create a SyneTune Bayesian Optimization instance for a given problem statement.

        Args:
            problem: The problem statement.
            working_directory: The working directory to store the results.
            config_space: The configuration space.
            **kwargs: Additional arguments for the BayesianOptimization.
        """
        metric_name: str
        mode: Literal["min", "max"]
        match problem.objective:
            case Mapping():
                raise NotImplementedError(
                    "# TODO: Multiobjective not yet implemented for SyneTuneBO!"
                )
            case (name, metric):
                metric_name = name
                mode = "min" if metric.minimize else "max"
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        synetune_cs: dict[str, Domain]
        match config_space:
            case CS.ConfigurationSpace():
                synetune_cs = configspace_to_synetune_configspace(config_space)
            case list():
                raise ValueError("SyneTuneBO does not support tabular benchmarks")
            case _:
                raise TypeError("config_space must be of type ConfigSpace.ConfigurationSpace")

        match problem.fidelity:
            case None:
                pass
            case list() | dict():
                clsname = self.__class__.__name__
                raise ValueError(f"{clsname} does not support fidelity spaces")
            case _:
                raise TypeError("fidelity_space must be a list, dict or None")

        super().__init__(
            problem=problem,
            working_directory=working_directory,
            optimizer=BayesianOptimization(
                config_space=synetune_cs,
                metric=metric_name,
                mode=mode,
                random_seed=problem.seed,
                **kwargs,
            ),
        )


def configspace_to_synetune_configspace(
    config_space: CS.ConfigurationSpace,
) -> dict[str, Domain | Any]:
    """Convert ConfigSpace to SyneTune config_space."""
    if any(config_space.get_conditions()):
        raise NotImplementedError("ConfigSpace with conditions not supported")

    if any(config_space.get_forbiddens()):
        raise NotImplementedError("ConfigSpace with forbiddens not supported")

    synetune_cs: dict[str, Domain | Any] = {}
    for hp in config_space.get_hyperparameters():
        match hp:
            case CS.OrdinalHyperparameter():
                synetune_cs[hp.name] = ordinal(hp.sequence)
            case CS.CategoricalHyperparameter() if hp.weights is not None:
                raise NotImplementedError("CategoricalHyperparameter with weights not supported")
            case CS.CategoricalHyperparameter():
                synetune_cs[hp.name] = choice(hp.choices)
            case CS.UniformIntegerHyperparameter() if hp.log:
                synetune_cs[hp.name] = lograndint(hp.lower, hp.upper)
            case CS.UniformIntegerHyperparameter():
                synetune_cs[hp.name] = randint(hp.lower, hp.upper)
            case CS.UniformFloatHyperparameter() if hp.log:
                synetune_cs[hp.name] = loguniform(hp.lower, hp.upper)
            case CS.UniformFloatHyperparameter():
                synetune_cs[hp.name] = uniform(hp.lower, hp.upper)
            case CS.Constant():
                synetune_cs[hp.name] = hp.value
            case _:
                raise ValueError(f"Hyperparameter {hp.name} of type {type(hp)} is not supported")

    return synetune_cs
