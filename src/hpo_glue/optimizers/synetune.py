from __future__ import annotations

# https://syne-tune.readthedocs.io/en/latest/examples.html#ask-tell-interface
# https://syne-tune.readthedocs.io/en/latest/examples.html#ask-tell-interface-for-hyperband
import datetime
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ConfigSpace as CS  # noqa: N817
from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import choice, loguniform, ordinal, randint, uniform
from syne_tune.optimizer.baselines import BayesianOptimization
from syne_tune.optimizer.schedulers.hyperband import MultiFidelitySchedulerMixin

from hpo_glue.config import Config
from hpo_glue.optimizer import Optimizer
from hpo_glue.query import Query

if TYPE_CHECKING:
    from syne_tune.optimizer.scheduler import TrialScheduler

    from hpo_glue.problem import Problem
    from hpo_glue.result import Result


class SyneTuneOptimizer(Optimizer):
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
        if self.supports_multifidelity:
            if not isinstance(optimizer, MultiFidelitySchedulerMixin):
                raise ValueError("supports_multi_fidelity=True but optimizer does not support it")
            raise NotImplementedError("Multifidelity for synetune not supported yet!")

        working_directory.mkdir(parents=True, exist_ok=True)
        self.problem = problem
        self.optimizer = optimizer
        self._counter = 0
        self._trial_lookup: dict[str, Trial] = {}

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
        self._trial_lookup[name] = trial
        return Query(config=Config(id=name, values=trial.config), fidelity=None)

    def tell(self, result: Result) -> None:
        """Update the SyneTune Optimizer with the result of a query."""
        original_trial = self._trial_lookup.pop(result.config_id)
        match self.problem.objective:
            case str():
                results_obj_dict = {self.problem.objective: result.result[self.problem.objective]}
            case list():
                results_obj_dict = {
                    key: result.result[key]
                    for key in result.result
                    if key in self.problem.objective
                }
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        self.optimizer.on_trial_complete(trial=original_trial, result=results_obj_dict)


class SyneTuneBO(SyneTuneOptimizer):
    name = "SyneTune-BO"
    supports_tabular = False
    supports_manyfidelity = False
    supports_multifidelity = False
    supports_multiobjective = False

    def __init__(
        self,
        *,
        problem: Problem,
        working_directory: Path,
        **kwargs: Any,
    ):
        match problem.objective:
            case str():
                metric = problem.objective
                assert isinstance(problem.minimize, bool)
                mode = "min" if problem.minimize else "max"
            case list():
                clsname = self.__class__.__name__
                raise ValueError(f"{clsname} does not support multiple objectives")
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        match problem.config_space:
            case CS.ConfigurationSpace():
                synetune_cs = configspace_to_synetune_configspace(problem.config_space)
            case list():
                raise ValueError("SyneTune does not support tabular benchmarks")
            case _:
                raise TypeError("config_space must be of type ConfigSpace.ConfigurationSpace")

        match problem.fidelity_space:
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
                metric=metric,
                mode=mode,
                random_seed=problem.seed,
                **kwargs,
            ),
        )


def configspace_to_synetune_configspace(config_space: CS.ConfigurationSpace) -> dict:
    """Convert ConfigSpace to SyneTune config_space."""
    if not isinstance(config_space, CS.ConfigurationSpace):
        raise ValueError("config_space must be of type ConfigSpace.ConfigurationSpace")
    synetune_cs = {}
    for hp in config_space.get_hyperparameters():
        if isinstance(hp, CS.OrdinalHyperparameter):
            synetune_cs[hp.name] = ordinal(hp.sequence)
        elif isinstance(hp, CS.CategoricalHyperparameter):
            synetune_cs[hp.name] = choice(hp.choices)  # choice.weights in  CS -> check SyneTune
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            synetune_cs[hp.name] = randint(hp.lower, hp.upper)  # check for logscale (hp.log)
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            if hp.log:
                synetune_cs[hp.name] = loguniform(hp.lower, hp.upper)
            else:
                synetune_cs[hp.name] = uniform(hp.lower, hp.upper)
        elif isinstance(hp, CS.Constant):
            synetune_cs[hp.name] = hp.value
        else:
            raise ValueError(f"Hyperparameter {hp.name} of type {type(hp)} is not supported")

    return synetune_cs
