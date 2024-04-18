import os
import ConfigSpace as CS
from pathlib import Path
from hpo_glue.glu import Optimizer, Query, Result, Config, Problem
from syne_tune.optimizer.baselines import (BayesianOptimization, 
                                           HyperbandScheduler, 
                                           ASHA, 
                                           BOHB)
from syne_tune.backend.trial_status import Trial, Status, TrialResult
from syne_tune.config_space import uniform, loguniform, ordinal, choice, randint
import datetime

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from syne_tune.optimizer.scheduler import TrialScheduler


class SyneTuneOptimizer(Optimizer):
    name = "SyneTune"

    def __init__(
            self,
            problem: Problem,
            working_directory: Path,
            seed: int | None = None,
    ):
        """ Create a SyneTune Optimizer instance for a given problem statement 

        (Curretly running Bayesian Optimization as the base/default optimizer)
        """

        self.problem = problem
        self.config_space: CS.ConfigurationSpace = self.problem.problem_statement.benchmark.config_space
        self.synetune_cs = configspace_to_synetune_configspace(self.config_space)
        self.fidelity_space: list[int] | list[float] | None = self.problem.problem_statement.benchmark.fidelity_space
        self.objectives: str | list[str] = self.problem.objectives
        self.seed = seed 

        if os.path.exists(working_directory) is False:
            os.makedirs(working_directory)

        self.bscheduler: TrialScheduler | None = BayesianOptimization(
            config_space=self.synetune_cs,
            metric=self.objectives,
            mode = "min" if self.problem.minimize else "max",
            random_seed = self.seed,
        )
        
        self.trial_counter = 0
        self.trial = None

    def ask(
            self,
            config_id: str | None = None
    ) -> Query:
        
        """Get a configuration from the optimizer"""
        
        trial_suggestion = self.bscheduler.suggest(self.trial_counter)
        self.trial = Trial(
            trial_id=str(self.trial_counter),
            config = trial_suggestion.config,
            creation_time = datetime.datetime.now()
        )
        config = Config(
            id = str(self.trial_counter),
            values = self.trial.config
        )
        self.trial_counter += 1
        fidelity = None
        # if self.__class__.supports_multifidelity:
        #     fidelity = None

        return Query(config = config, fidelity = fidelity)
    
    def tell(
            self,
            result: Result
    ) -> None:
        """Update the SyneTune Optimizer with the result of a query"""

        results_obj_dict = {key: result.result[key] for key in result.result.keys() if key == self.objectives}

        cost = result.result[self.objectives]
        if self.problem.minimize:
            cost = -cost

        self.bscheduler.on_trial_complete(
            trial = self.trial, 
            result=results_obj_dict
        )

def configspace_to_synetune_configspace(config_space: CS.ConfigurationSpace) -> dict:
    """Convert ConfigSpace to SyneTune config_space"""
    if not isinstance(config_space, CS.ConfigurationSpace):
        raise ValueError("config_space must be of type ConfigSpace.ConfigurationSpace")
    synetune_cs = {}
    for hp in config_space.get_hyperparameters():
        if isinstance(hp, CS.OrdinalHyperparameter):
            synetune_cs[hp.name] = ordinal(hp.sequence)
        elif isinstance(hp, CS.CategoricalHyperparameter):
            synetune_cs[hp.name] = choice(hp.choices) # choice.weights in  CS -> check SyneTune
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            synetune_cs[hp.name] = randint(hp.lower, hp.upper) # check for logscale (hp.log)
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


# class SyneTuneHyperband(SyneTuneOptimizer):
#     name = "SyneTune_Hyperband"
#     supports_multifidelity = True

#     def __init__(
#             self,
#             problem: Problem,
#             working_directory: Path,
#             seed: int | None = None,
#             eta: int = 3,
#     ):
#         """ Create a SyneTune Hyperband instance for a given problem statement """

#         super().__init__(problem, working_directory, seed)


#         searcher = 'bayesopt'
#         self.bscheduler = HyperbandScheduler(
#         self.synetune_cs,
#         searcher=searcher,
#         type="stopping",
#         max_t = self.fidelity_space[-1],
#         resource_attr = problem.problem_statement.benchmark.fidelity_keys,
#         mode = "min" if self.problem.minimize else "max",
#         metric = self.objectives,
#         grace_period=1,
#         reduction_factor = eta,
#         random_seed = seed
#     )