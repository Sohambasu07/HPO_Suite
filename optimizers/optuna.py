from __future__ import annotations

import ConfigSpace as CS
import optuna
from optuna.samplers import *
from optuna.distributions import (
    CategoricalDistribution as Cat,
    FloatDistribution as Float,
    IntDistribution as Int
)
from pathlib import Path
import logging

from hpo_glue.glu import Optimizer, Query, Result, Config, Problem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OptunaOptimizer(Optimizer):
    name = "Optuna"
    # supports_multiobjective = True

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        sampler: str | None = None,
        seed: int | None = None
    ):
        if isinstance(problem.objectives, list):
            raise NotImplementedError("# TODO: Implement multiobjective for Optuna")
        
        if isinstance(problem.fidelities, list):
            raise NotImplementedError("# TODO: Manyfidelity not yet implemented for Optuna!")
        
        self.problem = problem
        self.working_directory = working_directory
        self.config_space = self.problem.problem_statement.benchmark.config_space
        self.fidelity_space = self.problem.problem_statement.benchmark.fidelity_space
        # if seed is None:
        #     seed = -1
        self.seed = seed

        self.study = optuna.create_study(
            direction = "minimize" if self.problem.minimize else "maximize"
        )

        if sampler is not None:
            self.study.sampler = eval(sampler)()
            __class__.name += f"_{sampler}"

        self.distributions = configspace_to_optuna_distributions( 
            self.problem.problem_statement.benchmark.config_space
        )
        self.counter = 0

    def ask(self, config_id: str | None =  None) -> Query:
        self.trial = self.study.ask(self.distributions)
        config = Config(
            id=f"{self.seed}_{self.counter}",
            values=self.trial.params,
        )
        self.counter += 1
        return Query(config=config, fidelity = None)

    def tell(self, result: Result) -> None:
        self.study.tell(self.trial, result.result[self.problem.objectives])


def configspace_to_optuna_distributions(config_space: CS.ConfigurationSpace) -> dict:
    if not isinstance(config_space, CS.ConfigurationSpace):
        raise ValueError("Need search space of type ConfigSpace.ConfigurationSpace}")
    optuna_space = dict()
    for hp in config_space.get_hyperparameters():
        if isinstance(hp, CS.UniformIntegerHyperparameter):
            optuna_space[hp.name] = Int(hp.lower, hp.upper, log=hp.log)
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            optuna_space[hp.name] = Float(hp.lower, hp.upper, log=hp.log)
        elif isinstance(hp, CS.CategoricalHyperparameter):
            optuna_space[hp.name] = Cat(hp.choices)
        elif isinstance(hp, CS.Constant):
            if isinstance(hp.value, (int, float)):
                optuna_space[hp.name] = Float(hp.value, hp.value)
            else:
                optuna_space[hp.name] = Cat((hp.value))
        elif isinstance(hp, CS.OrdinalHyperparameter):
            # TODO: handle categoricals better
            optuna_space[hp.name] = Cat(hp.sequence)
        else:
            raise ValueError("Unrecognized type of hyperparameter in ConfigSpace!")
    return optuna_space

