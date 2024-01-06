import os
from ConfigSpace import ConfigurationSpace, Configuration
from pathlib import Path
from hpo_glue.glu import Optimizer, Query, Result, Config, Problem
import random

class RandomSearch(Optimizer):
    name = "RandomSearch"
    supports_multifidelity = True
    supports_tabular = True

    def __init__(self, 
                 problem: Problem,
                 working_directory: Path,
                 seed: int | None = None):
        """ Create a Random Search Optimizer instance for a given problem statement """

        if isinstance(problem.objectives, list):
            raise NotImplementedError("# TODO: Implement multiobjective for RandomSearch")
        
        if isinstance(problem.fidelities, list):
            raise NotImplementedError("# TODO: Manyfidelity not yet implemented for RandomSearch!")

        self.problem = problem
        self.config_space: ConfigurationSpace = self.problem.problem_statement.benchmark.config_space
        self.fidelity_space: list[int] | list[float] = self.problem.problem_statement.benchmark.fidelity_space
        self.objectives = self.problem.objectives
        self.seed = seed
        self.rng = random.Random(seed)
        self.minimize = self.problem.minimize
        self.is_manyfidelity = self.problem.is_manyfidelity
        self.is_tabular = self.problem.is_tabular
        self.is_multiobjective = self.problem.is_multiobjective

        if os.path.exists(working_directory) is False:
            os.makedirs(working_directory)
        
    def get_config(self, num_configs: int) -> Configuration | list[Configuration]:
        """ Sample a random config or a list of configs from the configuration space """

        sample_seed = self.rng.randint(0, 2**31-1)  # Big number so we can sample 2**31-1 possible configs
        print(sample_seed)
        self.config_space.seed(sample_seed)
        config = self.config_space.sample_configuration(num_configs)
        return config
        
    def ask(self,   
            config_id: str | None = None) -> Query:
        """ Ask the optimizer for a new config to evaluate """

        fidelity = None
        # Randomly sampling from fidelity space for multifidelity
        fidelity = self.rng.choice(self.fidelity_space)

        # We are dealing with a tabular benchmark
        if self.is_tabular:
            config = self.rng.choice(self.config_space)
            return Query(config, fidelity)
        
        # We are dealing with a surrogate benchmark
        else:
            config = self.get_config(1)
            # NOTE: I'm not sure how to deal with the `id` here...
            # There's no real order as the order in which you get configs every seed
            # will differ.
            # Perhaps we can also have `Configs` without an id? No idea...
            return Query(Config(config_id, config), fidelity)
    
    def tell(self, result: Result) -> None:
        """ Tell the optimizer the result of the query """

        cost = result.result[self.problem.objectives]
        if self.minimize is False:
            cost = -cost