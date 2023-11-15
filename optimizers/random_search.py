from ConfigSpace import ConfigurationSpace, Configuration
from typing import ClassVar, Any
from pathlib import Path
from hpo_glue.glu import Optimizer, Query, Result, Config, ProblemStatement
import random

class RandomSearch(Optimizer):
    name: ClassVar[str] = "RandomSearch"

    def __init__(self, 
                 ProblemStatement: ProblemStatement,
                 working_directory: Path,
                 seed: int | None = None):
        """ Create a Random Search Optimizer instance for a given problem statement """

        self.problem = ProblemStatement
        self.config_space = ProblemStatement.config_space
        self.fidelity_space = ProblemStatement.fidelity_space
        self.objectives = ProblemStatement.result_keys
        self.seed = seed
        self.rng = random.Random(seed)
        
    def get_config(self, num_configs: int) -> Configuration | list[Configuration]:
        """ Sample a random config or a list of configs from the configuration space """

        sample_seed = self.rng.randint(0, 2**31-1)  # Big number so we can sample 2**31-1 possible configs
        self.config_space.seed(self.seed)
        config = self.config_space.sample_configuration(num_configs)
        return config
    
    def get_incumbent(self) -> Any:
        return None
    
    def ask(self,   
            config_id: str | None = None) -> Query:
        """ Ask the optimizer for a new config to evaluate """

        # We are dealing with a tabular benchmark
        if isinstance(self.config_space, list):
            config = random.choice(self.config_space)
            fidelity = random.choice(self.fidelity_space)
            return Query(config, fidelity)
        
        # We are dealing with a surrogate benchmark
        else:
            config = self.get_config(1)
            fidelity = random.choice(self.fidelity_space)
            # NOTE: I'm not sure how to deal with the `id` here...
            # There's no real order as the order in which you get configs every seed
            # will differ.
            # Perhaps we can also have `Configs` without an id? No idea...
            return Query(Config(config_id, config), fidelity)
    
    def tell(self, result: Result) -> None:
        """ Tell the optimizer the result of the query """
        
        pass