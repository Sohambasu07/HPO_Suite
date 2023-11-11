from ConfigSpace import ConfigurationSpace, Configuration
from typing import ClassVar
from hpo_glue.glu import Optimizer, Query, Result, Config
import random

class RandomSearch(Optimizer):
    name: ClassVar[str] = "RandomSearch"

    def __init__(self, 
                 config_space: ConfigurationSpace | list[Config], 
                 fidelity_space = ConfigurationSpace | list[int] | list[float],
                   seed: int | None = None):
        
        self.config_space = config_space
        self.fidelity_space = fidelity_space
        self.seed = None
        self.rng = random.Random(seed)
        # if seed is None:
        #     self._set_seed()
        # else: 
        #     self.seed = seed
        
    def get_config(self, num_configs: int) -> Configuration | list[Configuration]:
        """ Sample a random config or a list of configs from the configuration space """

        sample_seed = self.rng.randint(0, 2**31-1)  # Big number so we can sample 2**31-1 possible configs
        config = self.config_space.sample_configuration(num_configs, seed=sample_seed)
        return config
    
    # def _set_seed(self):
    #     #Set the seed for random search
    #     self.seed = random.randint(0, 10**5)
    #     random.seed(self.seed)
    
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
            config = self.config_space.sample_configuration(1, seed=self.seed)
            
            # NOTE: I'm not sure how to deal with the `id` here...
            # There's no real order as the order in which you get configs every seed
            # will differ.
            # Perhaps we can also have `Configs` without an id? No idea...
            return Query(Config(config_id, config))
    
    def tell(self, result: Result) -> None:
        """ Tell the optimizer the result of the query """
        
        pass