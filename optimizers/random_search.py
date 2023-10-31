from ConfigSpace import ConfigurationSpace, Configuration
from typing import Optional, Callable, ClassVar, Any, List, Dict
import numpy as np
from hpo_glue.glu import Optimizer, Query, Result, Config

class RandomSearch(Optimizer):
    name: ClassVar[str] = "RandomSearch"

    def __init__(self, config_space: ConfigurationSpace, seed: Optional[int] = None):
        self.config_space = config_space
        self.seed = None
        if seed is None:
            self._set_seed()
        else: self.seed = seed
        
    def get_config(self, num_configs: int) -> Configuration | list(Configuration):
        #Sample a random config or a list of configs from the configuration space
        config = self.config_space.sample_configuration(num_configs, seed=self.seed)
        return config
    
    def _set_seed(self):
        #Set the seed for random search
        self.seed = np.random.randint(0, 10**5)
    
    def ask(self) -> Query:
        #Ask the optimizer for a new config to evaluate
        config = self.get_config(1)
        return Query(Config("1", config))
    
    def tell(self, result: Result) -> None:
        #Tell the optimizer the result of the query
        pass