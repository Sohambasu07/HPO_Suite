from typing import List, Dict, Any, Optional
from ConfigSpace import ConfigurationSpace, Configuration
from tt import Result
import json
import os
from glu import Hist_Storage


class Storage(Hist_Storage):
    def __init__(self, config_space: ConfigurationSpace, curr_seed: int, curr_config: Configuration,
                 run_id: str, run_dir: str, save: bool = True):
        self.config_space = config_space
        self.seed = curr_seed
        self.config = curr_config
        self.run_id = run_id
        self.run_dir = run_dir
        self.save = save

    def create_run_dir(self):
        #Create a directory for the current run
        if self.save:
            os.makedirs(self.run_dir, exist_ok=True)

    def save_config(self):
        #Save the current config
        if self.save:
            with open(f"{self.run_dir}/{self.run_id}_config.json", "w") as f:
                json.dump(self.config.get_dictionary(), f, indent=4)
                json.dump({"seed": self.seed}, f, indent=4)

    def save_results(self, result: Result):
        #Save the results of the current run
        q = result.query
        r = result.result
        if self.save:
            with open(f"{self.run_dir}/{self.run_id}_results.json", "w") as f:
                json.dump({"query": q, "result": r}, f, indent=4)
