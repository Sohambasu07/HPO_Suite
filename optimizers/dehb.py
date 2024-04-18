import os
from ConfigSpace import ConfigurationSpace
from pathlib import Path
from hpo_glue.glu import Optimizer, Query, Result, Config, Problem
from dehb import DEHB
import datetime


class DEHB_Optimizer(Optimizer):
    name = "DEHB"
    supports_multifidelity = True

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        seed: int | None = None,
        eta: int = 3
    ):
        """ Create a DEHB Optimizer instance for a given problem statement """

        
        self.problem = problem
        self.config_space: ConfigurationSpace = self.problem.problem_statement.benchmark.config_space
        self.fidelity_space: list[int] | list[float] | None = self.problem.problem_statement.benchmark.fidelity_space
        self.objectives: str | list[str] = self.problem.objectives
        self.minimize = self.problem.minimize
        self.seed = seed
        # TODO: Set seed if seed is None

        if os.path.exists(working_directory) is False:
            os.makedirs(working_directory)

        min_budget = None
        max_budget = None

        if self.fidelity_space is not None:
            min_budget = self.fidelity_space[0]
            max_budget = self.fidelity_space[-1]

        self.is_manyfidelity = self.problem.is_manyfidelity
        self.is_tabular = self.problem.is_tabular        
        self.is_multiobjective = self.problem.is_multiobjective

        self.dehb = DEHB(
            cs = self.config_space,
            min_fidelity = min_budget,
            max_fidelity = max_budget,
            seed = self.seed,
            eta = eta,
            n_workers = 1,
            output_path = working_directory
        )

        self.info = None


    def ask(self,
            config_id: str | None =  None) -> Query:
        """ Ask DEHB for a new config to evaluate """

        self.info = self.dehb.ask()
        config = Config(
            id = f"{self.seed}_{datetime.time()}",
            values = self.info["config"]
        )
        fidelity = self.info["fidelity"]
        
        if isinstance(fidelity, float):
            fidelity = round(fidelity)

        return Query(config=config, fidelity=fidelity)
    
    def tell(self,
             result: Result) -> None:
        """ Tell DEHB the result of the query """

        cost = result.result[self.objectives]   #Not considering Multiobjective for now
        if self.minimize is False:
            cost = -cost
        
        dehb_result = {
            "fitness": float(cost), # Actual objective value
            "cost": self.info['fidelity'] # TODO: Benchmark function (time or fidelity)
        }
        self.dehb.tell(self.info, dehb_result)