import os
from ConfigSpace import ConfigurationSpace
from pathlib import Path
from hpo_glue.glu import Optimizer, Query, Result, Config, ProblemStatement
from smac import (HyperparameterOptimizationFacade as HPOFacade, 
                  MultiFidelityFacade as MFFacade,
                  Scenario)
from smac.runhistory.dataclasses import TrialInfo, TrialValue

class SMAC_Optimizer(Optimizer):
    name = "SMAC"
    supports_multifidelity = True


    def __init__(self,
                 problem_statement: ProblemStatement,
                 working_directory: Path,
                 seed: int | None = None):
        """ Create a SMAC Optimizer instance for a given problem statement """

        if isinstance(problem_statement.result_keys, list):
            raise NotImplementedError("# TODO: Implement multiobjective for SMAC")
        
        if isinstance(problem_statement.fidelity_keys, list):
            raise NotImplementedError("# TODO: Manyfidelity not yet implemented for SMAC!")
        
        self.problem_statement = problem_statement
        self.config_space: ConfigurationSpace = self.problem_statement.config_space
        self.fidelity_space: list[int] | list[float] | None = self.problem_statement.fidelity_space
        self.objectives: str | list[str] = self.problem_statement.result_keys
        if seed is None:
            seed = -1
        self.seed = seed
        self.smac_info : TrialInfo | None = None #No parallel support
        self.smac_val : TrialValue | None = None #No parallel support


        if os.path.exists(working_directory) is False:
            os.makedirs(working_directory)

        min_budget = None
        max_budget = None

        if self.fidelity_space is not None:
            min_budget = self.fidelity_space[0]
            max_budget = self.fidelity_space[-1]

        n_trials = 1
        if self.problem_statement.budget_type == "n_trials":
            n_trials = self.problem_statement.budget

        self.scenario = Scenario(
            configspace=self.config_space,
            deterministic=False,
            objectives=self.objectives,
            n_trials=n_trials,
            output_directory=working_directory,
            min_budget=min_budget,
            max_budget=max_budget,
            seed=self.seed
        )

        self.is_multifidelity = self.problem_statement.is_multifidelity
        self.is_manyfidelity = self.problem_statement.is_manyfidelity
        
        self.is_tabular = self.problem_statement.is_tabular
        if self.is_tabular:
            raise ValueError("SMAC does not support tabular benchmarks!")
        
        self.is_multiobjective = self.problem_statement.is_multiobjective
        self.minimize = self.problem_statement.minimize

        if self.is_multifidelity:
            self.facade = MFFacade
        else:
            self.facade = HPOFacade

        self.intensifier = self.facade.get_intensifier(
            self.scenario,
        )

        self.smac = self.facade(scenario=self.scenario,
                         target_function=lambda seed, budget: None,
                         intensifier=self.intensifier,
                         overwrite=True)
    
    def ask(self,
            config_id: str | None =  None) -> Query:
        """ Ask SMAC for a new config to evaluate """
        
        self.smac_info = self.smac.ask()
        config = self.smac_info.config
        budget = self.smac_info.budget
        instance = self.smac_info.instance
        seed = self.smac_info.seed
        fidelity = None

        if self.smac_info.budget is not None:
            fidelity = budget

        _config_id = self.intensifier.runhistory.config_ids[config] #For now using SMAC's own config_id

        config = Config(
            id=f"{_config_id=}_{seed=}_{instance=}",
            values=config.get_dictionary(),
            )
        
        if isinstance(fidelity, float):
            fidelity = round(fidelity)

        return Query(config=config, fidelity=fidelity)
    
    def tell(self,
             result: Result) -> None:
        """ Tell SMAC the result of the query """
        cost = result.result[self.objectives]   #Not considering Multiobjective for now
        if self.minimize is False:
            cost = -cost
        self.smac_val = TrialValue(cost = cost, time = 0.0)
        self.smac.tell(self.smac_info, self.smac_val)