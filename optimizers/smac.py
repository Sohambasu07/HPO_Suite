import os
from ConfigSpace import ConfigurationSpace, Configuration
from typing import ClassVar
from pathlib import Path
import random
from hpo_glue.glu import Optimizer, Query, Result, Config, ProblemStatement
from smac import (HyperparameterOptimizationFacade as HPOFacade, 
                  MultiFidelityFacade as MFFacade,
                  Scenario)
from smac.runhistory.dataclasses import TrialInfo, TrialValue

class SMAC_Optimizer(Optimizer):
    name: ClassVar[str] = "SMAC"
    def __init__(self,
                 ProblemStatement: ProblemStatement,
                 working_directory: Path,
                 seed: int | None = None):
        """ Create a SMAC Optimizer instance for a given problem statement """
        
        self.problem = ProblemStatement
        self.config_space: ConfigurationSpace = ProblemStatement.config_space
        self.fidelity_space: list[int] | list[float] = ProblemStatement.fidelity_space
        self.objectives: str | list[str] = ProblemStatement.result_keys
        self.seed = seed
        self.rng = random.Random(seed)
        self.smac_info : TrialInfo = None #No parallel support
        self.smac_val : TrialValue = None #No parallel support


        if os.path.exists(working_directory) is False:
            os.makedirs(working_directory)

        self.scenario = Scenario(
            configspace=self.config_space,
            deterministic=False,
            objectives=self.objectives,
            n_trials=ProblemStatement.n_trials,
            output_directory=working_directory,
        )

        self.is_multifidelity = self.problem.is_multifidelity
        self.is_manyfidelity = self.problem.is_manyfidelity
        self.is_tabular = self.problem.is_tabular
        self.is_multiobjective = self.problem.is_multiobjective
        self.minimize = self.problem.minimize


        facade = self.get_facade()
        self.intensifier = facade.get_intensifier(
            self.scenario,
            max_config_calls=1,
        )
        self.smac = facade(scenario=self.scenario,
                         target_function=lambda seed: self.seed,
                         intensifier=self.intensifier,
                         overwrite=True)
        
        self.best_cost = 0.0
        
    def get_incumbent(self) -> tuple[Configuration | list[Configuration], 
                                     float | list[float]]:
        return self.intensifier.get_incumbent, self.best_cost


    def get_facade(self):
        if self.is_multifidelity:
            return MFFacade
        else:
            return HPOFacade
    
    def ask(self,
            config_id: str | None =  None) -> Query:
        """ Ask SMAC for a new config to evaluate """
        
        self.smac_info = self.smac.ask()
        config_id = self.intensifier.runhistory.get_config_id(self.smac_info.config) #For now using SMAC's own config_id
        config = self.smac_info.config.get_dictionary()
        fidelity = None
        if self.problem.fidelity_keys is None:
            if isinstance(self.fidelity_space, list):
                fidelity = self.fidelity_space[-1]
            elif isinstance(self.fidelity_space, ConfigurationSpace):
                #ideally get the highest fidelity; haven't figured out how to from ConfigSpace
                fidelity = self.fidelity_space.sample_configuration(1)
            else:
                raise ValueError("Fidelity space must be a list or ConfigSpace!")

        # else:
        #     fidelity = #Multifidelity setting for SMAC
        return Query(Config(config_id, config), fidelity=fidelity)
    
    def tell(self,
             result: Result) -> None:
        """ Tell SMAC the result of the query """
        cost = result.result[self.objectives]   #Not considering Multifidelity for now
        if self.minimize is False:
            cost = 1.0 - cost
        # if self.best_cost == [] or cost < self.best_cost[-1]:
        #     self.best_cost.append(cost)
        #     self.incumbents.append(self.smac_info.config)
        # else:
        #     self.best_cost.append(self.best_cost[-1])
        #     self.incumbents.append(self.incumbents[-1])
        if self.best_cost == 0.0 or cost < self.best_cost:
            self.best_cost = cost
        self.smac_val = TrialValue(cost = cost, time = 0.0)
        self.smac.tell(self.smac_info, self.smac_val)