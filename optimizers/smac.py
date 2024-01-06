import os
from ConfigSpace import ConfigurationSpace
from pathlib import Path
from hpo_glue.glu import Optimizer, Query, Result, Config, Problem
from smac import (
    HyperparameterOptimizationFacade as HPOFacade, 
    MultiFidelityFacade as MFFacade,
    HyperbandFacade as HBFacade,
    BlackBoxFacade as BOFacade,
    Scenario
)
from smac.runhistory.dataclasses import TrialInfo, TrialValue


class SMAC_Optimizer(Optimizer):
    name = "SMAC"

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        seed: int | None = None
    ):
        """ Create a SMAC Optimizer instance for a given problem statement """

        if isinstance(problem.objectives, list):
            raise NotImplementedError("# TODO: Implement multiobjective for SMAC")
        
        if isinstance(problem.fidelities, list):
            raise NotImplementedError("# TODO: Manyfidelity not yet implemented for SMAC!")
        
        self.problem = problem
        self.config_space: ConfigurationSpace = self.problem.problem_statement.benchmark.config_space
        self.fidelity_space: list[int] | list[float] | None = self.problem.problem_statement.benchmark.fidelity_space
        self.objectives: str | list[str] = self.problem.objectives
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
        # if self.problem.budget_type == "n_trials":
        #     n_trials = self.problem.budget

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

        self.is_manyfidelity = self.problem.is_manyfidelity
        
        self.is_tabular = self.problem.is_tabular
        if self.is_tabular:
            raise ValueError("SMAC does not support tabular benchmarks!")
        
        self.is_multiobjective = self.problem.is_multiobjective
        self.minimize = self.problem.minimize

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

        # if self.smac_info.budget is not None:
        #     fidelity = budget

        if self.__class__.supports_multifidelity is True:
            fidelity = budget

        _config_id = self.intensifier.runhistory.config_ids[config]  #For now using SMAC's own config_id

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

class SMAC_BO(SMAC_Optimizer):
    name = "SMAC_BO"

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        seed: int | None = None,
        xi: float = 0.0
    ):
        super().__init__(problem, working_directory, seed)
        self.facade = BOFacade
        self.intensifier = self.facade.get_intensifier(
            self.scenario,
        )
        self.acquisition_function = self.facade.get_acquisition_function(
            self.scenario,
            xi = xi
        )
        self.smac = self.facade(
            scenario = self.scenario,
            target_function = lambda seed, budget: None,
            intensifier = self.intensifier,
            acquisition_function = self.acquisition_function,
            overwrite = True)


class SMAC_Hyperband(SMAC_Optimizer):
    name = "SMAC_Hyperband"
    supports_multifidelity = True

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        eta: int = 3,
        seed: int | None = None
    ):
        super().__init__(problem, working_directory, seed)
        self.facade = HBFacade
        self.intensifier = self.facade.get_intensifier(
            self.scenario,
            eta = eta
        )
        self.smac = self.facade(
            scenario = self.scenario,
            target_function = lambda seed, budget: None,
            intensifier = self.intensifier,
            overwrite = True)