from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from ConfigSpace import ConfigurationSpace

from hpo_glue.budget import CostBudget, TrialBudget
from hpo_glue.config import Config
from hpo_glue.env import Env
from hpo_glue.optimizer import Optimizer
from hpo_glue.problem import Problem
from hpo_glue.query import Query

if TYPE_CHECKING:
    from smac.facade import AbstractFacade
    from smac.runhistory import TrialInfo

    from hpo_glue.problem import Fidelity
    from hpo_glue.result import Result


def _dummy_target_function(*args: Any, budget: int | float, seed: int) -> NoReturn:
    raise RuntimeError("This should never be called!")


class SMAC_Optimizer(Optimizer):
    """Default SMAC Optimizer.

    # TODO: Document me
    """

    env = Env(name="SMAC-2.1", python_version="3.10", requirements=("smac==2.1",))

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        config_space: list[Config] | ConfigurationSpace,
        optimizer: AbstractFacade,
        fidelity: Fidelity | None,
    ):
        """Create a SMAC Optimizer instance for a given problem statement.

        Args:
            problem: Problem statement.
            seed: Random seed for the optimizer.
            working_directory: Working directory to store SMAC run.
            config_space: Configuration space to optimize over.
            optimizer: SMAC optimizer instance.
            fidelity: Fidelity space to optimize over, if any.
        """
        self.problem = problem
        self.working_directory = working_directory
        self.optimizer = optimizer
        self.config_space = config_space
        self._trial_lookup: dict[Hashable, TrialInfo] = {}
        self._fidelity = fidelity
        self._seed = seed

    def ask(self) -> Query:
        """Ask SMAC for a new config to evaluate."""
        smac_info = self.optimizer.ask()
        assert smac_info.instance is None, "We don't do instance benchmarks!"

        config = smac_info.config
        raw_config = config.get_dictionary()
        config_id = str(self.optimizer.intensifier.runhistory.config_ids[config])

        fidelity = smac_info.budget

        match fidelity:
            case None:
                fidelity = None
            case float() | int():
                assert self._fidelity is not None
                assert isinstance(self.problem.fidelity, tuple)
                fidelity_value = int(fidelity) if self._fidelity.kind is int else fidelity
                fidelity_name = self.problem.fidelity[0]
                fidelity = (fidelity_name, fidelity_value)
            case _:
                raise NotImplementedError("Unexpected return type for SMAC budget!")

        return Query(
            config=Config(config_id=config_id, values=raw_config),
            fidelity=fidelity,
            optimizer_info=smac_info,
        )

    def tell(self, result: Result) -> None:
        """Tell SMAC the result of the query."""
        from smac.runhistory import StatusType, TrialValue

        match self.problem.objective:
            case Mapping():
                cost = [
                    obj.as_minimize(result.values[key])
                    for key, obj in self.problem.objective.items()
                ]
            case (key, obj):
                cost = obj.as_minimize(result.values[key])
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        self.optimizer.tell(
            result.query.optimizer_info,  # type: ignore
            TrialValue(
                cost=cost,
                time=0.0,
                starttime=0.0,
                endtime=0.0,
                status=StatusType.SUCCESS,
                additional_info={},
            ),
            save=True,
        )


class SMAC_BO(SMAC_Optimizer):
    """SMAC Bayesian Optimization."""

    name = "SMAC_BO"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("single", "many"),
        cost_awareness=(None,),
        tabular=False,
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        config_space: ConfigurationSpace | list[Config],
        xi: float = 0.0,
    ):
        """Create a SMAC BO Optimizer instance for a given problem statement.

        Args:
            problem: Problem statement.
            seed: Random seed for the optimizer.
            working_directory: Working directory to store SMAC run.
            config_space: Configuration space to optimize over.
            xi: Exploration-exploitation trade-off parameter. Defaults to 0.0.
        """
        match config_space:
            case ConfigurationSpace():
                pass
            case list():
                raise ValueError("SMAC does not support tabular benchmarks!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        match problem.fidelity:
            case None:
                pass
            case tuple() | Mapping():
                raise ValueError("SMAC BO does not support multi-fidelity benchmarks!")
            case _:
                raise TypeError("Fidelity must be a string or a list of strings!")

        match problem.objective:
            case Mapping():
                metric_names = list(problem.objective.keys())
            case (metric_name, _):
                metric_names = metric_name
            case _:
                raise TypeError("Objective must be a tuple of (name, metric) or a mapping")

        working_directory.mkdir(parents=True, exist_ok=True)

        match problem.budget:
            case TrialBudget():
                budget = problem.budget.total
            case CostBudget():
                raise ValueError("SMAC BO does not support cost-aware benchmarks!")
            case _:
                raise TypeError("Budget must be a TrialBudget or a CostBudget!")

        from smac import BlackBoxFacade, Scenario

        scenario = Scenario(
            configspace=config_space,
            deterministic=True,
            objectives=metric_names,
            n_trials=budget,
            seed=seed,
            output_directory=working_directory / "smac-output",
            min_budget=None,
            max_budget=None,
        )
        super().__init__(
            problem=problem,
            seed=seed,
            config_space=config_space,
            working_directory=working_directory,
            fidelity=None,
            optimizer=BlackBoxFacade(
                scenario=scenario,
                logging_level=False,
                target_function=_dummy_target_function,
                intensifier=BlackBoxFacade.get_intensifier(scenario),
                acquisition_function=BlackBoxFacade.get_acquisition_function(scenario, xi=xi),
                overwrite=True,
            ),
        )


class SMAC_Hyperband(SMAC_Optimizer):
    """SMAC Hyperband."""

    name = "SMAC_Hyperband"
    support = Problem.Support(
        fidelities=("single",),
        objectives=("single", "many"),
        cost_awareness=(None,),
        tabular=False,
    )
    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        config_space: ConfigurationSpace | list[Config],
        eta: int = 3,
    ):
        """Create a SMAC Hyperband Optimizer instance for a given problem statement.

        Args:
            problem: Problem statement.
            seed: Random seed for the optimizer.
            working_directory: Working directory to store SMAC run.
            config_space: Configuration space to optimize over.
            eta: Hyperband eta parameter. Defaults to 3.
        """
        match config_space:
            case ConfigurationSpace():
                pass
            case list():
                raise ValueError("SMAC does not support tabular benchmarks!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        min_fidelity: float | int
        max_fidelity: float | int
        match problem.fidelity:
            case None:
                raise ValueError("SMAC Hyperband requires a fidelity space!")
            case Mapping():
                raise ValueError("SMAC Hyperband does not support many-fidelity!")
            case (_, fidelity):
                _fid = fidelity
                min_fidelity = fidelity.min
                max_fidelity = fidelity.max
            case _:
                raise TypeError("Fidelity must be a string or a list of strings!")

        match problem.objective:
            case Mapping():
                metric_names = list(problem.objective.keys())
            case (metric_name, _):
                metric_names = metric_name
            case _:
                raise TypeError("Objective must be a tuple of (name, metric) or a mapping")

        working_directory.mkdir(parents=True, exist_ok=True)
        match problem.budget:
            case TrialBudget():
                budget = problem.budget.total
            case CostBudget():
                raise ValueError("SMAC BO does not support cost-aware benchmarks!")
            case _:
                raise TypeError("Budget must be a TrialBudget or a CostBudget!")

        from smac import HyperbandFacade, Scenario

        scenario = Scenario(
            configspace=config_space,
            deterministic=True,
            objectives=metric_names,
            n_trials=budget,
            seed=seed,
            output_directory=working_directory / "smac-output",
            min_budget=min_fidelity,
            max_budget=max_fidelity,
        )
        super().__init__(
            problem=problem,
            seed=seed,
            config_space=config_space,
            working_directory=working_directory,
            fidelity=_fid,
            optimizer=HyperbandFacade(
                scenario=scenario,
                logging_level=False,
                target_function=_dummy_target_function,
                intensifier=HyperbandFacade.get_intensifier(scenario, eta=eta),
                overwrite=True,
            ),
        )
