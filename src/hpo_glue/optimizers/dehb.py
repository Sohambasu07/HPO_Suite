from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from dehb import DEHB

from hpo_glue.optimizer import Optimizer
from hpo_glue.query import Config, Query

if TYPE_CHECKING:
    from hpo_glue.problem import Problem
    from hpo_glue.result import Result


class DEHB_Optimizer(Optimizer):
    """The DEHB Optimizer.

    TODO: Document me
    """

    name = "DEHB"
    supports_multifidelity = True
    supports_manyfidelity = False
    supports_tabular = False
    supports_multiobjective = False

    def __init__(
        self,
        *,
        problem: Problem,
        working_directory: Path,
        eta: int = 3,
        # TODO(eddiebergman): Add more DEHB parameters
    ):
        """Create a DEHB Optimizer instance for a given problem statement."""
        if isinstance(problem.fidelity_space, dict):
            raise NotImplementedError("# TODO: Manyfidelity not yet implemented for DEHB!")

        if isinstance(problem.config_space, list):
            raise NotImplementedError("# TODO: Tabular not yet implemented for DEHB!")

        if isinstance(problem.objective, list):
            raise NotImplementedError("# TODO: Multiobjective not yet implemented for DEHB!")

        working_directory.mkdir(parents=True, exist_ok=True)

        if problem.fidelity_space is not None:
            min_budget = problem.fidelity_space[0]
            max_budget = problem.fidelity_space[-1]
        else:
            min_budget = None
            max_budget = None

        self.problem = problem
        self.dehb = DEHB(
            cs=problem.config_space,
            min_fidelity=min_budget,
            max_fidelity=max_budget,
            seed=problem.seed,
            eta=eta,
            n_workers=1,
            output_path=working_directory,
        )
        self._info_lookup: dict[str, dict[str, Any]] = {}

    def ask(self) -> Query:
        """Ask DEHB for a new config to evaluate."""
        info = self.dehb.ask()
        fidelity = info.get("fidelity")
        if isinstance(fidelity, float):
            fidelity = round(fidelity)

        config_id = info.get("config_id")
        name = f"trial_{config_id}_{fidelity=}"
        self._info_lookup[name] = info

        raw_config = info["config"]
        config = Config(id=name, values=raw_config)
        return Query(config=config, fidelity=fidelity)

    def tell(self, result: Result) -> None:
        """Tell DEHB the result of the query."""
        match self.problem.objective:
            case str():
                assert isinstance(self.problem.minimize, bool)
                fitness = float(result.result[self.problem.objective])
                fitness = -fitness if not self.problem.minimize else fitness
            case list():
                raise NotImplementedError("# TODO: Multiobjective not yet implemented for DEHB!")
            case _:
                raise TypeError("Objective must be a string or a list of strings!")

        match self.problem.budget_type:
            case "n_trials":
                cost = 1
            case "time_budget":
                assert self.problem.benchmark.time_budget is not None
                cost = result.result[self.problem.benchmark.time_budget]
            case "fidelity_budget":
                assert result.fidelity is not None
                assert not isinstance(result.fidelity, dict)
                cost = result.fidelity
            case _:
                raise ValueError(f"Invalid budget type {self.problem.budget_type}!")

        dehb_result = {"fitness": fitness, "cost": cost}
        original_info = self._info_lookup.pop(result.config.id)
        self.dehb.tell(original_info, dehb_result)
