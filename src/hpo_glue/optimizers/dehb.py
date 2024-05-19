from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ConfigSpace import ConfigurationSpace
from dehb import DEHB

from hpo_glue.optimizer import Optimizer
from hpo_glue.problem import Problem
from hpo_glue.query import Config, Query

if TYPE_CHECKING:
    from hpo_glue.problem import Fidelity
    from hpo_glue.result import Result


class DEHB_Optimizer(Optimizer):
    """The DEHB Optimizer.

    TODO: Document me
    """

    name = "DEHB"

    support = Problem.Support(
        fidelities="single",
        objectives="single",
        cost_awareness="single",
        tabular=False,
    )

    def __init__(
        self,
        *,
        problem: Problem,
        working_directory: Path,
        config_space: list[Config] | ConfigurationSpace,
        eta: int = 3,
        # TODO(eddiebergman): Add more DEHB parameters
    ):
        """Create a DEHB Optimizer instance for a given problem statement."""
        match config_space:
            case ConfigurationSpace():
                pass
            case list():
                raise NotImplementedError("# TODO: Tabular not yet implemented for DEHB!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        self._fidelity: Fidelity | None
        match problem.fidelity:
            case None:
                self._fidelity = None
                min_fidelity = None
                max_fidelity = None
            case (_, fidelity):
                self._fidelity = fidelity
                min_fidelity = fidelity.min
                max_fidelity = fidelity.max
            case Mapping():
                raise NotImplementedError("# TODO: Manyfidelity not yet implemented for DEHB!")
            case _:
                raise TypeError("Fidelity must be a string or a list of strings!")

        working_directory.mkdir(parents=True, exist_ok=True)

        self.problem = problem
        self.dehb = DEHB(
            cs=config_space,
            min_fidelity=min_fidelity,
            max_fidelity=max_fidelity,
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
        match fidelity:
            case None:
                fidelity = None
            case float() | int():
                assert self._fidelity is not None
                fidelity = int(fidelity) if self._fidelity.kind is int else fidelity
            case _:
                raise NotImplementedError("Unexpected return type for SMAC budget!")

        config_id = info["config_id"]
        raw_config = info["config"]
        name = f"trial_{config_id}"

        return Query(
            config=Config(id=name, values=raw_config),
            fidelity=fidelity,
            optimizer_info=info,
        )

    def tell(self, result: Result) -> None:
        """Tell DEHB the result of the query."""
        fitness: float
        match self.problem.objective:
            case (name, metric):
                fitness = metric.as_minimize(result.full_results[name])
            case Mapping():
                raise NotImplementedError("# TODO: Multiobjective not yet implemented for DEHB!")
            case _:
                raise TypeError("Objective must be a string or a list of strings!")

        cost = self.problem.budget.calculate_used_budget(result=result, problem=self.problem)

        self.dehb.tell(
            result.query.optimizer_info,
            {"fitness": fitness, "cost": cost},
        )
