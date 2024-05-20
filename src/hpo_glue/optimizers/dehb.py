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
    from hpo_glue.result import Result


class DEHB_Optimizer(Optimizer):
    """The DEHB Optimizer.

    TODO: Document me
    """

    name = "DEHB"

    support = Problem.Support(
        fidelities=(None, "single"),
        objectives=("single",),
        cost_awareness=(None, "single"),
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

        match problem.fidelity:
            case None:
                min_fidelity = None
                max_fidelity = None
            case (_, fidelity):
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

        match self.problem.fidelity:
            case None:
                fidelity = None
            case (key, fidelity_def):
                _val = info["fidelity"]
                value = int(_val) if fidelity_def.kind is int else _val
                fidelity = (key, value)
            case Mapping():
                raise NotImplementedError("# TODO: many-fidleity not yet implemented for DEHB!")
            case _:
                raise TypeError("Fidelity must be None, a tuple, or a mapping!")

        config_id = info["config_id"]
        raw_config = info["config"]
        name = f"trial_{config_id}"

        return Query(
            config=Config(config_id=name, values=raw_config),
            fidelity=fidelity,
            optimizer_info=info,
        )

    def tell(self, result: Result) -> None:
        """Tell DEHB the result of the query."""
        fitness: float
        match self.problem.objective:
            case (name, metric):
                fitness = metric.as_minimize(result.values[name])
            case Mapping():
                raise NotImplementedError("# TODO: Multiobjective not yet implemented for DEHB!")
            case _:
                raise TypeError("Objective must be a string or a list of strings!")

        match self.problem.cost:
            case None:
                cost = None
            case tuple():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for DEHB!")
            case Mapping():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for DEHB!")
            case _:
                raise TypeError("Cost must be None or a mapping!")

        if cost is None:
            self.dehb.tell(
                result.query.optimizer_info,
                {"fitness": fitness},
            )
        else:
            raise NotImplementedError("# TODO: Cost-aware not yet implemented for DEHB!")
