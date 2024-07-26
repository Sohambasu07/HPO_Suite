from __future__ import annotations

import numpy as np
import math
from dataclasses import dataclass, field
import logging
import traceback
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Literal
from pathlib import Path

from tqdm import TqdmWarning, tqdm

from hpo_glue.budget import CostBudget, TrialBudget
from hpo_glue.fidelity import Fidelity
from hpo_glue.run import Run
from hpo_glue.utils import rescale

if TYPE_CHECKING:
    from hpo_glue.benchmark import Benchmark
    from hpo_glue.optimizer import Optimizer
    from hpo_glue.problem import Problem
    from hpo_glue.result import Result

logger = logging.getLogger(__name__)


# TODO(eddiebergman):
# 1. Do we include the results of everything in between?
#    Would be good to include it in the actual results, if wanting plot the learning
#    curve of individual configurations, not just at the gven fidelity point
# 2. Some optimizers prefer to have some kind of interuption mechanism, i.e.
# > config = opt.ask()
# > for step in steps:
# >     result = benchmark.query(config)
# >   decision = opt.tell(result)
# >   if decision == "stop":
# >         break
@dataclass
class Conf:
    t: tuple
    fid: int | float

@dataclass
class Runtime_hist():
    configs: dict[tuple, dict[str, list[int | float]]] = field(default_factory=dict)

    def add_conf(
        self, 
        config: Conf,
        fid_type: str
    ) -> None:
        if config.t not in self.configs:
            self.configs[config.t] = {
                fid_type: [config.fid]
            }
        else:
            if fid_type not in self.configs[config.t]:
                self.configs[config.t][fid_type] = [config.fid]
            else:
                self.configs[config.t][fid_type].append(config.fid)


    def search(
        self, 
        config: Conf,
        # fid_type: str
    ) -> bool:
        if len(self.configs) == 0:
            return False
        # else:
        #     idxs = np.searchsorted(self.configs, config)
        #     if np.any(idxs >= len(self.configs)):
        #         return False
        #     if np.all(self.configs[idxs] == config):
        #         return True
        elif config.t in self.configs:
            return True
        return False

    def get_conf_dict(self) -> dict:
        return self.configs


def _run(
    run: Run,
    *,
    on_error: Literal["raise", "continue"] = "raise",
    progress_bar: bool = False,
    continuations: bool = False,
) -> Run.Report:
    run.set_state(run.State.RUNNING)
    benchmark = run.benchmark.load(run.benchmark)
    opt = run.optimizer(
        problem=run.problem,
        working_directory=Path('./Optimizers_cache'),
        seed=run.seed,
        config_space=benchmark.config_space,
        **run.optimizer_hyperparameters,
    )

    match run.problem.budget:
        case TrialBudget(
            total=budget_total,
            minimum_fidelity_normalized_value=minimum_normalized_fidelity,
        ):
            report = _run_problem_with_trial_budget(
                run=run,
                benchmark=benchmark,
                optimizer=opt,
                budget_total=budget_total,
                on_error=on_error,
                minimum_normalized_fidelity=minimum_normalized_fidelity,
                progress_bar=progress_bar,
                continuations=continuations,
            )
        case CostBudget():
            raise NotImplementedError("CostBudget not yet implemented")
        case _:
            raise RuntimeError(f"Invalid budget type: {run.problem.budget}")

    logger.info(f"COMPLETED running {run.name}")
    logger.info(f"Saving {run.name} at {run.working_dir}")
    run.set_state(Run.State.COMPLETE, df=report.df())
    return report


def _run_problem_with_trial_budget(
    *,
    run: Run,
    optimizer: Optimizer,
    benchmark: Benchmark,
    budget_total: int,
    on_error: Literal["raise", "continue"],
    minimum_normalized_fidelity: float,
    progress_bar: bool,
    continuations: bool = False,
) -> Run.Report:
    used_budget: float = 0.0

    history: list[Result] = []

    if progress_bar:
        ctx = partial(tqdm, desc=f"{run.name}", total=run.problem.budget.total)
    else:
        ctx = partial(nullcontext, None)

    # NOTE(eddiebergman): Ignore the tqdm warning about the progress bar going past max
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=TqdmWarning)
        runhist = Runtime_hist()

        with ctx() as pbar:
            while used_budget < budget_total:
                try:
                    query = optimizer.ask()
                    config = Conf(query.config.to_tuple(run.problem.precision), used_budget)
                    # if continuations and runhist.search(config):
                    #     logger.warning(f"Configuration: {query.config} already evaluated!")
                    runhist.add_conf(
                        config=config,
                        fid_type=run.problem.fidelity[0] #TODO: Manyfidelity not implemented -> turn into error
                    )
                    result = benchmark.query(query)

                    budget_cost = _trial_budget_cost(
                        value=result.fidelity,
                        problem=run.problem,
                        minimum_normalized_fidelity=minimum_normalized_fidelity,
                    )

                    used_budget += budget_cost
                    result.budget_cost = budget_cost
                    result.budget_used_total = used_budget

                    optimizer.tell(result)
                    history.append(result)
                    if pbar is not None:
                        pbar.update(budget_cost)
                except Exception as e:
                    run.set_state(Run.State.CRASHED, err_tb=(e, traceback.format_exc()))
                    logger.exception(e)
                    logger.error(f"Error running {run.name}: {e}")
                    match on_error:
                        case "raise":
                            raise e
                        case "continue":
                            raise NotImplementedError("Continue not yet implemented!") from e
                        case _:
                            raise RuntimeError(f"Invalid value for `on_error`: {on_error}") from e
            # print(runhist.configs)
    return Run.Report(run=run, results=history)


def _trial_budget_cost(
    *,
    value: None | tuple[str, int | float] | Mapping[str, int | float],
    problem: Problem,
    minimum_normalized_fidelity: float,
) -> float:
    problem_fids = problem.fidelity
    match value:
        case None:
            assert problem_fids is None
            return 1

        case (name, v):
            assert isinstance(v, int | float)
            assert isinstance(problem_fids, tuple)
            assert problem_fids[0] == name
            normed_value = rescale(
                v,
                frm=(problem_fids[1].min, problem_fids[1].max),
                to=(minimum_normalized_fidelity, 1),
            )
            assert isinstance(normed_value, float)
            return normed_value

        case Mapping():
            assert isinstance(problem_fids, Mapping)
            assert len(value) == len(problem_fids)
            normed_fidelities: list[float] = []
            for k, v in value.items():
                assert isinstance(v, int | float)
                assert isinstance(problem_fids[k], Fidelity)
                normed_fid = rescale(
                    v,
                    frm=(problem_fids[k].min, problem_fids[k].max),
                    to=(minimum_normalized_fidelity, 1),
                )
                assert isinstance(normed_fid, float)
                normed_fidelities.append(normed_fid)
            return sum(normed_fidelities) / len(value)
        case _:
            raise TypeError("Fidelity must be None, tuple(str, value), or Mapping[str, fid]")
