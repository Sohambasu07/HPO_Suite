from __future__ import annotations

import logging
import traceback
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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
class Runtime_hist:
    configs: dict[tuple, dict[str, list[int | float]]] = field(default_factory=dict)

    def add_conf(
        self,
        config: Conf,
        fid_type: str
    ) -> int:
        flag = 0
        if config.t not in self.configs:
            self.configs[config.t] = {
                fid_type: [config.fid]
            }
        elif fid_type not in self.configs[config.t]:
            self.configs[config.t][fid_type] = [config.fid]
        elif config.fid in self.configs[config.t][fid_type]:
            warnings.warn(
                f"Fidelity {config.fid} sampled twice by Optimizer for config {config.t}!",
                stacklevel=2
            )
            flag = 1
        elif config.fid < self.configs[config.t][fid_type][-1]:
            raise NotImplementedError("Decreasing fidelity not yet implemented!")
        elif config.fid not in self.configs[config.t][fid_type]:
            self.configs[config.t][fid_type].append(config.fid)
        return flag

    def get_continuations_cost(
            self,
            config: Conf,
            fid_type: str
    ) -> float:
        fid_list = self.configs[config.t][fid_type]
        if len(fid_list) == 1:
            return fid_list[0]
        return fid_list[-1] - fid_list[-2]

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
        working_directory=Path("./Optimizers_cache"),
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
    logger.info(f"Results dumped at {run.df_path.absolute()}")
    run.set_state(Run.State.COMPLETE, df=report.df())
    return report


def _run_problem_with_trial_budget(  # noqa: C901, PLR0912
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

        tuple_configs: dict[tuple, dict[str, list[int | float]]]

        with ctx() as pbar:
            while used_budget < budget_total:
                try:
                    query = optimizer.ask()
                    config = Conf(query.config.to_tuple(run.problem.precision), query.fidelity[1])

                    resample_flag = False
                    flag = runhist.add_conf(
                        config=config,
                        fid_type=run.problem.fidelity[0] #TODO: Raise Manyfidelity NotImplementedError
                    )
                    if flag == 1:
                        resample_flag = True

                    if resample_flag: # NOTE: Not a cheap operation since we don't store the costs in the continuations dict
                        for res in history:
                            if Conf(res.config.to_tuple(run.problem.precision), res.fidelity[1]) == config:  # noqa: E501
                                result = res
                                result.query = query
                    else:
                        result = benchmark.query(query)
                        if continuations:
                            result.continuations_cost = runhist.get_continuations_cost(
                                config=config,
                                fid_type=run.problem.fidelity[0]
                            )

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
                    tuple_configs = runhist.get_conf_dict()
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
    return Run.Report(run=run, results=history, tuple_configs_dict=tuple_configs)


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
