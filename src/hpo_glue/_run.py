from __future__ import annotations

import logging
import traceback
import warnings
from collections.abc import Collection, Mapping
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Literal

import pandas as pd
from tqdm import TqdmWarning, tqdm

from hpo_glue.budget import CostBudget, TrialBudget
from hpo_glue.fidelity import Fidelity
from hpo_glue.run import Run
from hpo_glue.utils import rescale

if TYPE_CHECKING:
    from hpo_glue.benchmark import Benchmark
    from hpo_glue.experiment import Experiment
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
def _run(
    run: Run,
    experiment: Experiment,
    *,
    overwrite: Collection[Run.State],
    on_error: Literal["raise", "continue"] = "raise",
    progress_bar: bool = True,
) -> Run.Report:
    if on_error not in ("raise", "continue"):
        raise ValueError(f"Invalid value for `on_error`: {on_error}")

    paths = experiment.run_paths(run)
    state = experiment.state(run)
    if state in overwrite:
        logger.info(f"Overwriting {run.name} in `{state=}` at {paths.working_dir}.")
        experiment.set_state(run, Run.State.PENDING)

    if paths.df_path.exists():
        logger.info(f"Loading results for {run.name} from {paths.working_dir}")
        return Run.Report.from_df(
            df=pd.read_parquet(paths.df_path),
            run=run,
        )

    if paths.working_dir.exists():
        raise RuntimeError(
            "The optimizer ran before but no dataframe of results was found at "
            f"{paths.df_path}. Set `overwrite=[{state}]` to rerun problems in this state"
        )

    experiment.set_state(run, Run.State.RUNNING)
    benchmark = run.benchmark.load(run.benchmark)
    opt = run.optimizer(
        problem=run.problem,
        working_directory=paths.working_dir / "optimizer_dir",
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
                experiment=experiment,
                benchmark=benchmark,
                optimizer=opt,
                budget_total=budget_total,
                on_error=on_error,
                minimum_normalized_fidelity=minimum_normalized_fidelity,
                progress_bar=progress_bar,
            )
        case CostBudget():
            raise NotImplementedError("CostBudget not yet implemented")
        case _:
            raise RuntimeError(f"Invalid budget type: {run.problem.budget}")

    logger.info(f"COMPLETED running {run.name}")
    logger.info(f"Saving {run.name} at {paths.working_dir}")
    experiment.set_state(run, Run.State.COMPLETE, df=report.df())
    return report


def _run_problem_with_trial_budget(
    *,
    run: Run,
    experiment: Experiment,
    optimizer: Optimizer,
    benchmark: Benchmark,
    budget_total: int,
    on_error: Literal["raise", "continue"],
    minimum_normalized_fidelity: float,
    progress_bar: bool,
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

        with ctx() as pbar:
            while used_budget < budget_total:
                try:
                    query = optimizer.ask()
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
                    experiment.set_state(run, Run.State.CRASHED, err_tb=(e, traceback.format_exc()))
                    logger.exception(e)
                    logger.error(f"Error running {run.name}: {e}")
                    match on_error:
                        case "raise":
                            raise e
                        case "continue":
                            raise NotImplementedError("Continue not yet implemented!") from e
                        case _:
                            raise RuntimeError(f"Invalid value for `on_error`: {on_error}") from e

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
