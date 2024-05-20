from __future__ import annotations

import logging
import traceback
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import pandas as pd
from tqdm import TqdmWarning, tqdm

from hpo_glue.problem import Problem

if TYPE_CHECKING:
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
def _run_problem(
    problem: Problem,
    *,
    overwrite: Problem.State | str | Sequence[Problem.State | str] | bool = False,
    on_error: Literal["raise", "continue"] = "raise",
) -> Problem.Report:
    if on_error not in ("raise", "continue"):
        raise ValueError(f"Invalid value for `on_error`: {on_error}")

    state = problem.state()
    if state in Problem.State.collect(overwrite):
        logger.info(f"Overwriting {problem.name} in `{state=}` at {problem.working_dir}.")
        problem.set_state(Problem.State.PENDING)

    if problem.df_path.exists():
        logger.info(f"Loading results for {problem.name} from {problem.working_dir}")
        return Problem.Report.from_df(
            df=pd.read_parquet(problem.df_path),
            problem=problem,
        )

    if problem.working_dir.exists():
        raise RuntimeError(
            "The optimizer ran before but no dataframe of results was found at "
            f"{problem.df_path}. Set `overwrite=[{state}]` to rerun problems in this state"
        )

    problem.set_state(Problem.State.RUNNING)
    benchmark = problem.benchmark.load(problem.benchmark)

    history: list[Result] = []

    opt = problem.optimizer(
        problem=problem,
        working_directory=problem.working_dir / "optimizer_dir",
        config_space=benchmark.config_space,
        **problem.optimizer_hyperparameters,
    )

    budget_tracker = problem.budget.clone()

    # NOTE(eddiebergman): Ignore the tqdm warning about the progress bar going past max
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=TqdmWarning)

        try:
            with tqdm(desc=f"{problem.name}", total=problem.budget.total) as pbar:
                while not budget_tracker.should_stop():
                    config = opt.ask()
                    result = benchmark.query(config)
                    budget_increment = budget_tracker.update(result=result, problem=problem)

                    result.budget_cost = budget_increment
                    opt.tell(result)

                    history.append(result)

                    pbar.update(budget_increment)

        except Exception as e:  # noqa: BLE001
            problem.set_state(Problem.State.CRASHED, err_tb=(e, traceback.format_exc()))
            match on_error:
                case "raise":
                    raise e
                case "continue":
                    logger.exception(e)
                    logger.error(f"Error running {problem.name}: {e}")
                case _:
                    raise RuntimeError(f"Invalid value for `on_error`: {on_error}") from e

    logger.info(f"COMPLETED running {problem.name}")

    report = Problem.Report(problem=problem, results=history)
    logger.info(f"Saving {problem.name} at {problem.working_dir}")
    problem.set_state(Problem.State.COMPLETE, df=report.df())
    return report
