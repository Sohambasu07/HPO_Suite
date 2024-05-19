from __future__ import annotations

import logging
import shutil
import warnings
from pathlib import Path

import pandas as pd
from tqdm import TqdmWarning, tqdm

from hpo_glue.history import History
from hpo_glue.problem import Problem

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
def run_problem(
    problem: Problem,
    *,
    expdir: Path | str = "hpo-glue-output",
    save: bool = True,
    overwrite: bool = False,
) -> Problem.Report:
    """Runs an optimizer on a benchmark, returning a report."""
    path = Path(expdir) / problem.name
    df_path = Path(expdir) / "dfs" / f"{problem.name}.parquet"
    if overwrite:
        logger.info(f"Overwriting {problem.name} at {path} as `overwrite=True` was set.")
        if path.exists():
            try:
                shutil.rmtree(path)
            except Exception as e:
                logger.exception(e)
                logger.error(f"Error deleting {path}: {e}")
        if df_path.exists():
            try:
                df_path.unlink()
            except Exception as e:
                logger.exception(e)
                logger.error(f"Error deleting {df_path}: {e}")
    elif path.exists():
        if df_path.exists():
            logger.info(f"Loading results for {problem.name} from {path}")
            return Problem.Report.from_df(df=pd.read_parquet(df_path), problem=problem)
        raise RuntimeError(
            "The optimizer ran before but no dataframe of results was found at "
            f"{df_path}. Set `overwrite=True` to rerun the optimizer."
        )

    df_path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)

    benchmark = problem.benchmark.load(problem.benchmark)

    history = History()

    opt = problem.optimizer(
        problem=problem,
        working_directory=path / "optimizer_dir",
        config_space=benchmark.config_space,
        **problem.optimizer_hyperparameters,
    )

    budget_tracker = problem.budget.clone()

    # NOTE(eddiebergman): Ignore the tqdm warning about the progress bar going past max
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=TqdmWarning)

        with tqdm(desc=f"{problem.name}", total=problem.budget.budget) as pbar:
            while not budget_tracker.should_stop():
                config = opt.ask()
                result = benchmark.query(config)
                opt.tell(result)

                history.add(result)
                used_budget = budget_tracker.update(result=result, problem=problem)
                pbar.update(used_budget)

    report = Problem.Report(problem=problem, history=history)
    if save:
        logger.info(f"Saving {problem.name} at {path}")
        report.save(path=df_path)
        loaded = Problem.Report.from_df(df=pd.read_parquet(df_path), problem=problem)
        pd.testing.assert_frame_equal(loaded.df(), report.df())

    return report
