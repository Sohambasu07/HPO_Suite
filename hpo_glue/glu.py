from __future__ import annotations  # Makes things like `dict[str, Any]` work

from abc import ABC
from pathlib import Path
from typing import Any, Callable
from ConfigSpace import ConfigurationSpace, Configuration
import pandas as pd
import os
import datetime

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Config:
    id: str  
    """ Some unique identifier """
    
    values: dict[str, Any]  
    """ The actual config values to evaluate """

    def __init__(self, id: str, values: Configuration | dict[str, Any]):
        self.id = id
        if isinstance(values, Configuration):
            self.values = values.get_dictionary()
        else:
            self.values = values


class Query:
    config: Config  
    """ The config to evaluate """
    
    fidelity: Any | dict[str, Any]  
    """ What fidelity to evaluate at """

    def __init__(self, config: Config, fidelity: Any | dict[str, Any]):
        self.config = config
        self.fidelity = fidelity


class Result:
    """The result of a query from a benchmark."""

    query: Query
    """The query that generated this result"""


    result: dict[str, Any]
    """Everything returned by the benchmark for a given query."""
    # TODO: We will handle singling out what thing is the objective as a ProblemStatement

    def __init__(self, query: Query, result: dict[str, Any]):
        self.query = query
        self.result = result


class History:
    """Abstract Class for storing the history of an optimizer run."""

    results: list[Result]

    def __init__(self) -> None:
        self.results = []

    def add(self, result: Result) -> None:
        self.results.append(result)

    def df(self, columns) -> pd.DataFrame:
        """Return the history as a pandas DataFrame"""

        report = []
        
        for res in self.results:
            config = res.query.config.values
            id = res.query.config.id
            fidelity = res.query.fidelity
            result = res.result
            report.append([id, fidelity])
            report[-1].extend([val for key, val in config.items()])
            report[-1].extend([val for key, val in result.items()])

        hist_df = pd.DataFrame(report, columns=columns)
        return hist_df

    def _save(self, report: pd.DataFrame, savedir: Path) -> None:
        """ Save the history of the run """
        if os.path.exists(savedir) is False:
            os.mkdir(savedir)
        filename = "report" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
        report.to_csv(savedir / filename, mode='a', header=True, index=False)


class ProblemStatement:
    name: str
    """The name of the problem statement. This is used to identify the problem statement
    in the results and in the filesystem"""

    config_space: ConfigurationSpace | list[Config]
    """The space of configs to optimize over.

    * list[Config]-> tabular
    *  ConfigurationSpace -> surrogate
    """

    fidelity_space: ConfigurationSpace | list[int] | list[float]
    """The space of fidelities for Multifidelity/ Manyfidelity Optimization"""

    result_keys: str | list[str]
    """The key(s) in the result that we want to consider as the objective value

    * str -> single objective
    * list[str] -> multi-objective
    """

    fidelity_keys: str | list[str] | None
    """The key(s) in the result that we want to consider as the fidelity

    * str -> single fidelity parameter
    * list[str] -> many fidelity parameters
    * None -> no fidelity
    """

    minimize: bool | list[bool]
    """Whether to minimize or maximize the objective value. One per objective"""

    n_trials: int
    """The number of trials to run the optimizer for"""

    def __init__(
        self,
        name: str,
        config_space: ConfigurationSpace | list[Config],
        fidelity_space: ConfigurationSpace | list[int] | list[float],
        result_keys: str | list[str],
        fidelity_keys: str | list[str] | None = None,
        minimize: bool | list[bool] = True,
        n_trials: int = 1,
    ) -> None:
        cls = self.__class__
        cls.name = name
        cls.config_space = config_space
        cls.fidelity_space = fidelity_space
        cls.result_keys = result_keys
        cls.fidelity_keys = fidelity_keys
        cls.minimize = minimize
        cls.n_trials = n_trials

    # TODO: Will also need to define some criteria for stopping the optimization.
    # Easiest example is n_trials but more difficult to define is "fidelity_budget"
    # used or "time_budget" used.

    # The properties below are just to advertise what kind of problems we can define
    # with the above properties.
    @property
    def is_tabular(self) -> bool:
        return isinstance(self.config_space, list)

    @property
    def is_multiobjective(self) -> bool:
        return isinstance(self.result_keys, list)

    @property
    def is_multifidelity(self) -> bool:
        return isinstance(self.fidelity_keys, str)

    @property
    def is_manyfidelity(self) -> bool:
        return isinstance(self.fidelity_keys, list)


class Optimizer(ABC):
    """ Defines the common interface for Optimizers """

    def __init__(
        self,
        config_space: ConfigurationSpace | list[Config],
        fidelity_space: ConfigurationSpace | list[int] | list[float],
    ) -> None:
        ...

    def get_incumbent(self) -> Any:
        ...

    def ask(self) -> Query:
        ...

    def tell(self, result: Result) -> None:
        ...


class Benchmark(ABC):
    """Defines the common interface between tabular and surrogate benchmarks."""

    def __init__(self, name: str) -> None:
        ...

    def query(self, query: Query) -> Result:
        ...



class TabularBenchmark(Benchmark):
    """Defines the interface for a tabular benchmark."""

    table: pd.DataFrame
    """ The table holding all information """

    config_space: list[Config]  
    """ All possible configs for the benchmark """

    fidelity_space: list[int] | list[float]  
    """ All possible fidelities for the benchmark """

    def __init__(
        self,
        name: str,
        table: pd.DataFrame,
        id_key: str,
        config_keys: list[str],
        result_keys: list[str],
        fidelity_key: str | list[str] | None = None,
        remove_constants: bool = False,
        space: ConfigurationSpace | None = None,
        seed: int | None = None,
    ) -> None:
        

        # Make sure we work with a clean slate, no issue with index.
        table = table.reset_index()

        # Make sure all the keys they specified exist
        if id_key not in table.columns:
            raise ValueError(f"'{id_key=}' not in columns {table.columns}")

        if fidelity_key not in table.columns:
            raise ValueError(f"'{fidelity_key=}' not in columns {table.columns}")

        if not all(key in table.columns for key in result_keys):
            raise ValueError(f"{result_keys=} not in columns {table.columns}")

        if not all(key in table.columns for key in config_keys):
            raise ValueError(f"{config_keys=} not in columns {table.columns}")

        # Make sure that the column `id` only exist if it's the `id_key`
        if "id" in table.columns and id_key != "id":
            raise ValueError(
                f"Can't have `id` in the columns if it's not the {id_key=}."
                " Please drop it or rename it.",
            )

        # Remove constants from the table
        if remove_constants:

            def is_constant(_s: pd.Series) -> bool:
                _arr = _s.to_numpy()
                return bool((_arr == _arr[0]).all())

            constant_cols = [
                col for col in table.columns if is_constant(table[col])  # type: ignore
            ]
            table = table.drop(columns=constant_cols)  # type: ignore
            config_keys = [k for k in config_keys if k not in constant_cols]

        # Remap their id column to `id`
        table = table.rename(columns={id_key: "id"})

        # Index the table
        index_cols: list[str] = ["id", fidelity_key]

        # Drop all the columns that are not relevant
        relevant_cols: list[str] = [
            *index_cols,
            *result_keys,
            *config_keys,
        ]
        table = table[relevant_cols]  # type: ignore
        table = table.set_index(index_cols).sort_index()

        # We now have the following table
        #
        #     id    fidelity | **metric, **config_values
        #     0         0    |
        #               1    |
        #               2    |
        #     1         0    |
        #               1    |
        #               2    |
        #   ...

        # Create the configuration space
        if space is None:
            space = ConfigurationSpace(name, seed=seed)


        self.name = name
        self.table = table
        self.id_key = id_key
        self.fidelity_key = fidelity_key
        self.config_keys = sorted(config_keys)
        self.result_keys = sorted(result_keys)
        self.config_space = self._get_all_configs()
        self.fidelity_space, self.fidelity_range = self._get_all_fidelities()
        

    def query(self, query: Query) -> Result:
        """Query the benchmark for a result"""

        at = None
        if query.fidelity is not None:
            at = query.fidelity
        else:   
            at = self.fidelity_range[1]
        result = self.table.loc[(query.config.id, at)]
        result = result.get(self.result_keys).to_dict()
        return Result(query, result)
    
    def _get_all_configs(self) -> None:
        """Get all possible configs for the benchmark"""

        return [
            Config(str(i), config)  #enforcing str for id
            for i, config in enumerate(
                self.table[self.config_keys]
                .drop_duplicates()
                # Sorting is important to make sure it's always consistent
                .sort_values(by=self.config_keys) 
                .to_dict(orient="records")
            )
        ]

    def _get_all_fidelities(self) -> None:
        """Get all possible fidelities for the benchmark"""

        # Make sure we have equidistance fidelities for all configs
        fidelity_values = self.table.index.get_level_values(self.fidelity_key)
        fidelity_counts = fidelity_values.value_counts()
        if not (fidelity_counts == fidelity_counts.iloc[0]).all():
            raise ValueError(f"{self.fidelity_key=} not uniform. \n{fidelity_counts}")


        sorted_fids = sorted(fidelity_values.unique())
        start = sorted_fids[0]
        end = sorted_fids[-1]
        step = sorted_fids[1] - sorted_fids[0]

        # Here we get all the unique configs
        #     id    fidelity | **metric, **config_values
        #     0         0    |
        #     1         0    |
        #   ...

        return sorted_fids, (start, end, step)
    

class SurrogateBenchmark(Benchmark):
    def __init__(
        self,
        name: str,
        config_space: ConfigurationSpace,
        fidelity_space: list[int] | list[float],
        query_function: Callable[[Query], Result],
        benchmark
    ) -> None:
        self.name = name
        self.config_space = config_space
        self.fidelity_space = fidelity_space
        self.query_function = query_function
        self.benchmark = benchmark

    def query(self, query: Query) -> Result:
         result = self.query_function(self.benchmark, query)
         return Result(query, result)


class GLUEReport:
    optimizer_name: str
    benchmark_name: str
    # problem_statement: ProblemStatement
    history: list[Result]

    def __init__(
        self, 
        optimizer_name: str, 
        benchmark_name: str, 
        history: list[Result]
    ) -> None:
        self.optimizer_name = optimizer_name
        self.benchmark_name = benchmark_name
        self.history = history


class GLUE:
    root: Path

    def run(
        problem: ProblemStatement,
        optimizer: type[Optimizer],
        benchmark: Benchmark,
        budget: int, # number of trials
        save_dir: Path,
        seed: int | None = None,
    ) -> GLUEReport:
        """Runs an optimizer on a benchmark, returning a report."""
        trial = 0
        history = History()
        GLUE.root = Path('./')

        optimizer_working_path = (
            GLUE.root / optimizer.name / benchmark.name / problem.name
        )
        opt = optimizer(ProblemStatement=problem, working_directory=optimizer_working_path)


        while (
            trial<budget
        ):  # e.g. n_trials, duration, etc...
            # TODO: Here we will definitely need some more complex logic once we consider things
            # such as n_workers > 1, contunuing from a checkpoint, etc...
            # Ignore these for now, just specifying that this is likely where this kind of logic
            # would get executed.
            logger.info(f"Trial {trial}\n")
            print("-------------------------------")
            config = opt.ask(config_id=str(trial))
            result = benchmark.query(config)
            history.add(result)
            opt.tell(result)
            trial += 1
            logger.info(result.result)
            logger.info(opt.get_incumbent())

            print("-------------------------------\n")

        cols = (
            ["Config id", "Fidelity"]
            + list(result.query.config.values.keys())
            + list(result.result.keys())
        )

        report = history.df(cols)
        report["Optimizer Name"] = optimizer.name
        report["Benchmark Name"] = benchmark.name

        history._save(report, save_dir)
        print(report)
            

        return GLUEReport(optimizer.name, benchmark.name, history)
