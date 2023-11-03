from __future__ import annotations  # Makes things like `dict[str, Any]` work

from abc import ABC
from pathlib import Path
from typing import Any, Callable, ClassVar
from typing_extensions import Self
from ConfigSpace import ConfigurationSpace, Configuration
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Config:
    id: str  # Some unique identifier
    values: dict[str, Any]  # The actual config values to evaluate

    def __init__(self, id: str, values: Configuration | dict[str, Any]):
        self.id = id
        if isinstance(values, Configuration):
            self.values = values.get_dictionary()
        else:
            self.values = values


class Query:
    config: Config  # The config to evaluate
    fidelity: Any | dict[str, Any]  # What fidelity to evaluate at

    def __init__(self, config: Config, fidelity: Any | dict[str, Any]):
        self.config = config
        self.fidelity = fidelity


class Result:
    query: Query  # The query that generated this result
    result: dict[str, Any]  # Everything the benchmark can gives us for a query
    # We will handle singling out what thing is the objective as a ProblemStatement

    def __init__(self, query: Query, result: dict[str, Any]):
        self.query = query
        self.result = result


class Hist_Storage(ABC):
    def create_run_dir(self) -> None:
        ...

    def save_config(self) -> None:
        ...

    def save_results(self, result: Result) -> None:
        ...


class Optimizer(ABC):
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

    table: pd.DataFrame  # The table holding all information
    configs: list[Config]  # All possible configs for the benchmark
    fidelities: list[int]  # All possible fidelities for the benchmark

    def __init__(
        self,
        name: str,
        table: pd.DataFrame | Any,  
        # Assuming that the table is a pandas dataframe
        #  or a type specific to the Benchmark being queried

        id_key: str,
        config_keys: list[str],
        result_keys: list[str],
        fidelity_key: str | list[str] | None = None,
    ) -> None:
        self.name = name
        self.table = table
        self.id_key = id_key
        self.config_keys = config_keys
        self.result_keys = result_keys
        self.fidelity_key = fidelity_key
        self._get_all_configs()
        self._get_all_fidelities()

    def query(self, query: Query) -> Result:
        """Query the benchmark for a result"""
        max_fidelity = self.fidelities[-1]
        at = None
        if query.fidelity is not None:
            at = query.fidelity
        else:   at = max_fidelity
        result = self.table.loc[(query.config.id, at)]
        result = result.get(self.result_keys).to_dict()
        return Result(query, result)
    
    def _get_all_configs(self) -> None:
        """Get all possible configs for the benchmark"""
        all_configs = []
        for id, data in self.table.iterrows():
            config_data = data.get(self.config_keys).to_dict()
            config = Config(id[0], config_data)
            all_configs.append(config)
        self.configs = all_configs

    def _get_all_fidelities(self) -> None:
        """Get all possible fidelities for the benchmark"""
        all_fidelities = self.table.index.get_level_values(self.fidelity_key).unique()
        self.fidelities = all_fidelities.sort_values()

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
        # problem: ProblemStatement,
        optimizer: type[Optimizer],
        benchmark: Benchmark,
        budget: int, # number of trials
        is_tabular: bool = False,
    ) -> GLUEReport:
        """Runs an optimizer on a benchmark, returning a report."""
        trial = 0
        history: list[Result] = []
        opt = optimizer(config_space=benchmark.configs,
                        fidelity_space=benchmark.fidelities)
        while (
            trial<budget
        ):  # e.g. n_trials, duration, etc...
            # TODO: Here we will definitely need some more complex logic once we consider things
            # such as n_workers > 1, contunuing from a checkpoint, etc...
            # Ignore these for now, just specifying that this is likely where this kind of logic
            # would get executed.
            logger.info(f"\nTrial {trial}\n")
            print("-------------------------------")
            config = opt.ask(is_tabular=is_tabular)
            result = benchmark.query(config)
            history.append(result)
            opt.tell(result)
            trial += 1
            logger.info(result.result)
            print("-------------------------------\n")


        return GLUEReport(optimizer.name, benchmark.name, history)
