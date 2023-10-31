from __future__ import annotations  # Makes things like `dict[str, Any]` work

from abc import ABC
from pathlib import Path
from typing import Any, Callable, ClassVar
from typing_extensions import Self
from ConfigSpace import ConfigurationSpace, Configuration
import pandas as pd
from optimizers.random_search import RandomSearch
from storage import Storage
from benchmarks.lcbench import LCBenchSetup, LCBenchTabular

class Config:
    id: str  # Some unique identifier
    values: dict[str, Any]  # The actual config values to evaluate

    def __init__(self, id: str, values: Configuration):
        self.id = id
        self.values = values.get_dictionary()


class Query:
    config: Config  # The config to evaluate
    # fidelity: Any | dict[str, Any]  # What fidelity to evaluate at

    def __init__(self, config: Config):
        self.config = config
        # self.fidelity = fidelity


class Result:
    query: Query  # The query that generated this result
    result: dict[str, Any]  # Everything the benchmark can gives us for a query
    # We will handle singling out what thing is the objective as a ProblemStatement

    def __init__(self, query: Query, result: dict[str, Any]):
        self.query = query
        self.result = result


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

    def __init__(
        self,
        name: str,
        table: pd.DataFrame,
    ) -> None:
        self.name = name
        self.table = table

    def query(self, query: Query) -> Result:
        """Query the benchmark for a result"""
        