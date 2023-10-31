from __future__ import annotations  # Makes things like `dict[str, Any]` work

from abc import ABC
from pathlib import Path
from typing import Any, Callable, ClassVar
from typing_extensions import Self
from ConfigSpace import ConfigurationSpace
import pandas as pd


#### Glue ---------------------------------------
#### Defines the basic types and interfaces as well
#### as having the ability to run an optimizer on a benchmark
class Config:
    id: str  # Some unique identifier
    values: dict[str, Any]  # The actual config values to evaluate


class Query:
    config: Config  # The config to evaluate
    fidelity: Any | dict[str, Any]  # What fidelity to evaluate at


class Result:
    query: Query  # The query that generated this result
    result: dict[str, Any]  # Everything the benchmark can gives us for a query
    # We will handle singling out what thing is the objective as a ProblemStatement


class ProblemStatement:
    name: str
    """The name of the problem statement. This is used to identify the problem statement
    in the results and in the filesystem"""

    space: ConfigurationSpace | list[Config]
    """The space of configs to optimize over.

    * list[Config]-> tabular
    *  ConfigurationSpace -> surrogate
    """

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

    # TODO: Will also need to define some criteria for stopping the optimization.
    # Easiest example is n_trials but more difficult to define is "fidelity_budget"
    # used or "time_budget" used.

    # The properties below are just to advertise what kind of problems we can define
    # with the above properties.
    @property
    def is_tabular(self) -> bool:
        return isinstance(self.space, list)

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
    def ask(self) -> Query:
        ...

    def tell(self, result: Result) -> None:
        ...

    @classmethod
    def for_problem(cls, problem: ProblemStatement, output_path: Path) -> Self:
        ...
        """
        # Create an instance of this optimizer for a given problem statement
        # TODO: Not sure if we want this but I assume someone who implements their optimizer
        # should be the one to set this up as we can't possibly know how to construct their
        # optimizer.
        """


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
        id_key: str,
        config_keys: list[str],
        result_key: list[str],
        fidelity_key: str | list[str] | None = None,
    ) -> None:
        """Args:
        name: The name of the benchmark
        table: The table containing the config and results of the benchmark
        id_key: The key in the table that corresponds to the config id
        config_keys: The keys in the table that correspond to the config
        result_key: The key in the table that corresponds to the result
        fidelity_keys: The keys in the table that correspond to the fidelity.
        """


class SurrogateBenchmark(Benchmark):
    def __init__(
        self,
        name: str,
        space: ConfigurationSpace,
        query_function: Callable[[Query], Result],
    ) -> None:
        ...

    def query(self, query: Query) -> Result:
        return self.query_function(config)


class GLUEReport:
    optimizer_name: str
    benchmark_name: str
    problem_statement: ProblemStatement
    history: list[Result]


class GLUE:
    root: Path

    def run(
        self,
        problem: ProblemStatement,
        optimizer: type[Optimizer],
        benchmark: Benchmark,
    ) -> GLUEReport:
        """Runs an optimizer on a benchmark, returning a report."""
        # Create the optimizer for this problem, giving it a unique place to work from
        optimizer_working_path = (
            self.root / optimizer.name / benchmark.name / problem.name
        )
        opt = optimizer.for_problem(problem=problem, path=optimizer_working_path)

        history: list[Result] = []
        while (
            some_criterion_defined_by_problem_statement
        ):  # e.g. n_trials, duration, etc...
            # TODO: Here we will definitely need some more complex logic once we consider things
            # such as n_workers > 1, contunuing from a checkpoint, etc...
            # Ignore these for now, just specifying that this is likely where this kind of logic
            # would get executed.
            config = opt.ask()
            result = benchmark.query(config)
            history.append(result)
            opt.tell(result)

        return GLUEReport(optimizer.name, benchmark.name, problem, history)


#### Benchmarks Repo ---------------------------------------
#### Contains implementations of benchmarks such that they
#### can be used in GLUE. Likely we should wrap these in functions as we
#### can't construct all tables as soon as we import but
#### It's main responsibility is to be able to create objects
#### that implement the Benchmark interface.


def get_benchmark(name: str, **kwargs) -> Benchmark:
    """Entry point of the repo to get a benchmark."""
    # This will probably look like a big if statement
    # or do some introspection on Benchmark.__subclasses__
    BENCHMARK_FACTORIES = {
        "lcbench-tabular": lcbench_tabular,
        "yahpo": yahpo_surrogate_benchmark,
    }
    factory = BENCHMARK_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown benchmark {name}")

    return factory(**kwargs)


# Tabular is relatively easy
def lcbench_tabular(task_id: str, datadir: Path) -> TabularBenchmark:
    table_for_task = pd.load(...)
    return TabularBenchmark(
        name=f"lcbench-tabular-{task_id}",
        table=table_for_task,
        id_key=...,  # Key in the table to uniquely identify tasks
        config_keys=...,  # Keys in the table that correspond to the config values
        result_key=...,  # Key in the table that corresponds to the result
        fidelity_key=...,  # Key in the table that corresponds to the fidelity (e.g. "epoch")
    )


# Surrogate it a little trickier as each benchmark will have a different
# surrogate model. Each surrogate based benchmark will need it's own
# Callable thing that we pass as `query_function=` to the `SurrogateBenchmark`
def yahpo_surrogate_benchmark(task_id: str, datadir: Path) -> SurrogateBenchmark:
    from yahpo_gym.benchmarks import (
        get_benchmark,
    )

    yahpo_benchmark = get_benchmark(task_id, datadir)

    # See below for the implementation of this class
    return SurrogateBenchmark(
        name=f"yahpo-{task_id}",
        space=yahpo_benchmark.space,
        query_function=YahpoSurrogateQuerier(yahpo_benchmark=yahpo_benchmark),
    )


# ... hence we can not generalize the use of a surrogate model.
class YahpoSurrogateQuerier:
    yahpo_benchmark: YahpoGymBenchmarkThing

    def __call__(self, query: Query) -> Result:
        config = massage_into_yahpo_format(query.config)
        fidelity = also_massage_into_yahpo_format(query.fidelity)

        # Get some arbitrary thing out of this surrogate
        result_dict = self.yahpo_benchmark.get_result(config, fidelity)
        return Result(query=query, result=result_dict)


#### Optimizer Repo ---------------------------------------
#### Contains implementations of optimizers such that they
#### can be used in the GLUE
class SMACOptimizer(Optimizer):
    name: ClassVar[str] = "SMAC"

    def ask(self) -> Query:
        ...

    def tell(self, result: Result) -> None:
        ...

    @classmethod
    def for_problem(cls, problem: ProblemStatement, output_path: Path) -> Self:
        ...


class MyOptimizer(Optimizer):
    name: ClassVar[str] = "my_cool_optimizer"

    def ask(self) -> Query:
        ...

    def tell(self, result: Result) -> None:
        ...

    @classmethod
    def for_problem(cls, problem: ProblemStatement, output_path: Path) -> Self:
        ...


#### Experiments Repo ---------------------------------------
### Goal 1 will be able to run an optmizer on a benchmark and get a report
### Later we will work on larger scale setups to automate a lot of the manual
### process required to set up and run GLUE.
### This repo will likely contain a bunch of preset problem statements we can make
### from the benchmarks we know exist.
yahpo_cifar10 = get_benchmark("yahpo", task_id="cifar10", datadir=Path("..."))

# Defining problem statements that people could import and run glue with
# Note how we can define many problems given a single benchmark
# Also note how the benchmark itself is not part of the problem statement. This
# is purposefully.
# TODO: I don't know if we want the definition of problems to live with benchmarks.
# I like the idea of keeping them seperate so we can define new problems without
# having to go into the benchmark library
single_obj = (
    ProblemStatement(
        name="yahpo-cifar10-single-obj-loss",
        space=yahpo_space,
        result_keys="loss",
    ),
)
multi_fidelity = ProblemStatement(
    name="yahpo-cifar10-multi-fidelity-loss",
    space=yahpo_space,
    result_keys="loss",
    fidelity_keys="epoch",
)
multi_obj = ProblemStatement(
    name="yahpo-cifar10-multi-obj-loss-accuracy",
    space=yahpo_space,
    result_keys=["loss", "accuracy"],
    minimize=[True, False],
)

# Likely need to set a bunch of control parameters here that aren't directly part of the
# problem statement, such as paths, where to store things etc...
glue = GLUE(root="some_path_it_should_operate_out_of")

for problem in [single_obj, multi_fidelity, multi_obj]:
    report = glue.run(problem=problem, benchmark=yahpo_cifar10, optimizer=SMACOptimizer)
    report.save(where=glue.root / "reports")
