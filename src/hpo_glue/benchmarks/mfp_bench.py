"""Script to interface with the MF Prior Bench library."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mfpbench

from hpo_glue.benchmark import BenchmarkFactory, SurrogateBenchmark, TabularBenchmark
from hpo_glue.result import Result

if TYPE_CHECKING:
    from hpo_glue.query import Query

# NOTE(eddiebergman): MFPbench doesn't ever consider a time budget so I just made
# a look up here instead.
_time_budget_lookup: Mapping[type[mfpbench.Benchmark], str] = {
    mfpbench.IAMLglmnetBenchmark: "timetrain",
    mfpbench.IAMLrangerBenchmark: "timetrain",
    mfpbench.IAMLrpartBenchmark: "timetrain",
    mfpbench.IAMLSuperBenchmark: "timetrain",
    mfpbench.IAMLxgboostBenchmark: "timetrain",
    mfpbench.LCBenchBenchmark: "time",
    mfpbench.NB301Benchmark: "runtime",
    mfpbench.RBV2aknnBenchmark: "timetrain",
    mfpbench.RBV2glmnetBenchmark: "timetrain",
    mfpbench.RBV2rangerBenchmark: "timetrain",
    mfpbench.RBV2rpartBenchmark: "timetrain",
    mfpbench.RBV2SuperBenchmark: "timetrain",
    mfpbench.RBV2svmBenchmark: "timetrain",
    mfpbench.RBV2xgboostBenchmark: "timetrain",
    mfpbench.JAHSBenchmark: "runtime",
    mfpbench.LCBenchBenchmark: "time",
    mfpbench.LCBenchTabularBenchmark: "time",
    mfpbench.MFHartmann3BenchmarkBad: "fid_cost",
    mfpbench.MFHartmann3BenchmarkGood: "fid_cost",
    mfpbench.MFHartmann3BenchmarkModerate: "fid_cost",
    mfpbench.MFHartmann3BenchmarkTerrible: "fid_cost",
    mfpbench.MFHartmann6BenchmarkBad: "fid_cost",
    mfpbench.MFHartmann6BenchmarkGood: "fid_cost",
    mfpbench.MFHartmann6BenchmarkModerate: "fid_cost",
    mfpbench.MFHartmann6BenchmarkTerrible: "fid_cost",
    mfpbench.NB301Benchmark: "runtime",
    mfpbench.PD1cifar100_wideresnet_2048: "train_cost",
    mfpbench.PD1imagenet_resnet_512: "train_cost",
    mfpbench.PD1lm1b_transformer_2048: "train_cost",
    mfpbench.PD1translatewmt_xformer_64: "train_cost",
    # NOTE(eddiebergman): Its not a good one...
    # mfpbench.PD1uniref50_transformer_128: "train_cost"8,
}


def _mfpbench_surrogate_query_function(query: Query, benchmark: mfpbench.Benchmark) -> Result:
    return Result(
        query=query,
        result=benchmark.query(query.config.values, at=query.fidelity).as_dict(),
    )


def _lcbench_tabular(
    unique_name: str,
    task_id: str,
    datadir: Path | str | None = None,
) -> TabularBenchmark:
    if isinstance(datadir, str):
        datadir = Path(datadir).absolute().resolve()

    if datadir is None:
        datadir = Path("data", "lcbench-tabular").absolute().resolve()

    bench = mfpbench.LCBenchTabularBenchmark(
        task_id=task_id,
        datadir=datadir,
        remove_constants=True,
    )
    result_type = mfpbench.LCBenchTabularResult
    return TabularBenchmark(
        name=unique_name,
        table=bench.table,
        id_key="id",  # Key in the table to uniquely identify configs
        config_keys=bench.config_keys,  # Keys in the table that correspond to configs
        result_keys=bench.result_keys,  # Key in the table that corresponds to the result
        fidelity_keys=bench.fidelity_key,  # Key that corresponds to the fidelity (e.g. "epoch")
        default_objective=result_type.default_value_metric,  # Default objective to optimize
        time_budget="time",  # Time budget key
        minimize_default=True,  # Whether the default objective should be minimized
    )


def _get_surrogate_benchmark(
    unique_name: str,
    benchmark_name: str,
    **kwargs: Any,
) -> SurrogateBenchmark:
    bench = mfpbench.get(benchmark_name, **kwargs)
    query_function = partial(_mfpbench_surrogate_query_function, benchmark=bench)

    default_metric = bench.Result.default_value_metric
    time_budget_name = _time_budget_lookup.get(type(bench), None)
    return SurrogateBenchmark(
        name=unique_name,
        config_space=bench.space,
        fidelity_space=list(bench.iter_fidelities()),
        result_keys=dir(bench.Result),
        fidelity_keys=bench.fidelity_name,
        query_function=query_function,
        benchmark=bench,
        time_budget=time_budget_name,
        default_objective=default_metric,  # Default objective to optimize
        minimize_default=bench.Result.metric_defs[default_metric].minimize,
    )


_mfp_benchmarks = (
    mfpbench.IAMLglmnetBenchmark,
    mfpbench.IAMLrangerBenchmark,
    mfpbench.IAMLrpartBenchmark,
    mfpbench.IAMLSuperBenchmark,
    mfpbench.IAMLxgboostBenchmark,
    mfpbench.LCBenchBenchmark,
    mfpbench.LCBenchTabularBenchmark,
    mfpbench.MFHartmann3BenchmarkBad,
    mfpbench.MFHartmann3BenchmarkGood,
    mfpbench.MFHartmann3BenchmarkModerate,
    mfpbench.MFHartmann3BenchmarkTerrible,
    mfpbench.MFHartmann6BenchmarkBad,
    mfpbench.MFHartmann6BenchmarkGood,
    mfpbench.MFHartmann6BenchmarkModerate,
    mfpbench.MFHartmann6BenchmarkTerrible,
    mfpbench.JAHSBenchmark,
    mfpbench.NB301Benchmark,
    mfpbench.PD1cifar100_wideresnet_2048,
    mfpbench.PD1imagenet_resnet_512,
    mfpbench.PD1lm1b_transformer_2048,
    mfpbench.PD1translatewmt_xformer_64,
    mfpbench.PD1uniref50_transformer_128,
    mfpbench.RBV2aknnBenchmark,
    mfpbench.RBV2glmnetBenchmark,
    mfpbench.RBV2rangerBenchmark,
    mfpbench.RBV2rpartBenchmark,
    mfpbench.RBV2SuperBenchmark,
    mfpbench.RBV2svmBenchmark,
    mfpbench.RBV2xgboostBenchmark,
)


def _get_benchmark_factories() -> list[BenchmarkFactory]:
    factories: list[BenchmarkFactory] = []
    for benchmark in _mfp_benchmarks:
        if issubclass(benchmark, mfpbench.YAHPOBenchmark):
            assert benchmark.yahpo_instances is not None
            has_conditionals = benchmark.yahpo_has_conditionals
            if has_conditionals:
                # TODO(eddiebergman): Raises a bug in mfpbench
                continue

            factories.extend(
                [
                    BenchmarkFactory(
                        _get_surrogate_benchmark,
                        unique_name=f"yahpo_{benchmark.yahpo_base_benchmark_name}-{task_id}",
                        is_tabular=False,
                        supports_multifidelity=True,
                        supports_multiobjective=True,
                        supports_manyfidelity=False,
                        has_conditionals=has_conditionals,
                        kwargs={
                            "unique_name": f"yahpo_{benchmark.yahpo_base_benchmark_name}-{task_id}",
                            "benchmark_name": benchmark.yahpo_base_benchmark_name,
                            "task_id": task_id,
                            "datadir": Path("data", "yahpo"),
                        },
                    )
                    for task_id in benchmark.yahpo_instances
                ]
            )
        elif issubclass(benchmark, mfpbench.LCBenchTabularBenchmark):
            factories.extend(
                [
                    BenchmarkFactory(
                        _lcbench_tabular,
                        unique_name=f"lcbench_tabular-{task_id}",
                        has_conditionals=False,
                        is_tabular=True,
                        supports_multifidelity=True,
                        supports_manyfidelity=False,
                        supports_multiobjective=False,
                        kwargs={
                            "unique_name": f"lcbench_tabular-{task_id}",
                            "task_id": task_id,
                            "datadir": Path("data", "lcbench-tabular"),
                        },
                    )
                    for task_id in benchmark.task_ids
                ]
            )
        elif issubclass(benchmark, mfpbench.MFHartmannBenchmark):
            name = f"mfh{benchmark.mfh_dims}_{benchmark.mfh_suffix}"
            factories.append(
                BenchmarkFactory(
                    _get_surrogate_benchmark,
                    unique_name=name,
                    has_conditionals=False,
                    is_tabular=False,
                    supports_multifidelity=True,
                    supports_manyfidelity=False,
                    supports_multiobjective=False,
                    kwargs={"unique_name": name, "benchmark_name": name},
                )
            )
        elif issubclass(benchmark, mfpbench.JAHSBenchmark):
            for task_id in benchmark.task_ids:
                factories.append(
                    BenchmarkFactory(
                        _get_surrogate_benchmark,
                        unique_name=f"jahs-{task_id}",
                        has_conditionals=False,
                        is_tabular=False,
                        supports_multifidelity=True,
                        supports_manyfidelity=False,
                        supports_multiobjective=False,
                        kwargs={
                            "unique_name": f"jahs-{task_id}",
                            "benchmark_name": "jahs",
                            "task_id": task_id,
                            "datadir": Path("data", "jahs"),
                        },
                    )
                )
        elif issubclass(benchmark, mfpbench.PD1Benchmark):
            factories.append(
                BenchmarkFactory(
                    _get_surrogate_benchmark,
                    unique_name=f"pd1-{benchmark.pd1_name}",
                    has_conditionals=False,
                    is_tabular=False,
                    supports_multifidelity=True,
                    supports_manyfidelity=False,
                    supports_multiobjective=False,
                    kwargs={
                        "unique_name": f"pd1-{benchmark.pd1_name}",
                        "benchmark_name": benchmark.pd1_name,
                        "datadir": Path("data", "pd1"),
                    },
                )
            )

    return factories
