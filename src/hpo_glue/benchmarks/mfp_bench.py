"""Script to interface with the MF Prior Bench library."""
# TODO(eddiebergman): Right now it's not clear how to set defaults for multi-objective.
# Do we want to prioritize obj and cost (i.e. accuracy and time) or would we rather two
# objectives (i.e. accuracy and cross entropy)?
# Second, for a benchmark to be useful, it should provide a reference point from which to compute
# hypervolume. For bounded costs this is fine but we can't do so for something like time.
# For tabular datasets, we could manually look for the worst time value
# TODO(eddiebergman): Have not included any of the conditional benchmarks for the moment
# as it seems to crash
# > "nb301": NB301Benchmark,
# > "rbv2_super": RBV2SuperBenchmark,
# > "rbv2_aknn": RBV2aknnBenchmark,
# > "rbv2_glmnet": RBV2glmnetBenchmark,
# > "rbv2_ranger": RBV2rangerBenchmark,
# > "rbv2_rpart": RBV2rpartBenchmark,
# > "rbv2_svm": RBV2svmBenchmark,
# > "rbv2_xgboost": RBV2xgboostBenchmark,
# > "iaml_glmnet": IAMLglmnetBenchmark,
# > "iaml_ranger": IAMLrangerBenchmark,
# > "iaml_rpart": IAMLrpartBenchmark,
# > "iaml_super": IAMLSuperBenchmark,
# > "iaml_xgboost": IAMLxgboostBenchmark,

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mfpbench
import numpy as np
from mfpbench import JAHSBenchmark, LCBenchBenchmark, LCBenchTabularBenchmark

from hpo_glue.benchmark import BenchmarkDescription, SurrogateBenchmark, TabularBenchmark
from hpo_glue.fidelity import RangeFidelity
from hpo_glue.measure import Measure
from hpo_glue.result import Result

if TYPE_CHECKING:
    from hpo_glue.query import Query


def _get_surrogate_benchmark(
    description: BenchmarkDescription,
    *,
    benchmark_name: str,
    datadir: Path | str | None = None,
    **kwargs: Any,
) -> SurrogateBenchmark:
    if datadir is not None:
        datadir = Path(datadir).absolute().resolve()
        kwargs["datadir"] = datadir
    bench = mfpbench.get(benchmark_name, **kwargs)
    query_function = partial(_mfpbench_surrogate_query_function, benchmark=bench)
    return SurrogateBenchmark(
        desc=description,
        benchmark=bench,
        config_space=bench.space,
        query=query_function,
    )


def _mfpbench_surrogate_query_function(query: Query, benchmark: mfpbench.Benchmark) -> Result:
    assert isinstance(query.fidelity, tuple)
    _, fid_value = query.fidelity
    return Result(
        query=query,
        values=benchmark.query(
            query.config.values,
            at=fid_value,
        ).as_dict(),
        fidelity=query.fidelity,
    )


def _lcbench_tabular(
    description: BenchmarkDescription,
    *,
    task_id: str,
    datadir: Path | str | None = None,
    remove_constants: bool = True,
) -> TabularBenchmark:
    if isinstance(datadir, str):
        datadir = Path(datadir).absolute().resolve()

    if datadir is None:
        datadir = Path("data", "lcbench-tabular").absolute().resolve()

    bench = mfpbench.LCBenchTabularBenchmark(
        task_id=task_id,
        datadir=datadir,
        remove_constants=remove_constants,
    )
    return TabularBenchmark(
        desc=description,
        table=bench.table,
        id_key="id",  # Key in the table to uniquely identify configs
        config_keys=bench.config_keys,  # Keys in the table that correspond to configs
    )


descriptions = []


def lcbench_surrogate(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    for task_id in LCBenchBenchmark.yahpo_instances:  # type: ignore
        yield BenchmarkDescription(
            name=f"yahpo-lcbench-{task_id}",
            load=partial(_lcbench_tabular, task_id=task_id, datadir=datadir),
            metrics={
                "val_accuracy": Measure.metric((0.0, 100.0), minimize=False),
                "val_cross_entropy": Measure.metric((0, np.inf), minimize=True),
                "val_balanced_accuracy": Measure.metric((0, 100), minimize=False),
            },
            test_metrics={
                "test_balanced_accuracy": Measure.test_metric((0, 100), minimize=False),
                "test_cross_entropy": Measure.test_metric(bounds=(0, np.inf), minimize=True),
            },
            costs={
                "time": Measure.cost((0, np.inf), minimize=True),
            },
            fidelities={
                "epoch": RangeFidelity.from_tuple((1, 52, 1), supports_continuation=True),
            },
        )


def lcbench_tabular(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    for task_id in LCBenchTabularBenchmark.task_ids:
        yield BenchmarkDescription(
            name=f"lcbench_tabular-{task_id}",
            load=partial(_lcbench_tabular, task_id=task_id, datadir=datadir),
            is_tabular=True,
            fidelities={
                "epoch": RangeFidelity.from_tuple((1, 51, 1), supports_continuation=True),
            },
            costs={
                "time": Measure.cost((0, np.inf), minimize=True),
            },
            metrics={
                "val_accuracy": Measure.metric((0.0, 100.0), minimize=False),
                "val_cross_entropy": Measure.metric((0, np.inf), minimize=True),
                "val_balanced_accuracy": Measure.metric((0, 100), minimize=False),
            },
            test_metrics={
                "test_accuracy": Measure.metric((0.0, 100.0), minimize=False),
                "test_balanced_accuracy": Measure.test_metric((0, 100), minimize=False),
                "test_cross_entropy": Measure.test_metric(bounds=(0, np.inf), minimize=True),
            },
        )


def mfh(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    for correlation in ("bad", "good", "moderate", "terrible"):
        for dims in (3, 6):
            name = f"mfh{dims}_{correlation}"
            _min = -3.32237 if dims == 3 else -3.86278  # noqa: PLR2004
            yield BenchmarkDescription(
                name=name,
                load=partial(_get_surrogate_benchmark, benchmark_name=name, datadir=datadir),
                costs={
                    "fid_cost": Measure.cost((0.05, 1), minimize=True),
                },
                fidelities={
                    "z": RangeFidelity.from_tuple((1, 100, 1), supports_continuation=True),
                },
                metrics={
                    "value": Measure.metric((_min, np.inf), minimize=True),
                },
            )


def jahs(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    for task_id in JAHSBenchmark.task_ids:
        name = f"jahs-{task_id}"
        yield BenchmarkDescription(
            name=name,
            load=partial(
                _get_surrogate_benchmark,
                benchmark_name="jahs",
                task_id=task_id,
                datadir=datadir,
            ),
            metrics={
                "valid_acc": Measure.metric((0.0, 100.0), minimize=False),
            },
            test_metrics={
                "test_acc": Measure.test_metric((0.0, 100.0), minimize=False),
            },
            fidelities={
                "epoch": RangeFidelity.from_tuple((1, 200, 1), supports_continuation=True),
            },
            costs={
                "runtime": Measure.cost((0, np.inf), minimize=True),
            },
            has_conditionals=False,
            is_tabular=False,
        )


def pd1(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    BS = [
        mfpbench.PD1cifar100_wideresnet_2048,
        mfpbench.PD1imagenet_resnet_512,
        mfpbench.PD1lm1b_transformer_2048,
        mfpbench.PD1translatewmt_xformer_64,
    ]
    for B in BS:
        if "test_error_rate" in B.result_type.metric_defs:
            test_metrics = {"test_error_rate": Measure.test_metric((0, 1), minimize=True)}
        else:
            test_metrics = None

        name = f"pd1-{B.pd1_name}"
        benchmark_name = B.pd1_name.replace("-", "_")
        yield BenchmarkDescription(
            name=name,
            load=partial(_get_surrogate_benchmark, benchmark_name=benchmark_name, datadir=datadir),
            metrics={
                "valid_error_rate": Measure.metric((0, 1), minimize=True),
            },
            test_metrics=test_metrics,
            costs={
                "train_cost": Measure.cost((0, np.inf), minimize=True),
            },
            fidelities={
                "epoch": RangeFidelity.from_tuple(
                    B.pd1_fidelity_range,
                    supports_continuation=True,
                )
            },
        )


def mfpbench_benchmarks(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    yield from lcbench_surrogate(datadir)
    yield from lcbench_tabular(datadir)
    yield from mfh(datadir)
    yield from jahs(datadir)
