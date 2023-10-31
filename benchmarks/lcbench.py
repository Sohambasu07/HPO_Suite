import os
import zipfile
import json
import logging
import pandas as pd
from pathlib import Path
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    Constant, 
    UniformFloatHyperparameter, 
    UniformIntegerHyperparameter)

from typing import List

from hpo_glue.glu import TabularBenchmark

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LCBenchSetup:
    def __init__(self):
        self.name = "LCBench"
        self.url = "https://figshare.com/ndownloader/files/21001299"
        self.path = Path(self.name)
        self._download()
        self.filepath = self.path / "six_datasets_lw.json"
        self.tablepath = None

    def _download(self):
        if not os.path.exists(self.path):
            os.system(f"wget {self.url} -O {self.path}")
            with zipfile.ZipFile(self.path / '.zip', 'r') as zip_ref:
                zip_ref.extractall(".")
        self._setup_benchmark()

    def _setup_benchmark(self):
        filepath: Path = self.path / "six_datasets_lw.json"
        with open(filepath, "r") as f:
            all_data = json.load(f)

        for dataset_name, data in all_data.items():
            logger.info(f"Processing {dataset_name}")
            config_frames_for_dataset = []
            for config_id, config_data in data.items():
                config: dict = config_data["config"]

                log_data: dict = config_data["log"]
                loss: list[float] = log_data["Train/loss"]
                val_ce: list[float] = log_data["Train/val_cross_entropy"]
                val_acc: list[float] = log_data["Train/val_accuracy"]
                val_bal_acc: list[float] = log_data["Train/val_balanced_accuracy"]

                test_ce: list[float] = log_data["Train/test_cross_entropy"]
                test_bal_acc: list[float] = log_data["Train/test_balanced_accuracy"]

                # NOTE: Due to there being a lack of "Test/val_accuracy" in the
                # data but a "Train/test_result" we use the latter as the test accuracy
                test_acc: list[float] = log_data["Train/test_result"]

                time = log_data["time"]
                epoch = log_data["epoch"]

                df = pd.DataFrame(
                    {
                        "time": time,
                        "epoch": epoch,
                        "loss": loss,
                        "val_accuracy": val_acc,
                        "val_cross_entropy": val_ce,
                        "val_balanced_accuracy": val_bal_acc,
                        "test_accuracy": test_acc,
                        "test_cross_entropy": test_ce,
                        "test_balanced_accuracy": test_bal_acc,
                    },
                )
                # These are single valued but this will make them as a list into
                # the dataframe
                df = df.assign(**{"id": config_id, **config})

                config_frames_for_dataset.append(df)

            #                     | **metrics, **config_params
            # (config_id, epoch)  |
            df_for_dataset = (
                pd.concat(config_frames_for_dataset, ignore_index=True)
                .convert_dtypes()
                .set_index(["id", "epoch"])
                .sort_index()
            )
            table_path = self.path / f"{dataset_name}.parquet"
            df_for_dataset.to_parquet(table_path)
            logger.info(f"Processed {dataset_name} to {table_path}")

class LCBenchTabular:
    def __init__(self, task_name):
        self.task_name = task_name

    def get_config_space(seed: int) -> ConfigurationSpace:
        #code taken from https://github.com/automl/mf-prior-bench/blob/main/src/mfpbench/lcbench_tabular/benchmark.py
        
        cs = ConfigurationSpace(seed=seed)

        cs.add_hyperparameters(
            [
                UniformIntegerHyperparameter(
                "batch_size",
                lower=16,
                upper=512,
                log=True,
                default_value=128,  # approximately log-spaced middle of range
            ),
            UniformFloatHyperparameter(
                "learning_rate",
                lower=1.0e-4,
                upper=1.0e-1,
                log=True,
                default_value=1.0e-3,  # popular choice of LR
            ),
            UniformFloatHyperparameter(
                "momentum",
                lower=0.1,
                upper=0.99,
                log=False,
                default_value=0.9,  # popular choice, also not on the boundary
            ),
            UniformFloatHyperparameter(
                "weight_decay",
                lower=1.0e-5,
                upper=1.0e-1,
                log=False,
                default_value=1.0e-2,  # reasonable default
            ),
            UniformIntegerHyperparameter(
                "num_layers",
                lower=1,
                upper=5,
                log=False,
                default_value=3,  # middle of range
            ),
            UniformIntegerHyperparameter(
                "max_units",
                lower=64,
                upper=1024,
                log=True,
                default_value=256,  # approximately log-spaced middle of range
            ),
            UniformFloatHyperparameter(
                "max_dropout",
                lower=0,
                upper=1,
                log=False,
                default_value=0.2,  # reasonable default
            ),
            Constant("cosine_annealing_T_max", 50),
            Constant("cosine_annealing_eta_min", 0.0),
            Constant("normalization_strategy", "standardize"),
            Constant("optimizer", "sgd"),
            Constant("learning_rate_scheduler", "cosine_annealing"),
            Constant("network", "shapedmlpnet"),
            Constant("activation", "relu"),
            Constant("mlp_shape", "funnel"),
            Constant("imputation_strategy", "mean"),
            ]
        )

        return cs
    
# Tabular is relatively easy
def lcbench_tabular(task_id: str, datadir: Path) -> TabularBenchmark:

    LCBenchSetup()
    table_path = datadir / f"{task_id}.parquet"
    table_for_task = pd.load(table_path)
    return TabularBenchmark(
        name=f"lcbench-tabular-{task_id}",
        table=table_for_task,
        id_key =
        config_keys=...,  # Keys in the table that correspond to the config values
        result_key=...,  # Key in the table that corresponds to the result
        fidelity_key=...,  # Key in the table that corresponds to the fidelity (e.g. "epoch")
    )
