from __future__ import annotations

from typing import TYPE_CHECKING

from hpo_glue.optimizers.dehb import DEHB_Optimizer
from hpo_glue.optimizers.myopt import My_Opt
from hpo_glue.optimizers.optuna import OptunaOptimizer
from hpo_glue.optimizers.smac import SMAC_BO, SMAC_Hyperband
from hpo_glue.optimizers.synetune import SyneTuneBO, SyneTuneBOHB

if TYPE_CHECKING:
    from hpo_glue.optimizer import Optimizer

OPTIMIZERS: dict[str, type[Optimizer]] = {
    DEHB_Optimizer.name: DEHB_Optimizer,
    SMAC_BO.name: SMAC_BO,
    SMAC_Hyperband.name: SMAC_Hyperband,
    SyneTuneBO.name: SyneTuneBO,
    SyneTuneBOHB.name: SyneTuneBOHB,
    OptunaOptimizer.name: OptunaOptimizer,
    My_Opt.name: My_Opt,
}

__all__ = [
    "OPTIMIZERS",
]
