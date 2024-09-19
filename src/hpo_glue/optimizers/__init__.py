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

MF_OPTIMIZERS: dict[str, type[Optimizer]] = {}
NON_MF_OPTIMIZERS: dict[str, type[Optimizer]] = {}
MO_OPTIMIZERS: dict[str, type[Optimizer]] = {}
SO_OPTIMIZERS: dict[str, type[Optimizer]] = {}

for name, opt in OPTIMIZERS.items():
    if "single" in opt.support.fidelities:
        MF_OPTIMIZERS[name] = opt
    else:
        NON_MF_OPTIMIZERS[name] = opt
    if "many" in opt.support.objectives:
        MO_OPTIMIZERS[name] = opt
    if "single" in opt.support.objectives:
        SO_OPTIMIZERS[name] = opt
__all__ = [
    "OPTIMIZERS",
    "MF_OPTIMIZERS",
    "NON_MF_OPTIMIZERS",
    "MO_OPTIMIZERS",
    "SO_OPTIMIZERS",
]
