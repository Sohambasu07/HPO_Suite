from __future__ import annotations
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from hpo_glue.glu import Optimizer

from optimizers.random_search import RandomSearch
from optimizers.smac import SMAC_Optimizer, SMAC_BO, SMAC_Hyperband
from optimizers.optuna import OptunaOptimizer

OPTIMIZERS: Dict[str, Optimizer] = {
    "RandomSearch": RandomSearch,
    "SMAC_Optimizer": SMAC_Optimizer,
    "SMAC_BO": SMAC_BO,
    "SMAC_Hyperband": SMAC_Hyperband,
    "OptunaOptimizer": OptunaOptimizer,
}

def get_all_optimizers() -> Dict[str, Optimizer]:
    """Get all optimizers."""

    return OPTIMIZERS