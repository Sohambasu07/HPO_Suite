from __future__ import annotations
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from hpo_glue.glu import Optimizer

from optimizers.optuna import Optuna_Optimizer
from optimizers.smac import SMAC_Optimizer, SMAC_BO, SMAC_Hyperband

OPTIMIZERS: Dict[str, Optimizer] = {
    "SMAC": SMAC_Optimizer,
    "SMAC_BO": SMAC_BO,
    "SMAC_Hyperband": SMAC_Hyperband,
    "Optuna": Optuna_Optimizer,
}

def get_all_optimizers() -> Dict[str, Optimizer]:
    """Get all optimizers."""

    return OPTIMIZERS