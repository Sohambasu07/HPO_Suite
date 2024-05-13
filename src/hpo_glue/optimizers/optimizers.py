from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hpo_glue.optimizers.random_search import RandomSearch

if TYPE_CHECKING:
    from hpo_glue.optimizer import Optimizer

OPTIMIZERS: dict[str, type[Optimizer]] = {
    "RandomSearch": RandomSearch,
}

try:
    from hpo_glue.optimizers.smac import SMAC_BO, SMAC_Hyperband

    OPTIMIZERS.update(
        {
            SMAC_BO.__name__: SMAC_BO,
            SMAC_Hyperband.__name__: SMAC_Hyperband,
        }
    )
except ImportError:
    logging.info("Could not import smac, skipping SMAC optimizers")

try:
    from hpo_glue.optimizers.optuna import OptunaOptimizer

    OPTIMIZERS.update(
        {
            OptunaOptimizer.__name__: OptunaOptimizer,
        }
    )
except ImportError:
    logging.info("Could not import optuna, skipping Optuna optimizer")

try:
    from hpo_glue.optimizers.dehb import DEHB_Optimizer

    OPTIMIZERS.update(
        {
            DEHB_Optimizer.__name__: DEHB_Optimizer,
        }
    )
except ImportError:
    logging.info("Could not import dehb, skipping Optuna optimizer")

try:
    from hpo_glue.optimizers.synetune import SyneTuneBO

    OPTIMIZERS.update(
        {
            SyneTuneBO.__name__: SyneTuneBO,
        }
    )
except ImportError:
    logging.info("Could not import dehb, skipping Optuna optimizer")


def get_all_optimizers() -> dict[str, type[Optimizer]]:
    """Get all optimizers."""
    return OPTIMIZERS
