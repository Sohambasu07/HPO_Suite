from __future__ import annotations

from hpo_glue.benchmarks import BENCHMARKS

# from hpo_glue.optimizers import OPTIMIZERS

for name, desc in BENCHMARKS.items():
    print(name, desc)  # noqa: T201

from hpo_glue.optimizers.smac import SMAC_BO, SMAC_Hyperband  # noqa: F401
# for name, opt in OPTIMIZERS.items():
#    print(name, opt)
