from __future__ import annotations

from hpo_glue import BENCHMARKS

for name, desc in BENCHMARKS.items():
    print(name, desc)  # noqa: T201
