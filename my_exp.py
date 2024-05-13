from __future__ import annotations

from hpo_glue import all_benchmarks

for _k, _v in all_benchmarks().items():
    print(_k, _v)  # noqa: T201
    _v()
