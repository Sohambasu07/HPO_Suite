from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Literal, Mapping

if TYPE_CHECKING:
    from hpo_glue.benchmark import BenchmarkDescription
    from hpo_glue.optimizer import Optimizer
    from hpo_glue.fidelity import Fidelity


def _generate_problem_set(
    *,
    optimizer: type[Optimizer],
    benchmark: BenchmarkDescription,
    seed: int | list[int],
    fidelities: int = 0,
    objectives: int = 1,
) -> Iterator[Problem]:
    """Generate a set of problems for the given optimizer and benchmark.

    Args:
        optimizer: The optimizer to use
        benchmark: The benchmark to use
        seed: The seed to use for the problems
        multiobjective: The number of objectives to use
        multifidelity: Whether to use multifidelity
        manyfidelity: Whether to provide many fidelities
    """
    fidelity: tuple[str, Fidelity] | Mapping[str, Fidelity] | None
    match fidelities:
        case int() if fidelities < 0:
            raise ValueError(
                f"Number of fidelities must be greater than 0, but got `{fidelities=}`"
            )
        case int() if fidelities == 1:
            if not optimizer.supports_multifidelity:
                raise ValueError(
                    f"Optimizer {optimizer.name} does not support multi-fidelity but"
                    " `many_fidelity=True` was passed."
                )

            match benchmark.fidelities:
                case None:
                    raise ValueError(
                        f"Multi-fidelity is enabled but no fidelities are defined for the"
                        f" benchmark {benchmark.name}"
                    )
                case Mapping():
                    fidelity = next(iter(benchmark.fidelities.items()))
                case _:
                    raise TypeError(
                        f"Expected a Mapping of fidelities, but got {benchmark.fidelities=}"
                    )

        case int() if fidelities > 1:
            if not optimizer.supports_manyfidelity:
                raise ValueError(
                    f"Optimizer {optimizer.name} does not support many-fidelity (>1) but"
                    f" {fidelities=} was passed."
                )

    match multiobjective:
        case int() if multiobjective < 2:
            raise ValueError(
                f"Number of objectives must be greater than 1, but got `{multiobjective=}`"
            )
        case Sequence() if any(obj < 2 for obj in multiobjective):
            raise ValueError(
                f"Number of objectives must be greater than 1, but got `{multiobjective=}`"
            )
        case int() if not optimizer.supports_multiobjective:
            raise ValueError(
                f"Optimizer {optimizer.name} does not support multi-objective but"
                " `multiobjective=True` was passed."
            )




    if multiobjective is not False:
    elif multiobjective < 2:
        raise ValueError(
            f"Number of objectives must be greater than 1, but got `{multiobjective=}`"
        )
