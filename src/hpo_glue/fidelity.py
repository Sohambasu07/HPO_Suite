from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class Orderable(Protocol):
    def __lt__(self, other: Orderable) -> bool: ...
    def __gt__(self, other: Orderable) -> bool: ...
    def __le__(self, other: Orderable) -> bool: ...
    def __ge__(self, other: Orderable) -> bool: ...
    def __eq__(self, other: Orderable) -> bool: ...


T = TypeVar("T", bound=int | float)


@runtime_checkable
class Fidelity(Protocol[T]):
    kind: type[T]
    min: T
    max: T
    supports_continuation: bool

    def __iter__(self) -> Iterator[T]: ...
    def normalize(self, value: T) -> float: ...


@dataclass(kw_only=True)
class ListFidelity(Generic[T]):
    kind: type[T]
    values: list[T]
    supports_continuation: bool
    min: T = field(init=False)
    max: T = field(init=False)

    def __post_init__(self):
        self.min = min(self.values)
        self.max = max(self.values)

    def __iter__(self) -> Iterator[T]:
        return iter(self.values)

    def normalize(self, value: T) -> float:
        """Normalize a value to the range [0, 1]."""
        return (value - self.min) / (self.max - self.min)


@dataclass(kw_only=True)
class RangeFidelity(Generic[T]):
    kind: type[T]
    min: T
    max: T
    stepsize: T
    supports_continuation: bool

    def __post_init__(self):
        if self.min >= self.max:
            raise ValueError(f"min must be less than max, got {self.min} and {self.max}")

    def __iter__(self) -> Iterator[T]:
        current = self.min
        yield self.min
        while current < self.max:
            current += self.stepsize
            yield max(current, self.max)  # type: ignore

    @classmethod
    def from_tuple(
        cls,
        values: tuple[T, T, T],
        *,
        supports_continuation: bool = False,
    ) -> RangeFidelity[T]:
        """Create a RangeFidelity from a tuple of (min, max, stepsize)."""
        _type = type(values[0])
        if _type not in (int, float):
            raise ValueError(f"all values must be of type int or float, got {_type}")

        if not all(isinstance(v, _type) for v in values):
            raise ValueError(f"all values must be of type {_type}, got {values}")

        return cls(
            kind=_type,
            min=values[0],
            max=values[1],
            stepsize=values[2],
            supports_continuation=supports_continuation,
        )

    def normalize(self, value: T) -> float:
        """Normalize a value to the range [0, 1]."""
        return (value - self.min) / (self.max - self.min)
