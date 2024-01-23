from collections.abc import Collection, Iterable
from typing import NamedTuple, TypeVar

T = TypeVar("T")


def fix_entry_single_value(
    x: T, to_drop: Collection[T], to_merge: dict[T, T]
) -> T | None:
    if x in to_drop:
        return None

    return to_merge.get(x, x)


def fix_entry_multi_value(
    x: Iterable[T], to_drop: Collection[T], to_merge: dict[T, T]
) -> set[T]:
    return {
        fixed_val
        for val in x
        if (fixed_val := fix_entry_single_value(val, to_drop, to_merge)) is not None
    }
