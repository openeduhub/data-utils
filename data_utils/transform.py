import operator as op
from collections.abc import Callable, Collection, Sequence
from functools import reduce
from typing import Any, Optional, TypeVar

import numpy as np

T = TypeVar("T")


def as_boolean_array(
    values: Collection[None | T | Collection[T]],
    sort_fn: Optional[Callable[[list[T]], list[T]]] = None,
) -> tuple[np.ndarray[Any, np.dtypes.BoolDType], list[T]]:
    def to_set(x: None | T | Collection[T]) -> set[T]:
        if x is None:
            return set()
        if isinstance(x, Collection):
            return set(x)
        return {x}

    values_as_sets = [to_set(value) for value in values]

    unique_values_list = list(reduce(op.or_, values_as_sets))
    if sort_fn is not None:
        unique_values_list = sort_fn(unique_values_list)

    def set_to_bool_array(x: set[T]) -> np.ndarray:
        found_indices = [unique_values_list.index(value) for value in x]
        res = np.zeros(len(unique_values_list), dtype=bool)
        for found_index in found_indices:
            res[found_index] = True
        return res

    return (
        np.stack([set_to_bool_array(value) for value in values_as_sets]),
        unique_values_list,
    )
