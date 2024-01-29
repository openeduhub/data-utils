import operator as op
from collections.abc import Callable, Collection, Iterable, Sequence
from functools import reduce
from typing import Any, Optional, TypeVar

import numpy as np

from data_utils.utils import (
    Basic_Value,
    Basic_Value_Not_None,
    Nested_Dict,
    Nested_Dict_Subtree,
    Terminal_Value,
)

T = TypeVar("T")


def with_changed_value(
    entry: Nested_Dict,
    keys: Sequence[str],
    to_drop: Collection[Basic_Value_Not_None],
    to_remap: dict[Basic_Value_Not_None, Basic_Value],
) -> Nested_Dict:
    if None in to_drop or None in to_remap:
        raise ValueError(
            "Dropping or remapping from None is not supported, as this can, and will, result in unexpected behavior."
        )

    if len(keys) == 0:
        return entry

    key = keys[0]
    keys = keys[1:]

    # nothing to fix
    if key not in entry:
        return entry

    return entry | {key: _with_changed_value(entry[key], keys, to_drop, to_remap)}


def _with_changed_value(
    subtree: Nested_Dict_Subtree,
    keys: Sequence[str],
    to_drop: Collection[Basic_Value_Not_None],
    to_remap: dict[Basic_Value_Not_None, Basic_Value],
) -> Nested_Dict_Subtree:
    # we have reached the subtree we need to modify
    if len(keys) == 0:
        if isinstance(subtree, Basic_Value):
            return change_single_value(subtree, to_drop, to_remap)

        if isinstance(subtree, list):
            return change_multi_value(subtree, to_drop, to_remap)

        return subtree  # do not touch nested dictionaries

    key = keys[0]
    keys = keys[1:]

    if isinstance(subtree, Basic_Value):
        return subtree

    if isinstance(subtree, list):
        return [
            subtree_value
            | {key: _with_changed_value(subtree_value[key], keys, to_drop, to_remap)}
            if isinstance(subtree_value, dict) and key in subtree_value
            else subtree_value  # nothing to fix here
            for subtree_value in subtree
        ]

    # nothing to fix here
    if key not in subtree:
        return subtree

    return subtree | {key: _with_changed_value(subtree[key], keys, to_drop, to_remap)}


def change_single_value(
    x: Basic_Value,
    to_drop: Collection[Basic_Value_Not_None],
    to_remap: dict[Basic_Value_Not_None, Basic_Value],
) -> Basic_Value:
    x = to_remap.get(x, x)  # type: ignore
    if x in to_drop:
        return None

    return x


def change_multi_value(
    subtree: Iterable[Basic_Value | Nested_Dict],
    to_drop: Collection[Basic_Value_Not_None],
    to_remap: dict[Basic_Value_Not_None, Basic_Value],
) -> list[Basic_Value | Nested_Dict]:
    to_ret: list[Basic_Value | Nested_Dict] = list()
    for subtree_value in subtree:
        # do not touch nested dictionaries
        if isinstance(subtree_value, dict):
            to_ret.append(subtree_value)
            continue

        remapped_value = to_remap.get(subtree_value, subtree_value)  # type: ignore
        if remapped_value in to_drop:
            continue

        to_ret.append(remapped_value)

    return to_ret


def as_boolean_array(
    values: Collection[Collection[T]],
    sort_fn: Optional[Callable[[list[T]], list[T]]] = None,
) -> tuple[np.ndarray[Any, np.dtypes.BoolDType], list[T]]:
    if len(values) == 0:
        return np.array([]), []

    values_as_sets = [set(value) if value is not None else set() for value in values]

    unique_values_list: list[T] = list(reduce(op.or_, values_as_sets, set()))
    if sort_fn is not None:
        unique_values_list = sort_fn(unique_values_list)

    value_indices = {value: index for index, value in enumerate(unique_values_list)}

    def set_to_bool_array(x: set[T]) -> np.ndarray:
        found_indices = [value_indices[value] for value in x]
        res = np.zeros(len(unique_values_list), dtype=bool)
        for found_index in found_indices:
            res[found_index] = True
        return res

    return (
        np.stack([set_to_bool_array(value) for value in values_as_sets]),
        unique_values_list,
    )
