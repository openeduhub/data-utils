import operator as op
from collections.abc import Callable, Collection, Iterable, Sequence
from functools import reduce
from typing import Any, Optional, TypeVar

import numpy as np

from data_utils.data import (
    Basic_Value,
    Basic_Value_Not_None,
    Data_Point,
    Data_Point_Subtree,
)

T = TypeVar("T")


def as_boolean_array(
    values: Collection[Collection[T]],
    sort_fn: Optional[Callable[[list[T]], list[T]]] = None,
) -> tuple[np.ndarray[Any, np.dtypes.BoolDType], list[T]]:
    """
    Turn any two-dimensional collection of values into a Boolean array.

    :param values: Each value in ``values`` represents one data point with
        potentially multiple values (hence two-dimensionality).
    :param sort_fn: If given, sort the indices that indicate which value is
        being assigned according to the result of this function. Otherwise,
        these will be sorted in order of first occurrence in ``values``.
    :returns: A tuple containing

        1. The resulting 2d (numpy) Boolean array
        2. A list translating each column to the represented value.
    """
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


def with_changed_value(
    entry: Data_Point,
    field_seq: Sequence[str],
    to_drop: Collection[Basic_Value_Not_None],
    to_remap: dict[Basic_Value_Not_None, Basic_Value],
) -> Data_Point:
    """
    Create a new :any:`Data_Point` object with a modified field.

    :param entry: The data-point to copy and modify.
    :param field_seq: The sequence of keys to access in order to get to the
        field to modify.
    :param to_drop: The values within the specified field to drop.

        - If the field contains a singular value to be dropped, it will be
          replaced by ``None``.
        - If the field contains a list of values, all values to be dropped will
          be removed from the list.

        Note that dropping ``None`` is not supported, as this may result in
        unexpected behavior, since a list-value may not actually be represented
        by a leaf that contains a list of values. Because of this,
        ``None``-values may also be introduced to lists of values.
    :param to_remap: The values within the specified field to remap.
        The same caveats as with ``to_drop`` apply here.
    """
    if None in to_drop or None in to_remap:
        raise ValueError(
            "Dropping or remapping from None is not supported, as this can, and will, result in unexpected behavior."
        )

    if len(field_seq) == 0:
        return entry

    key = field_seq[0]
    field_seq = field_seq[1:]

    # nothing to fix
    if key not in entry:
        return entry

    return entry | {
        key: _with_changed_value(
            subtree=entry[key],
            field_seq=field_seq,
            to_drop=to_drop,
            to_remap=to_remap,
        )
    }


def _with_changed_value(
    subtree: Data_Point_Subtree,
    field_seq: Sequence[str],
    to_drop: Collection[Basic_Value_Not_None],
    to_remap: dict[Basic_Value_Not_None, Basic_Value],
) -> Data_Point_Subtree:
    # we have reached the subtree we need to modify
    if len(field_seq) == 0:
        if isinstance(subtree, Basic_Value):
            return _change_single_value(subtree, to_drop, to_remap)

        if isinstance(subtree, list):
            return _change_multi_value(subtree, to_drop, to_remap)

        return subtree  # do not touch nested dictionaries

    key = field_seq[0]
    field_seq = field_seq[1:]

    if isinstance(subtree, Basic_Value):
        return subtree

    if isinstance(subtree, list):
        return [
            subtree_value
            | {
                key: _with_changed_value(
                    subtree_value[key], field_seq, to_drop, to_remap
                )
            }
            if isinstance(subtree_value, dict) and key in subtree_value
            else subtree_value  # nothing to fix here
            for subtree_value in subtree
        ]

    # nothing to fix here
    if key not in subtree:
        return subtree

    return subtree | {
        key: _with_changed_value(subtree[key], field_seq, to_drop, to_remap)
    }


def _change_single_value(
    x: Basic_Value,
    to_drop: Collection[Basic_Value_Not_None],
    to_remap: dict[Basic_Value_Not_None, Basic_Value],
) -> Basic_Value:
    x = to_remap.get(x, x)  # type: ignore
    if x in to_drop:
        return None

    return x


def _change_multi_value(
    subtree: Iterable[Basic_Value | Data_Point],
    to_drop: Collection[Basic_Value_Not_None],
    to_remap: dict[Basic_Value_Not_None, Basic_Value],
) -> list[Basic_Value | Data_Point]:
    to_ret: list[Basic_Value | Data_Point] = list()
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
