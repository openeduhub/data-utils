"""Module defining fundamental data types and how to interact with them."""
from __future__ import annotations

import operator as op
from collections.abc import Iterable, Sequence
from functools import reduce
from typing import Union

Basic_Value_Not_None = str | int | float
Basic_Value = Basic_Value_Not_None | None
Terminal_Value = Basic_Value | list[Basic_Value]

Data_Point = dict[
    str, Union[Basic_Value, "Data_Point", list[Union[Basic_Value, "Data_Point"]]]
]
Data_Point_Subtree = Basic_Value | Data_Point | list[Basic_Value | Data_Point]
Query_Result = Terminal_Value | Data_Point | list["Query_Result"]


def get_leaves(data_point: Data_Point) -> set[tuple[str, ...]]:
    return _get_leaves(data_point, tuple(), set())


def get_in(
    data_point: Data_Point,
    keys: Sequence[str],
    catch_errors: tuple[type[Exception], ...] = tuple(),
) -> Query_Result:
    return _get_in(data_point, keys, catch_errors)


def get_terminal_in(
    data_point: Data_Point,
    keys: Sequence[str],
    catch_errors: tuple[type[Exception], ...] = (KeyError, TypeError),
) -> Terminal_Value:
    return _to_terminal(get_in(data_point, keys, catch_errors))


def _get_leaves(
    data_point: Data_Point_Subtree,
    current_keys: tuple[str, ...],
    current_leaves: set[tuple[str, ...]],
) -> set[tuple[str, ...]]:
    if isinstance(data_point, Basic_Value):
        return current_leaves | {current_keys}

    if isinstance(data_point, dict):
        return reduce(
            op.or_,
            (
                _get_leaves(subtree, current_keys + (key,), current_leaves)
                for key, subtree in data_point.items()
            ),
            current_leaves,
        )

    return reduce(
        op.or_,
        (_get_leaves(subtree, current_keys, current_leaves) for subtree in data_point),
        current_leaves,
    )


def _get_in(
    data_point: Query_Result,
    keys: Sequence[str],
    catch_errors: Iterable[type[Exception]] = tuple(),
) -> Query_Result:
    if not keys:
        return data_point

    key = keys[0]
    keys = keys[1:]

    try:
        # type errors are expected here
        val = data_point[key]  # type: ignore
    except catch_errors as e:
        return None

    if isinstance(val, list):
        return [_get_in(sub_val, keys, catch_errors) for sub_val in val]

    return _get_in(val, keys, catch_errors)


def _to_terminal(result: Query_Result) -> Terminal_Value:
    if isinstance(result, dict):
        return None

    if isinstance(result, list):
        to_ret: list[Basic_Value] = list()
        for x in result:
            x_terminal = _to_terminal(x)
            # ensure that we are returning a flat list
            if isinstance(x_terminal, list):
                to_ret.extend(x_terminal)
            else:
                to_ret.append(x_terminal)

        return to_ret

    return result
