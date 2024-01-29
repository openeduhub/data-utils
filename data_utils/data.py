"""Module defining fundamental data types and how to interact with them."""
from __future__ import annotations

import operator as op
from collections.abc import Iterable, Sequence
from functools import reduce
from typing import Union

Basic_Value_Not_None = str | int | float
Basic_Value = Basic_Value_Not_None | None
Terminal_Value = Basic_Value | list[Basic_Value]


Nested_Dict = dict[
    str, Union[Basic_Value, "Nested_Dict", list[Union[Basic_Value, "Nested_Dict"]]]
]
Nested_Dict_Subtree = Basic_Value | Nested_Dict | list[Basic_Value | Nested_Dict]
Query_Result = Terminal_Value | Nested_Dict | list["Query_Result"]


def get_leaves(nested_dict: Nested_Dict) -> set[tuple[str, ...]]:
    return _get_leaves(nested_dict, tuple(), set())


def _get_leaves(
    nested_dict: Nested_Dict_Subtree,
    current_keys: tuple[str, ...],
    current_leaves: set[tuple[str, ...]],
) -> set[tuple[str, ...]]:
    if isinstance(nested_dict, Basic_Value):
        return current_leaves | {current_keys}

    if isinstance(nested_dict, dict):
        return reduce(
            op.or_,
            (
                _get_leaves(subtree, current_keys + (key,), current_leaves)
                for key, subtree in nested_dict.items()
            ),
            current_leaves,
        )

    return reduce(
        op.or_,
        (_get_leaves(subtree, current_keys, current_leaves) for subtree in nested_dict),
        current_leaves,
    )


def get_in(
    nested_dict: Nested_Dict,
    keys: Sequence[str],
    catch_errors: tuple[type[Exception], ...] = tuple(),
) -> Query_Result:
    """
    Recursively access a nested dictionary.

    :param catch_errors: Errors to catch when accessing the nested dictionary.
      Catch KeyErrors to return None when a nested dictionary
        does not contain the key sequence.
      Catch TypeErrors to return None when a nested dictionary ends
        before all keys have been used up.
    """
    return _get_in(nested_dict, keys, catch_errors)


def _get_in(
    nested_dict: Query_Result,
    keys: Sequence[str],
    catch_errors: Iterable[type[Exception]] = tuple(),
) -> Query_Result:
    if not keys:
        return nested_dict

    key = keys[0]
    keys = keys[1:]

    try:
        # type errors are expected here
        val = nested_dict[key]  # type: ignore
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


def get_terminal_in(
    nested_dict: Nested_Dict,
    keys: Sequence[str],
    catch_errors: tuple[type[Exception], ...] = (KeyError, TypeError),
) -> Terminal_Value:
    """
    Like get_in, but replace non-terminal results with None and flatten nested lists.
    """
    return _to_terminal(get_in(nested_dict, keys, catch_errors))
