from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Union

Basic_Value = str | int | float | None
Terminal_Value = Basic_Value | list[Basic_Value]


Nested_Dict = dict[
    str, Union[Basic_Value, "Nested_Dict", list[Union["Nested_Dict", Basic_Value]]]
]
Query_Result = Terminal_Value | Nested_Dict | list["Query_Result"]


def _nested_get_in(
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
        return [_nested_get_in(sub_val, keys, catch_errors) for sub_val in val]

    return _nested_get_in(val, keys, catch_errors)


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
    return _nested_get_in(nested_dict, keys, catch_errors)


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
