from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

Basic_Value = str | int | float | None
Terminal_Value = Basic_Value | list["Terminal_Value"]


Nested_Dict = dict[str, Any]
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
        return [_to_terminal(x) for x in result]

    return result


def get_terminal_in(
    nested_dict: Nested_Dict,
    keys: Sequence[str],
    catch_errors: tuple[type[Exception], ...] = (KeyError, TypeError),
) -> Terminal_Value:
    """Like get_in, but replace non-terminal results with None."""
    return _to_terminal(get_in(nested_dict, keys, catch_errors))
