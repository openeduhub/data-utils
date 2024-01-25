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


def _with_new_value(
    nested_dict: Nested_Dict | Terminal_Value | list[Any],
    keys: Sequence[str],
    value: Terminal_Value,
    replace_lists=False,
) -> Nested_Dict | Terminal_Value:
    # base case
    if len(keys) == 0:
        return value

    key = keys[0]
    keys = keys[1:]

    # if we have reached a list and we are not replacing lists,
    # create a new sub-dictionary with the current key and append it
    if not replace_lists and isinstance(nested_dict, list):
        return nested_dict + [
            {key: _with_new_value(dict(), keys, value, replace_lists)}
        ]
    # if we have reached a dictionary, update its value for the current key,
    # keeping the previous value, if it existed
    if isinstance(nested_dict, dict):
        return nested_dict | {
            key: _with_new_value(
                nested_dict.get(key, dict()), keys, value, replace_lists
            )
        }

    # if we have reached a terminal value, but still have keys
    # to assign, replace the value with a new dictionary
    return {key: _with_new_value(dict(), keys, value, replace_lists)}


def with_new_value(
    nested_dict: Nested_Dict,
    keys: Sequence[str],
    value: Terminal_Value,
    replace_lists=False,
) -> Nested_Dict:
    if len(keys) == 0:
        return nested_dict

    key = keys[0]
    keys = keys[1:]

    return nested_dict | {
        key: _with_new_value(nested_dict.get(key, dict()), keys, value, replace_lists)
    }
