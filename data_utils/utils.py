from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

Basic_Value = str | int | float | None
Terminal_Value = Basic_Value | list[Basic_Value]


Nested_Dict = dict[str, Any]
Query_Result = Terminal_Value | Nested_Dict | list[Nested_Dict]


def get_in(
    nested_dict: Nested_Dict,
    keys: Sequence[str],
    catch_errors: Iterable[type[Exception]] = tuple(),
) -> Query_Result:
    """
    Recursively access a nested dictionary.

    :param catch_errors: Errors to catch when accessing the nested dictionary.
      Catch KeyErrors to return None when a nested dictionary
        does not contain the key sequence.
      Catch TypeErrors to return None when a nested dictionary ends
        before all keys have been used up.
    """
    nested_dict_iter: Nested_Dict | Terminal_Value = nested_dict
    while True:
        if not keys:
            return nested_dict_iter

        key = keys[0]
        keys = keys[1:]

        try:
            # type errors are OK here. errors will be caught below
            val = nested_dict_iter[key]  # type: ignore

            # check if the value is a list of dictionaries.
            # if so, we need to iterate over this list
            if isinstance(val, list):
                if len(val) == 0:
                    if keys:
                        raise TypeError()

                    return []

                sub_val = val[0]
                if isinstance(sub_val, dict):
                    # this returns either a list of terminal values
                    # or a list of nested dicts
                    return [get_in(sub_val, keys_iter) for sub_val in val]  # type: ignore

            nested_dict_iter = val  # type: ignore

        except catch_errors as e:
            return None


def get_terminal_in(
    nested_dict: Nested_Dict,
    keys: Sequence[str],
    catch_errors: Iterable[type[Exception]] = (KeyError, TypeError),
) -> Terminal_Value:
    val = get_in(nested_dict, keys, catch_errors)
    if isinstance(val, dict):
        return None
    if isinstance(val, list):
        if len(val) > 0 and isinstance(val[0], dict):
            return None

    return val  # type: ignore
