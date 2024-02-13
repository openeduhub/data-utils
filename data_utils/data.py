"""Module defining fundamental data types and how to interact with them."""
from __future__ import annotations

import operator as op
from collections.abc import Iterable, Sequence
from functools import reduce
from typing import TypeVar, Union

#: A basic, singular value that may not be ``None``
Basic_Value_Not_None = str | int | float
#: Like :any:`Basic_Value_Not_None`, but may be ``None``
Basic_Value = Basic_Value_Not_None | None
#: A terminal value is any value that may be in a leaf of the tree
#: representing each :any:`Data_Point`.
Terminal_Value = Basic_Value | list[Basic_Value]

#: A data point is represented as a recursively nested dictionary,
#: where each key maps to a :any:`Terminal_Value`, another dictionary,
#: or a list of dictionaries. Thus, each data points follows a tree structure.
Data_Point = dict[
    str, Union[Basic_Value, "Data_Point", list[Union[Basic_Value, "Data_Point"]]]
]
#: The possible values of any given entry in the data point's tree.
Data_Point_Subtree = Basic_Value | Data_Point | list[Basic_Value | Data_Point]
#: The result of a query on the tree representing a data point.
#: Because we may be accessing multiple nested lists, the dimension
#: of a query result is not certain a-priori.
Query_Result = Terminal_Value | Data_Point | list["Query_Result"]


def get_leaves(data_point: Data_Point) -> set[tuple[str, ...]]:
    """Get all leaf-nodes of a :any:`Data_Point`, as tuples of keys."""
    return _get_leaves(data_point, tuple(), set())


def get_in(
    data_point: Data_Point,
    keys: Sequence[str],
    catch_errors: tuple[type[Exception], ...] = tuple(),
) -> Query_Result:
    """
    Recursively access a :any:`Data_Point`.

    :param catch_errors:
      Errors to catch when accessing the data-point.

      - Add ``KeyError`` to return ``None`` when a data-point does not contain
        the key sequence.
      - Add ``TypeError`` to return ``None`` when a data-point ends before all
        keys have been used up.
    """
    return _get_in(data_point, keys, catch_errors)


def get_terminal_in(
    data_point: Data_Point,
    keys: Sequence[str],
    catch_errors: tuple[type[Exception], ...] = (KeyError, TypeError),
) -> Terminal_Value:
    """
    Like :func:`get_in`, but replace non-:any:`Terminal_Value` results with
    ``None`` and flatten nested lists.
    """
    return _to_terminal(get_in(data_point, keys, catch_errors))


def get_children_map(
    full_schema: Data_Point,
    id_seq: Sequence[str],
    subcategory_fields: Iterable[str],
) -> dict[Basic_Value, tuple[Basic_Value, ...]]:
    """
    Take a nested schema and turn it into a flat map from id to children ids.
    """
    entries: dict[Basic_Value, tuple[Basic_Value, ...]] = dict()
    nodes: list[Data_Point] = [full_schema]
    while len(nodes) > 0:
        node = nodes.pop()
        node_id: Basic_Value = get_terminal_in(node, id_seq)  # type: ignore
        node_children: list[str] = list()

        for subcategory_field in subcategory_fields:
            if subcategory_field not in node:
                continue

            child_nodes: list[Data_Point] = node[subcategory_field]  # type: ignore
            nodes += child_nodes
            node_children += [
                get_terminal_in(child_node, id_seq)  # type: ignore
                for child_node in child_nodes
            ]

        entries[node_id] = tuple(node_children)

    return entries


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def get_parent_map(children_map: dict[_KT, tuple[_VT, ...]]) -> dict[_VT, _KT]:
    entries: dict[_VT, _KT] = dict()
    for parent_id, children in children_map.items():
        for child in children:
            entries[child] = parent_id

    return entries


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
