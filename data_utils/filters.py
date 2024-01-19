import operator as op
from collections.abc import Callable, Collection, Sequence
from functools import partial
from typing import TypeVar

from data_utils.utils import Basic_Value, Nested_Dict, Terminal_Value, get_terminal_in

Filter = Callable[[Nested_Dict], bool]

T = TypeVar("T")


def get_predicate_on_terminal(
    simple_predicate: Callable[[Basic_Value], bool],
    list_semantics: Callable[[Collection[bool]], bool],
) -> Callable[[Terminal_Value], bool]:
    def fun(comp_value: Terminal_Value) -> bool:
        if isinstance(comp_value, list):
            return list_semantics([fun(sub_value) for sub_value in comp_value])

        return simple_predicate(comp_value)

    return fun


def get_filter(
    predicate_fun: Callable[[Terminal_Value], bool], key_seq: Sequence[str]
) -> Callable[[Nested_Dict], bool]:
    def fun(entry: Nested_Dict) -> bool:
        value = get_terminal_in(entry, key_seq, catch_errors=(KeyError, TypeError))
        return predicate_fun(value)

    return fun


def get_filter_with_basic_predicate(
    predicate_fun: Callable[[Basic_Value], bool],
    key_seq: Sequence[str],
    list_semantics: Callable[[Collection[bool]], bool],
) -> Callable[[Nested_Dict], bool]:
    return get_filter(get_predicate_on_terminal(predicate_fun, list_semantics), key_seq)


def kibana_basic_filter(entry: Nested_Dict) -> bool:
    must_filters = [
        get_filter(
            get_predicate_on_terminal(partial(op.eq, value), any), field.split(".")
        )
        for field, value in [
            ("nodeRef.storeRef.protocol", "workspace"),
            ("type", "ccm:io"),
            ("properties.cm:edu_metadataset", "mds_oeh"),
        ]
    ]
    must_filters.append(
        get_filter(
            get_predicate_on_terminal(
                partial(op.ne, "ccm:collection_io_reference"), all
            ),
            ["aspects"],
        )
    )

    return all(fun(entry) for fun in must_filters)


def kibana_publicly_visible(entry: Nested_Dict) -> bool:
    should_filters = [
        get_filter_with_basic_predicate(partial(op.eq, value), field.split("."), any)
        for field, value in [
            ("permissions.Read", "GROUP_EVERYONE"),
            ("collections.permissions.Read", "GROUP_EVERYONE"),
        ]
    ]

    return any(fun(entry) for fun in should_filters)


kibana_redaktionsbuffet = get_filter_with_basic_predicate(
    predicate_fun=partial(op.eq, "Redaktionsbuffet"),
    key_seq="collections.properties.cm:title".split("."),
    list_semantics=any,
)
