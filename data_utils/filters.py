from __future__ import annotations
import operator as op
from collections.abc import Callable, Collection
from functools import partial
from typing import Iterable

from data_utils.data import Basic_Value, Nested_Dict, Terminal_Value, get_terminal_in
from data_utils.defaults import Fields

Filter = Callable[[Nested_Dict], bool]


def get_predicate_on_terminal(
    simple_predicate: Callable[[Basic_Value], bool],
    multi_value_semantics: Callable[[Iterable[bool]], bool],
) -> Callable[[Terminal_Value], bool]:
    """
    Create a predicate function that acts on a terminal value from one
    that acts on basic values.
    """

    def fun(comp_value: Terminal_Value) -> bool:
        if isinstance(comp_value, list):
            return multi_value_semantics([fun(sub_value) for sub_value in comp_value])

        return simple_predicate(comp_value)

    return fun


def get_filter(
    predicate_fun: Callable[[Terminal_Value], bool], key: str, key_separator: str = "."
) -> Callable[[Nested_Dict], bool]:
    """
    Create a filter that returns true iff the given predicate function
    evaluates to True for the value at the given key sequence.
    """

    def fun(entry: Nested_Dict) -> bool:
        value = get_terminal_in(
            entry, key.split(key_separator), catch_errors=(KeyError, TypeError)
        )
        return predicate_fun(value)

    return fun


def get_filter_with_basic_predicate(
    predicate_fun: Callable[[Basic_Value], bool],
    key: str,
    multi_value_semantics: Callable[[Iterable[bool]], bool],
    key_separator: str = ".",
) -> Callable[[Nested_Dict], bool]:
    """
    Convenience function to create filters from predicate functions on basic values.
    """
    return get_filter(
        get_predicate_on_terminal(predicate_fun, multi_value_semantics),
        key=key,
        key_separator=key_separator,
    )


def negated(_filter: Filter) -> Filter:
    """Return a new filter that evaluates to the opposite of the given one."""

    def fun(entry: Nested_Dict) -> bool:
        return not _filter(entry)

    return fun


def kibana_basic_filter(entry: Nested_Dict) -> bool:
    """
    The 'Basic Filter' from Kibana.

    This filters rejects all data that we probably don't want,
    e.g. those that are in another dataset or do not represent a material.
    """
    must_filters = [
        get_filter(get_predicate_on_terminal(partial(op.eq, value), any), field)
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
            "aspects",
        )
    )

    return all(fun(entry) for fun in must_filters)


def kibana_publicly_visible(entry: Nested_Dict) -> bool:
    """
    Only accepts data that is publicly visible.

    This may be because the data is explicitly set to be public,
    or because it is contained within a collection that is public.
    """
    should_filters = [
        get_filter_with_basic_predicate(partial(op.eq, value), field, any)
        for field, value in [
            ("permissions.Read", "GROUP_EVERYONE"),
            ("collections.permissions.Read", "GROUP_EVERYONE"),
        ]
    ]

    return any(fun(entry) for fun in should_filters)


kibana_redaktionsbuffet = get_filter_with_basic_predicate(
    predicate_fun=partial(op.eq, "Redaktionsbuffet"),
    key=Fields.COLLECTIONS.value,
    multi_value_semantics=any,
)


def get_test_data_filter(included_labels: Iterable[str]) -> Filter:
    """
    Get a filter that only include test data according to the given test data labels.
    """
    return get_filter_with_basic_predicate(
        lambda x: any(x == label for label in included_labels), "test_data", any
    )


def get_language_filter(accepted_languages: Collection[str]) -> Filter:
    if len(accepted_languages) == 0:
        return lambda x: True

    def fun(entry: Nested_Dict) -> bool:
        langs = get_terminal_in(entry, Fields.LANGUAGE.value.split("."))

        if langs is None:
            return True

        if not isinstance(langs, list):
            return langs in accepted_languages

        unaccepted_langs = set(langs) - set(accepted_languages)
        return not unaccepted_langs

    return fun


german_filter = get_language_filter(["de", "de_DE", "de-DE"])


def get_labeled_filter(
    fields: Collection[str],
    key_separator: str = ".",
    multi_field_semantics: Callable[[Iterable[bool]], bool] = any,
) -> Filter:
    def fun(entry: Nested_Dict) -> bool:
        found_values = (
            get_terminal_in(entry, field.split(key_separator)) for field in fields
        )
        return multi_field_semantics(bool(value) for value in found_values)

    return fun


def get_len_filter(
    fields: Collection[str],
    min_lengths: int | Collection[int],
    key_separator: str = ".",
    multi_field_semantics: Callable[[Iterable[bool]], bool] = any,
) -> Filter:
    if len(fields) == 0:
        return lambda x: True

    if not isinstance(min_lengths, Collection):
        min_lengths = [min_lengths for _ in fields]

    def fun(entry: Nested_Dict) -> bool:
        found_values = (
            get_terminal_in(entry, field.split(key_separator)) for field in fields
        )
        lengths = (
            len(value) if isinstance(value, Collection) else 0 for value in found_values
        )
        return multi_field_semantics(
            length >= min_length for length, min_length in zip(lengths, min_lengths)
        )

    return fun


existing_text_filter = get_len_filter(
    [Fields.TITLE.value, Fields.DESCRIPTION.value],
    min_lengths=1,
    multi_field_semantics=all,
)
