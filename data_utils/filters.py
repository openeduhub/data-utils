from __future__ import annotations
import operator as op
from collections.abc import Callable, Collection
from functools import partial
from typing import Iterable

from data_utils.data import Basic_Value, Data_Point, Terminal_Value, get_terminal_in
from data_utils.defaults import Fields

#: A filter evaluates a particular data entry
#: and returns True iff. it shall be kept.
Filter = Callable[[Data_Point], bool]

#: A predicate evaluates a particular (potentially multi-value) field's
#: value(s) and returns True (the value(s) shall be accepted) or False
#: (the value(s) shall be rejected)
Predicate = Callable[[Terminal_Value], bool]

#: A simple predicate is like a ``Predicate``,
#: but only applies to single values.
Simple_Predicate = Callable[[Basic_Value], bool]

#: A multi-value predicate takes multiple ``Predicate`` or ``Simple_Predicate``
#: results and summarizes them into one Boolean value.
#:
#: Examples: The builtin ``any`` and ``all``
Multi_Value_Predicate = Callable[[Iterable[bool]], bool]


def get_predicate_on_terminal(
    simple_predicate: Simple_Predicate,
    multi_value_semantics: Multi_Value_Predicate,
) -> Predicate:
    """
    Create a predicate function that acts on a terminal value from one
    that acts on basic values.

    This allows for the simpler creation of predicate functions,
    as additional semantics stemming from multi-value fields can be handled
    automatically.

    :param multi_value_semantics: The predicate function to use in order to
        summarize multi-value fields into one Boolean value.
        Usually set to `any` or `all`, depending on the desired behavior.
    """

    def fun(comp_value: Terminal_Value) -> bool:
        if isinstance(comp_value, list):
            return multi_value_semantics([fun(sub_value) for sub_value in comp_value])

        return simple_predicate(comp_value)

    return fun


def get_filter(
    predicate_fun: Predicate,
    field: str,
    separator: str = ".",
) -> Filter:
    """
    Create a filter keeps entries given by the predicate function evaluated
    on a particular field.

    :param predicate_fun: The predicate function to apply to values
         in the given ``field``.
    :param field: The field which the filter shall evaluate ``predicate_fun``
         on.
    :param separator: The separator to use for splitting ``field``.
    """

    def fun(entry: Data_Point) -> bool:
        value = get_terminal_in(
            entry, field.split(separator), catch_errors=(KeyError, TypeError)
        )
        return predicate_fun(value)

    return fun


def get_filter_with_basic_predicate(
    predicate_fun: Simple_Predicate,
    field: str,
    multi_value_semantics: Multi_Value_Predicate,
    separator: str = ".",
) -> Filter:
    """
    Create a filter from a predicate function on basic values.

    Essentially just a combines :func:`get_predicate_on_terminal` and
    :func:`get_filter` in order to derive the filter to return.
    """
    return get_filter(
        get_predicate_on_terminal(predicate_fun, multi_value_semantics),
        field=field,
        separator=separator,
    )


def negated(_filter: Filter) -> Filter:
    """Create a new filter that evaluates to the opposite of the given one."""

    def fun(entry: Data_Point) -> bool:
        return not _filter(entry)

    return fun


def kibana_basic_filter(entry: Data_Point) -> bool:
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


def kibana_publicly_visible(entry: Data_Point) -> bool:
    """
    A filter that only accepts data that is publicly visible.

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


#: A filter that only accepts data that has been confirmed by editors
#: (i.e. it is located within the "Redaktionsbuffet")
kibana_redaktionsbuffet: Filter = get_filter_with_basic_predicate(
    predicate_fun=partial(op.eq, "Redaktionsbuffet"),
    field=Fields.COLLECTIONS.value,
    multi_value_semantics=any,
)


def get_test_data_filter(included_labels: Iterable[str]) -> Filter:
    """
    Create a filter that only includes data if it is part of a test data set
    with any of the given labels.
    """
    included_labels_set = set(included_labels)
    return get_filter_with_basic_predicate(
        lambda x: x in included_labels_set, "test_data", any
    )


def get_language_filter(accepted_languages: Collection[str]) -> Filter:
    """
    Create a filter that rejects any data that contains a not explicitly
    allowed language.
    """
    if len(accepted_languages) == 0:
        return lambda _: True

    def fun(entry: Data_Point) -> bool:
        langs = get_terminal_in(entry, Fields.LANGUAGE.value.split("."))

        if langs is None:
            return True

        if not isinstance(langs, list):
            return langs in accepted_languages

        unaccepted_langs = set(langs) - set(accepted_languages)
        return not unaccepted_langs

    return fun


#: A filter that only accepts data that *only* contains German content
german_filter: Filter = get_language_filter(["de", "de_DE", "de-DE"])


def get_labeled_filter(
    fields: Collection[str],
    separator: str = ".",
    multi_field_semantics: Multi_Value_Predicate = any,
) -> Filter:
    """
    Create a filter that only accepts data that contains values in the given
    fields.

    :param multi_field_semantics: If multiple values for ``fields`` have been
        given, this determines how they shall be combined (e.g. does only one
        field need to contain a value or do all fields need to?)
    """

    def fun(entry: Data_Point) -> bool:
        found_values = (
            get_terminal_in(entry, field.split(separator)) for field in fields
        )
        return multi_field_semantics(bool(value) for value in found_values)

    return fun


def get_len_filter(
    fields: Collection[str],
    min_lengths: int | Collection[int],
    separator: str = ".",
    multi_field_semantics: Multi_Value_Predicate = any,
) -> Filter:
    """
    Create a filter that only accepts data that contains values with a minimum
    length.

    :param min_lengths:
        - If a single integer, all values from all given fields have their
          length compared to this value.
        - If a list, the first field\'s length is compared to the first value,
          the second to the second...

          Note that it is assumed that ``fields`` and ``min_lengths`` are of
          the same length.
    :param multi_field_semantics: If multiple fields have been given, this
        determines how they shall be combined (e.g. does only one field need to
        contain a value or do all fields need to?)
    """
    if len(fields) == 0:
        return lambda _: True

    if not isinstance(min_lengths, Collection):
        min_lengths = [min_lengths for _ in fields]

    def fun(entry: Data_Point) -> bool:
        found_values = (
            get_terminal_in(entry, field.split(separator)) for field in fields
        )
        lengths = (
            len(value) if isinstance(value, Collection) else 0 for value in found_values
        )
        return multi_field_semantics(
            length >= min_length for length, min_length in zip(lengths, min_lengths)
        )

    return fun


#: A filter that only keeps data that includes both a title and a description.
existing_text_filter: Filter = get_len_filter(
    [Fields.TITLE.value, Fields.DESCRIPTION.value],
    min_lengths=1,
    multi_field_semantics=all,
)
