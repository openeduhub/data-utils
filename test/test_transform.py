import operator as op
import test.strategies as myst
from collections.abc import Iterable
from functools import reduce

import data_utils.transform as trans
import hypothesis.strategies as st
from data_utils.utils import (
    Basic_Value,
    Basic_Value_Not_None,
    Nested_Dict,
    get_in,
    get_leaves,
    get_terminal_in,
)
import pytest
from hypothesis import given, settings


def untouched_are_unchanged(
    old: Nested_Dict, new: Nested_Dict, changed_field: Iterable[str]
):
    fields_to_check = (get_leaves(old) | get_leaves(new)) - {tuple(changed_field)}
    for field in fields_to_check:
        old_value = get_terminal_in(old, field)
        new_value = get_terminal_in(new, field)
        if old_value is not new_value and old_value != new_value:
            print(f"{field=}")
            print(f"{old_value=}")
            print(f"{new_value=}")
            return False

    return True


@given(myst.basic_values)
def test_fix_entry_single_value_no_change(value: Basic_Value):
    assert value is trans.change_single_value(value, list(), dict())


@given(myst.basic_values_not_none, myst.basic_values)
def test_fix_entry_single_value_remap(value: Basic_Value_Not_None, remap: Basic_Value):
    assert remap is trans.change_single_value(value, list(), {value: remap})


@given(myst.basic_values_not_none)
def test_fix_entry_single_value_drop(value: Basic_Value_Not_None):
    assert None is trans.change_single_value(value, [value], dict())


@given(myst.basic_values_not_none, myst.basic_values_not_none)
def test_fix_entry_single_value_remap_before_drop(
    value: Basic_Value_Not_None, remap: Basic_Value_Not_None
):
    assert None is trans.change_single_value(value, [remap], {value: remap})


@given(
    st.data(),
    myst.mutated_nested_dict(
        {
            "target_value": 1,
            "target_list": list(range(3)),
            "target_nested_dict": [
                {"a": 1},
                {"a": 2},
                {"a": 3, "b": ["foo"]},
                {"foo": "bar"},
            ],
        },
        allow_cut=False,
        allow_replace=False,
    ),
    st.sampled_from([["target_value"], ["target_list"], ["target_nested_dict", "a"]]),
)
def test_with_changed_value_do_nothing(
    data: st.DataObject, entry: Nested_Dict, fields: list[str]
):
    new_entry = trans.with_changed_value(entry, fields, [], dict())
    assert new_entry == entry


def test_with_changed_value_raises_value_error():
    with pytest.raises(ValueError):
        trans.with_changed_value({}, [], [None], {})  # type: ignore
        trans.with_changed_value({}, [], [], {None: "foo"})  # type: ignore


@given(
    st.data(),
    myst.mutated_nested_dict({"value": {"subtree": 1}}, min_iters=0),
)
def test_with_changed_value_drop_from_value(data: st.DataObject, entry: Nested_Dict):
    fields = ["value", "subtree"]
    value = get_terminal_in(entry, fields)

    # not valid for this test
    if not isinstance(value, Basic_Value):
        return

    to_drop = [value] if value is not None else []

    new_entry = trans.with_changed_value(entry, fields, to_drop, dict())
    new_value = get_terminal_in(new_entry, fields)

    # we just dropped this value
    assert new_value is None

    # nothing else may have changed
    assert untouched_are_unchanged(entry, new_entry, fields)


@given(
    st.data(),
    myst.mutated_nested_dict({"list": {"subtree": list(range(3))}}, min_iters=0),
)
def test_with_changed_value_drop_from_list(data: st.DataObject, entry: Nested_Dict):
    fields = ["list", "subtree"]
    values = get_terminal_in(entry, fields)

    # not valid for this test
    if not isinstance(values, list):
        return

    values_set = set(values)

    to_drop: set[Basic_Value_Not_None] = (
        data.draw(st.sets(st.sampled_from(list(values_set - {None}))))
        if len(values_set - {None}) > 0
        else set()
    )

    new_entry = trans.with_changed_value(entry, fields, to_drop, dict())

    new_values = get_terminal_in(new_entry, fields)
    assert isinstance(new_values, list)

    new_values_set = set(new_values)

    # the resulting values are not necessarily equal to values_set - to_drop,
    # because None values may have been introduced
    assert all(value not in new_values_set for value in to_drop)
    assert all(value in new_values_set for value in (values_set - to_drop))
    assert untouched_are_unchanged(entry, new_entry, fields)


@given(
    st.data(),
    myst.mutated_nested_dict(
        {
            "dict": [
                {"a": 1},
                {"a": 2},
                {"a": 3, "b": ["foo"]},
                {"foo": "bar"},
            ]
        }
    ),
)
def test_with_changed_value_drop_from_nested_dict(
    data: st.DataObject, entry: Nested_Dict
):
    fields = data.draw(
        st.sampled_from(
            [["dict", "a"], ["dict", "b"], ["dict", "foo"], ["dict", "not-there"]]
        )
    )
    values = get_terminal_in(entry, fields)

    if not isinstance(values, list):
        values_set = {values}
    else:
        values_set = set(values)

    to_drop: set[Basic_Value_Not_None] = (
        data.draw(st.sets(st.sampled_from(list(values_set - {None}))))
        if len(values_set - {None}) > 0
        else set()
    )

    new_entry = trans.with_changed_value(entry, fields, to_drop, dict())

    new_values = get_terminal_in(new_entry, fields)
    if not isinstance(new_values, list):
        new_values_set = {new_values}
    else:
        new_values_set = set(new_values)

    assert all(value not in new_values_set for value in to_drop)
    assert all(value in new_values_set for value in (values_set - to_drop))
    assert untouched_are_unchanged(entry, new_entry, fields)


@given(
    st.data(),
    myst.mutated_nested_dict(
        {
            "value": 1,
            "list": list(range(3)),
            "dict": [
                {"a": 1},
                {"a": 2},
                {"a": 3, "b": ["foo"]},
                {"foo": "bar"},
            ],
        },
        allow_cut=False,
        allow_replace=False,
    ),
    st.sampled_from(
        [
            ["value"],
            ["list"],
            ["dict", "a"],
            ["dict", "b"],
            ["dict", "foo"],
            ["dict", "not-there"],
        ]
    ),
)
def test_with_changed_value_remap(
    data: st.DataObject,
    entry: Nested_Dict,
    fields: list[str],
):
    values = get_terminal_in(entry, fields)
    if not isinstance(values, list):
        values_set = {values}
    else:
        values_set = set(values)

    to_remap: dict[Basic_Value_Not_None, Basic_Value] = (
        data.draw(
            st.dictionaries(
                st.sampled_from(list(values_set - {None})), myst.basic_values
            )
        )
        if len(values_set - {None}) > 0
        else dict()
    )

    new_entry = trans.with_changed_value(entry, fields, [], to_remap)

    new_values = get_terminal_in(new_entry, fields)
    if not isinstance(new_values, list):
        new_values_set = {new_values}
    else:
        new_values_set = set(new_values)

    # the new values should be exactly the set of remapped old values
    assert new_values_set == {
        to_remap.get(value, value) for value in values_set  # type: ignore
    }

    # the number of items in lists should not change through remapping
    if isinstance(values, list):
        assert isinstance(new_values, list)
        assert len(new_values) == len(values)

    assert untouched_are_unchanged(entry, new_entry, fields)


@given(st.data(), st.lists(st.one_of(st.lists(myst.basic_values))))
def test_boolean_array(data: st.DataObject, nested_values: list[list[Basic_Value]]):
    true_unique_values: list[Basic_Value] = list(set(reduce(op.add, nested_values, [])))
    permutation = data.draw(st.permutations(true_unique_values))
    sort_fn = lambda x: sorted(x, key=permutation.index)

    arr, test_unique_values = trans.as_boolean_array(nested_values, sort_fn=sort_fn)

    if not nested_values:
        assert arr.size == 0
        assert not test_unique_values
        return

    assert len(nested_values) == arr.shape[-2]
    assert len(true_unique_values) == arr.shape[-1]
    assert len(true_unique_values) == len(test_unique_values)
    assert set(true_unique_values) == set(test_unique_values)
    assert all(x is y for x, y in zip(permutation, test_unique_values))

    for i, values in enumerate(nested_values):
        values_set = set(values)
        for j, value in enumerate(test_unique_values):
            assert arr[i, j] == (value in values_set)
