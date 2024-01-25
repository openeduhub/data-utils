import operator as op
from functools import reduce
from test.strategies import basic_values

import data_utils.transform as trans
import hypothesis.strategies as st
from data_utils.utils import Basic_Value
from hypothesis import given


@given(basic_values)
def test_fix_entry_single_value_no_change(value: Basic_Value):
    assert value is trans.fix_entry_single_value(value, list(), dict())


@given(basic_values, basic_values)
def test_fix_entry_single_value_remap(value: Basic_Value, remap: Basic_Value):
    assert remap is trans.fix_entry_single_value(value, list(), {value: remap})


@given(basic_values)
def test_fix_entry_single_value_drop(value: Basic_Value):
    assert None is trans.fix_entry_single_value(value, [value], dict())


@given(basic_values, basic_values)
def test_fix_entry_single_value_remap_before_drop(
    value: Basic_Value, remap: Basic_Value
):
    assert None is trans.fix_entry_single_value(value, [remap], {value: remap})


@given(st.lists(basic_values.filter(lambda x: x is not None)))
def test_fix_entry_multi_value_no_changes(values: list[Basic_Value]):
    results = trans.fix_entry_multi_value(values, [], dict())

    assert len(results) == len(values)
    assert all(result is value for result, value in zip(results, values))


@given(
    st.lists(basic_values.filter(lambda x: x is not None)),
    st.dictionaries(basic_values, basic_values.filter(lambda x: x is not None)),
)
def test_fix_entry_multi_value_remap(
    values: list[Basic_Value], remap: dict[Basic_Value, Basic_Value]
):
    results = trans.fix_entry_multi_value(values, [], remap)

    assert len(results) == len(values)
    assert all(
        result is remap.get(value, value) for result, value in zip(results, values)
    )


@given(st.lists(basic_values.filter(lambda x: x is not None)), st.sets(basic_values))
def test_fix_entry_multi_value_drop(
    values: list[Basic_Value], to_drop: set[Basic_Value]
):
    results = trans.fix_entry_multi_value(values, to_drop, dict())

    assert all(result not in to_drop for result in results)
    assert all(value in results for value in values if value not in to_drop)


@given(st.lists(basic_values))
def test_fix_entry_multiple_value_drops_none(values: list[Basic_Value]):
    results = trans.fix_entry_multi_value(values, [], dict())
    is_none = [value is None for value in values]

    assert None not in results
    assert len(results) == (len(values) - sum(is_none))
    assert all(value in results for value in values if value is not None)


@given(
    st.lists(basic_values),
    st.dictionaries(basic_values, basic_values.filter(lambda x: x is not None)),
)
def test_fix_entry_multi_value_remap_before_drop(
    values: list[Basic_Value], remap: dict[Basic_Value, Basic_Value]
):
    results = trans.fix_entry_multi_value(
        values, [remap.get(value, value) for value in values], remap
    )

    assert not results


@given(st.data(), st.lists(st.one_of(st.lists(basic_values))))
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
        values = set(values)
        for j, value in enumerate(test_unique_values):
            assert arr[i, j] == (value in values)
