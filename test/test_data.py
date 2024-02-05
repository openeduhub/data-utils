import operator as op
from functools import reduce
import test.strategies as myst

import hypothesis.strategies as st
from data_utils.data import (
    Basic_Value,
    Basic_Value_Not_None,
    Data_Point,
    Terminal_Value,
)
import data_utils.data as data
from hypothesis import given, settings


def test_get_leaves_static():
    entry = Data_Point(
        {
            "a": 1,
            "b": [1, 2, 3],
            "c": {"d": {"e": 1}},
            "f": [{"g": {"h": 1}}, {"g": {"h": 3, "i": 10}, "j": 5}],
        }
    )
    assert data.get_leaves(entry) == {
        ("a",),
        ("b",),
        ("c", "d", "e"),
        ("f", "g", "h"),
        ("f", "g", "i"),
        ("f", "j"),
    }


@given(myst.mutated_data_point({}))
def test_all_leaves_are_terminals(entry: Data_Point):
    leaves = data.get_leaves(entry)
    terminals = [data.get_in(entry, leaf) for leaf in leaves]
    for terminal in terminals:
        assert isinstance(terminal, Basic_Value | list)

        if isinstance(terminal, list) and len(terminal) > 0:
            assert any(isinstance(value, Basic_Value) for value in terminal)
