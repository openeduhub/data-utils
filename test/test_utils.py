from test.strategies import basic_values, mutated_nested_dict

import data_utils.utils as utils
import hypothesis.strategies as st
from data_utils.utils import Basic_Value, Nested_Dict
from hypothesis import given, settings


@given(
    mutated_nested_dict(dict()), st.lists(st.text(min_size=1), min_size=1), basic_values
)
@settings(max_examples=500)
def test_with_new_value(entry: Nested_Dict, fields: list[str], value: Basic_Value):
    new_entry = utils.with_new_value(entry, fields, value)

    print("result:", new_entry)

    get_value = utils.get_in(new_entry, fields)
    if isinstance(get_value, list):
        assert value in get_value
    else:
        assert value is get_value
