from collections.abc import Iterable
from test.strategies import basic_values, mutated_nested_dict

import data_utils.filters as filt
import hypothesis.strategies as st
from data_utils.utils import Nested_Dict
from hypothesis import given


@given(
    mutated_nested_dict(
        {
            "nodeRef": {"storeRef": {"protocol": "workspace"}},
            "type": "ccm:io",
            "properties": {"cm:edu_metadataset": "mds_oeh"},
            "aspects": [],
        },
        value_blacklist=["ccm:collection_io_reference"],
        mod_key_blacklist=["nodeRef", "type", "properties"],
        min_iters=0,
    )
)
def test_kibana_basic_filter_accepts_from_minimal(entry: Nested_Dict):
    assert filt.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        {
            "nodeRef": {"storeRef": {"protocol": "workspace"}},
            "type": "ccm:io",
            "properties": {"cm:edu_metadataset": "mds_oeh"},
        },
        allow_remove_from_list=False,  # would not falsify
        allow_add_key=False,  # would not falsify
        min_iters=1,
    )
)
def test_kibana_basic_filter_rejects_from_minimal(entry: Nested_Dict):
    assert not filt.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        {
            "nodeRef": {"storeRef": {"protocol": "workspace"}},
            "type": "ccm:io",
            "properties": {"cm:edu_metadataset": "mds_oeh"},
            "aspects": ["ccm:collection_io_reference"],
        },
        mod_key_blacklist=["aspects"],
        min_iters=0,
    )
)
def test_kibana_basic_filter_rejects_with_bad_aspect(entry: Nested_Dict):
    assert not filt.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        dict(),
        value_blacklist=["workspace"],
        min_iters=0,
    )
)
def test_kibana_basic_filter_rejects_without_workspace(entry: Nested_Dict):
    assert not filt.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        dict(),
        value_blacklist=["ccm:io"],
        min_iters=0,
    )
)
def test_kibana_basic_filter_rejects_without_ccm_io(entry: Nested_Dict):
    assert not filt.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        dict(),
        value_blacklist=["mds_oeh"],
        min_iters=0,
    )
)
def test_kibana_basic_filter_rejects_without_mds_oeh(entry: Nested_Dict):
    assert not filt.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        {
            "permissions": {"Read": ["foo", "GROUP_EVERYONE", "bar"]},
        },
        mod_key_blacklist=["permissions"],
        min_iters=0,
    )
)
def test_kibana_publicly_visible_accepts_from_minimal_self(entry: Nested_Dict):
    assert filt.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        {
            "collections": [
                {"permissions": {"Read": ["foo", "GROUP_EVERYONE", "bar"]}},
                "baz",
            ],
        },
        mod_key_blacklist=["collections"],
        min_iters=0,
    )
)
def test_kibana_publicly_visible_accepts_from_minimal_collection(entry: Nested_Dict):
    assert filt.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        {
            "permissions": {"Read": ["GROUP_EVERYONE"]},
        },
        allow_add_key=False,  # would not falsify
        allow_add_to_list=False,  # would not falsify
        min_iters=1,
    )
)
def test_kibana_publicly_visible_rejects_from_minimal_self(entry: Nested_Dict):
    assert not filt.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        {
            "collections": [{"permissions": {"Read": ["GROUP_EVERYONE"]}}],
        },
        allow_add_key=False,  # would not falsify
        allow_add_to_list=False,  # would not falsify
        min_iters=1,
    )
)
def test_kibana_publicly_visible_rejects_from_minimal_collection(entry: Nested_Dict):
    assert not filt.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        dict(),
        value_blacklist=["GROUP_EVERYONE"],
        min_iters=0,
    )
)
def test_kibana_publicly_visible_rejects_without_adding_correct_value(
    entry: Nested_Dict,
):
    assert not filt.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        {
            "collections": [
                "foo",
                {"properties": {"cm:title": "Redaktionsbuffet"}},
                "bar",
            ]
        },
        mod_key_blacklist=["collections"],
        min_iters=0,
    )
)
def test_kibana_redaktionsbuffet_accepts_from_minimal(entry: Nested_Dict):
    assert filt.kibana_redaktionsbuffet(entry)


@given(
    mutated_nested_dict(
        {"collections": [{"properties": {"cm:title": "Redaktionsbuffet"}}]},
        allow_add_key=False,  # would not falsify
        allow_add_to_list=False,  # would not falsify
        value_blacklist={"Redaktionsbuffet"},  # would not falsify
        min_iters=1,
    )
)
def test_kibana_redaktionsbuffet_rejects_from_minimal(entry: Nested_Dict):
    assert not filt.kibana_redaktionsbuffet(entry)


@given(
    mutated_nested_dict(
        dict(),
        value_blacklist={"Redaktionsbuffet"},
        min_iters=1,
    )
)
def test_kibana_redaktionsbuffet_rejects_without_correct_value(entry: Nested_Dict):
    assert not filt.kibana_redaktionsbuffet(entry)


@given(
    st.data(),
    mutated_nested_dict(
        {"properties": {"cclom:general_language": []}},
        mod_key_blacklist="properties",
        min_iters=0,
    ),
    st.lists(st.text(), max_size=10, unique=True),
)
def test_language_filter_accepts(data, entry: Nested_Dict, langs: list[str]):
    # add any number of accepted languages
    if len(langs) > 0:
        entry["properties"]["cclom:general_language"] = data.draw(
            st.lists(st.sampled_from(langs))
        )

    fun = filt.get_language_filter(langs)
    assert fun(entry)


@given(
    st.data(),
    mutated_nested_dict(
        {"properties": {"cclom:general_language": []}},
        mod_key_blacklist="properties",
        min_iters=0,
    ),
    st.lists(
        st.text().filter(lambda x: "." not in x), min_size=1, max_size=10, unique=True
    ),
)
def test_language_filter_rejects_one_incorrect(
    data, entry: Nested_Dict, langs: list[str]
):
    # add any number of accepted languages, plus one that is not
    entry["properties"]["cclom:general_language"] = data.draw(
        st.lists(st.sampled_from(langs))
    ) + [data.draw(st.text().filter(lambda x: x not in langs))]

    fun = filt.get_language_filter(langs)
    assert not fun(entry)


@given(
    mutated_nested_dict(
        {"properties": {"cclom:general_language": []}},
        mod_key_blacklist="properties",
        min_iters=0,
    ),
    st.lists(
        st.text().filter(lambda x: "." not in x), min_size=1, max_size=10, unique=True
    ),
)
def test_language_filter_accepts_none(entry: Nested_Dict, langs: list[str]):
    entry["properties"]["cclom:general_language"] = None

    fun = filt.get_language_filter(langs)
    assert fun(entry)


@given(
    mutated_nested_dict(
        {"properties": {"cclom:general_language": []}},
        min_iters=0,
    )
)
def test_language_filter_accepts_with_empty_accepted_langs(entry: Nested_Dict):
    langs = []
    fun = filt.get_language_filter(langs)
    assert fun(entry)


@given(
    st.data(),
    st.lists(
        st.text().filter(lambda x: "." not in x), min_size=1, max_size=10, unique=True
    ),
)
def test_labeled_filter_accepts_with_any(data, fields: list[str]):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=fields,
            add_key_blacklist=fields,
        )
    )

    for field in data.draw(st.lists(st.sampled_from(fields), min_size=1, unique=True)):
        entry[field] = data.draw(basic_values.filter(bool))

    fun = filt.get_labeled_filter(fields, multi_field_semantics=any)
    assert fun(entry)


@given(
    st.data(),
    st.lists(
        st.text().filter(lambda x: "." not in x), min_size=1, max_size=10, unique=True
    ),
)
def test_labeled_filter_accepts_with_all(data, fields: list[str]):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=fields,
            add_key_blacklist=fields,
        )
    )

    for field in fields:
        entry[field] = data.draw(basic_values.filter(bool))

    fun = filt.get_labeled_filter(fields, multi_field_semantics=all)
    assert fun(entry)


@given(st.data(), st.text())
def test_labeled_filter_rejects_none(data, field: str):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=[field],
            add_key_blacklist=[field],
        )
    )

    entry[field] = None

    fun = filt.get_labeled_filter([field], multi_field_semantics=any)
    assert not fun(entry)


@given(st.data(), st.text())
def test_labeled_filter_rejects_empty_list(data, field: str):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=[field],
            add_key_blacklist=[field],
        )
    )

    entry[field] = []

    fun = filt.get_labeled_filter([field], multi_field_semantics=any)
    assert not fun(entry)


@given(st.data(), st.text())
def test_labeled_filter_rejects_empty_string(data, field: str):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=[field],
            add_key_blacklist=[field],
        )
    )

    entry[field] = ""

    fun = filt.get_labeled_filter([field], multi_field_semantics=any)
    assert not fun(entry)


@given(
    st.data(),
    st.lists(
        st.text().filter(lambda x: "." not in x), min_size=1, max_size=10, unique=True
    ),
)
def test_labeled_filter_rejects_all_one_missing(data, fields: list[str]):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=fields,
            add_key_blacklist=fields,
        )
    )

    for field in data.draw(
        st.lists(st.sampled_from(fields), min_size=0, max_size=len(fields) - 1)
    ):
        entry[field] = data.draw(basic_values)

    fun = filt.get_labeled_filter(fields, multi_field_semantics=all)
    assert not fun(entry)


@given(
    st.data(),
    st.lists(
        st.text().filter(lambda x: "." not in x), min_size=1, max_size=10, unique=True
    ),
)
def test_labeled_filter_rejects_any_all_missing(data, fields: list[str]):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=fields,
            add_key_blacklist=fields,
        )
    )

    fun = filt.get_labeled_filter(fields, multi_field_semantics=any)
    assert not fun(entry)


@given(
    st.data(),
    st.lists(
        st.text().filter(lambda x: "." not in x), min_size=1, max_size=10, unique=True
    ),
)
def test_len_filter_accepts_any_any_true(data, fields: list[str]):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=fields,
            add_key_blacklist=fields,
        )
    )
    wrong_lengths = data.draw(
        st.lists(st.booleans(), min_size=len(fields), max_size=len(fields)).filter(
            lambda x: sum(x) < len(fields)
        )
    )

    min_lengths = [data.draw(st.integers(min_value=0, max_value=100)) for _ in fields]

    for field, size, wrong_length in zip(fields, min_lengths, wrong_lengths):
        if wrong_length:
            entry[field] = data.draw(st.lists(st.none(), max_size=max(0, size - 1)))
        else:
            entry[field] = data.draw(st.lists(st.none(), min_size=size))

    fun = filt.get_len_filter(fields, min_lengths, multi_field_semantics=any)

    assert fun(entry)


@given(
    st.data(),
    st.lists(
        st.text().filter(lambda x: "." not in x), min_size=1, max_size=10, unique=True
    ),
)
def test_len_filter_accepts_all_all_true(data, fields: list[str]):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=fields,
            add_key_blacklist=fields,
        )
    )

    min_lengths = [data.draw(st.integers(min_value=0, max_value=100)) for _ in fields]

    for field, size in zip(fields, min_lengths):
        entry[field] = data.draw(st.lists(st.none(), min_size=size))

    fun = filt.get_len_filter(fields, min_lengths, multi_field_semantics=all)

    assert fun(entry)


@given(
    st.data(),
    st.lists(
        st.text().filter(lambda x: "." not in x), min_size=1, max_size=10, unique=True
    ),
)
def test_len_filter_rejects_any_all_false(data, fields: list[str]):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=fields,
            add_key_blacklist=fields,
        )
    )

    min_lengths = [data.draw(st.integers(min_value=1, max_value=100)) for _ in fields]

    for field, size in zip(fields, min_lengths):
        entry[field] = data.draw(st.lists(st.none(), max_size=size - 1))

    fun = filt.get_len_filter(fields, min_lengths, multi_field_semantics=any)

    assert not fun(entry)


@given(
    st.data(),
    st.lists(
        st.text().filter(lambda x: "." not in x), min_size=1, max_size=10, unique=True
    ),
)
def test_len_filter_rejects_all_any_false(data: st.DataObject, fields: list[str]):
    entry = data.draw(
        mutated_nested_dict(
            dict(),
            min_iters=0,
            mod_key_blacklist=fields,
            add_key_blacklist=fields,
        )
    )

    wrong_lengths = data.draw(
        st.lists(st.booleans(), min_size=len(fields), max_size=len(fields)).filter(
            lambda x: sum(x) > 0
        )
    )
    min_lengths = [data.draw(st.integers(min_value=1, max_value=100)) for _ in fields]

    for field, size, wrong_length in zip(fields, min_lengths, wrong_lengths):
        if wrong_length:
            entry[field] = data.draw(st.lists(st.none(), max_size=max(0, size - 1)))
        else:
            entry[field] = data.draw(st.lists(st.none(), min_size=size))

    fun = filt.get_len_filter(fields, min_lengths, multi_field_semantics=all)

    assert not fun(entry)


@given(
    mutated_nested_dict(
        {
            "properties": {
                "cclom:title": "foo",
                "cclom:general_description": "bar",
            },
        },
        mod_key_blacklist=["properties"],
        min_iters=0,
    )
)
def test_existing_text_filter_accepts(entry: Nested_Dict):
    assert filt.existing_text_filter(entry)


@given(
    mutated_nested_dict(
        {
            "properties": {
                "cclom:title": "foo",
                "cclom:general_description": "",
            },
        },
        mod_key_blacklist=["properties"],
        min_iters=0,
    )
)
def test_existing_text_filter_rejects_empty_description(entry: Nested_Dict):
    assert not filt.existing_text_filter(entry)


@given(
    mutated_nested_dict(
        {
            "properties": {
                "cclom:title": "",
                "cclom:general_description": "bar",
            },
        },
        mod_key_blacklist=["properties"],
        min_iters=0,
    )
)
def test_existing_text_filter_rejects_empty_title(entry: Nested_Dict):
    assert not filt.existing_text_filter(entry)


@given(
    mutated_nested_dict(
        {
            "properties": {
                "cclom:general_description": "bar",
            },
        },
        mod_key_blacklist=["properties"],
        min_iters=0,
    )
)
def test_existing_text_filter_rejects_no_title(entry: Nested_Dict):
    assert not filt.existing_text_filter(entry)


@given(
    mutated_nested_dict(
        {
            "properties": {
                "cclom:title": "foo",
            },
        },
        mod_key_blacklist=["properties"],
        min_iters=0,
    )
)
def test_existing_text_filter_rejects_no_description(entry: Nested_Dict):
    assert not filt.existing_text_filter(entry)
