from test.strategies import mutated_nested_dict

import data_utils.filters as filters
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
        allow_remove_from_list=False,  # pointless
        allow_cut=False,  # would falsify example
        allow_replace=False,  # would falsify example
        min_iters=0,
    )
)
def test_kibana_basic_filter_accepts_from_minimal(entry: Nested_Dict):
    assert filters.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        {
            "nodeRef": {"storeRef": {"protocol": "workspace"}},
            "type": "ccm:io",
            "properties": {"cm:edu_metadataset": "mds_oeh"},
        },
        value_whitelist=["ccm:collection_io_reference"],
        allow_remove_from_list=False,
        allow_add_key=False,
    )
)
def test_kibana_basic_filter_rejects_from_minimal(entry: Nested_Dict):
    assert not filters.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        {
            "nodeRef": {"storeRef": {"protocol": "workspace"}},
            "type": "ccm:io",
            "properties": {"cm:edu_metadataset": "mds_oeh"},
            "aspects": ["ccm:collection_io_reference"],
        },
        allow_add_key=True,
        allow_add_to_list=True,
        allow_remove_from_list=False,
        allow_cut=False,
        allow_replace=False,
        min_iters=0,
    )
)
def test_kibana_basic_filter_rejects_with_bad_aspect(entry: Nested_Dict):
    assert not filters.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        {},
        value_blacklist=["workspace"],
        min_iters=0,
    )
)
def test_kibana_basic_filter_rejects_without_workspace(entry: Nested_Dict):
    assert not filters.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        {},
        value_blacklist=["ccm:io"],
        min_iters=0,
    )
)
def test_kibana_basic_filter_rejects_without_ccm_io(entry: Nested_Dict):
    assert not filters.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        {},
        value_blacklist=["mds_oeh"],
        min_iters=0,
    )
)
def test_kibana_basic_filter_rejects_without_mds_oeh(entry: Nested_Dict):
    assert not filters.kibana_basic_filter(entry)


@given(
    mutated_nested_dict(
        {
            "permissions": {"Read": "GROUP_EVERYONE"},
        },
        allow_remove_from_list=False,  # would falsify example
        allow_cut=False,  # would falsify example
        allow_replace=False,  # would falsify example
        min_iters=0,
    )
)
def test_kibana_publicly_visible_accepts_from_minimal_self(entry: Nested_Dict):
    assert filters.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        {
            "collections": [{"permissions": {"Read": "GROUP_EVERYONE"}}],
        },
        allow_remove_from_list=False,  # would falsify example
        allow_cut=False,  # would falsify example
        allow_replace=False,  # would falsify example
        min_iters=0,
    )
)
def test_kibana_publicly_visible_accepts_from_minimal_collection(entry: Nested_Dict):
    assert filters.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        {
            "permissions": {"Read": "GROUP_EVERYONE"},
        },
        allow_add_key=False,  # would not falsify
        allow_add_to_list=False,  # would not falsify
        min_iters=1,
    )
)
def test_kibana_publicly_visible_rejects_from_minimal_self(entry: Nested_Dict):
    assert not filters.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        {
            "collections": [{"permissions": {"Read": "GROUP_EVERYONE"}}],
        },
        allow_add_key=False,  # would not falsify
        allow_add_to_list=False,  # would not falsify
        min_iters=1,
    )
)
def test_kibana_publicly_visible_rejects_from_minimal_collection(entry: Nested_Dict):
    assert not filters.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        {},
        value_blacklist=["GROUP_EVERYONE"],
        min_iters=0,
    )
)
def test_kibana_publicly_visible_rejects_stays(entry: Nested_Dict):
    assert not filters.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        {},
        value_blacklist=["GROUP_EVERYONE"],
        min_iters=1,
    )
)
def test_kibana_publicly_visible_rejects_without_adding_correct_value(
    entry: Nested_Dict,
):
    assert not filters.kibana_publicly_visible(entry)


@given(
    mutated_nested_dict(
        {"collections": [{"properties": {"cm:title": "Redaktionsbuffet"}}]},
        allow_cut=False,
        allow_replace=False,
        allow_remove_from_list=False,
        min_iters=0,
    )
)
def test_kibana_redaktionsbuffet_accepts_from_minimal(entry: Nested_Dict):
    assert filters.kibana_redaktionsbuffet(entry)


@given(
    mutated_nested_dict(
        {"collections": [{"properties": {"cm:title": "Redaktionsbuffet"}}]},
        allow_add_key=False,
        allow_add_to_list=False,
        value_blacklist={"Redaktionsbuffet"},
    )
)
def test_kibana_redaktionsbuffet_rejects_from_minimal(entry: Nested_Dict):
    assert not filters.kibana_redaktionsbuffet(entry)


@given(
    mutated_nested_dict(
        {},
        value_blacklist={"Redaktionsbuffet"},
        min_iters=1,
    )
)
def test_kibana_redaktionsbuffet_rejects_stays(entry: Nested_Dict):
    assert not filters.kibana_redaktionsbuffet(entry)
