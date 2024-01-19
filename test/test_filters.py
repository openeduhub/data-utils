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
        add_blacklist=["ccm:collection_io_reference"],
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
            "aspects": [],
        },
        add_whitelist=["ccm:collection_io_reference"],
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
        allow_add_to_list=True,
        allow_remove_from_list=False,
    )
)
def test_kibana_basic_filter_rejects_with_bad_aspect(entry: Nested_Dict):
    assert not filters.kibana_basic_filter(entry)
