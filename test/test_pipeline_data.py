import operator as op

from nlprep import Iterable
import test.strategies as myst
from typing import Optional

import hypothesis.strategies as st
import numpy as np
from data_utils.default_pipelines.data import (
    BoW_Data,
    Data,
    Processed_Data,
    Target_Data,
    subset_categories,
    subset_data_points,
)
from hypothesis import given

MAX_DATA_LEN = 10


@st.composite
def target_data_st(
    draw: st.DrawFn, n: Optional[int] = None, m: Optional[int] = None
) -> Target_Data:
    if n is None:
        n = draw(st.integers(min_value=1, max_value=MAX_DATA_LEN))
    if m is None:
        m = draw(st.integers(min_value=1, max_value=MAX_DATA_LEN))

    arr = draw(
        st.lists(
            st.lists(st.booleans(), min_size=m, max_size=m), min_size=n, max_size=n
        )
    )
    in_test_set = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    uris = draw(st.lists(st.text(), min_size=m, max_size=m))
    labels = draw(st.lists(st.text(), min_size=m, max_size=m))

    return Target_Data(
        arr=np.array(arr),
        in_test_set=np.array(in_test_set),
        uris=np.array(uris),
        labels=np.array(labels),
    )


@st.composite
def data_st(draw: st.DrawFn, n: Optional[int] = None) -> Data:
    if n is None:
        n = draw(st.integers(min_value=1, max_value=MAX_DATA_LEN))

    raw_texts = draw(st.lists(st.text(), min_size=n, max_size=n))
    ids = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    redaktion_arr = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    target_data = draw(st.dictionaries(st.text(), target_data_st(n=n), max_size=3))

    return Data(
        raw_texts=np.array(raw_texts),
        ids=np.array(ids),
        redaktion_arr=np.array(redaktion_arr),
        target_data=target_data,
    )


@st.composite
def processed_data_st(draw: st.DrawFn, n: Optional[int] = None) -> Processed_Data:
    data = draw(data_st(n=n))
    return Processed_Data.from_data(data)


@st.composite
def bow_data_st(draw: st.DrawFn, n: Optional[int] = None) -> BoW_Data:
    data = draw(data_st(n=n))
    return BoW_Data.from_data(data)


def assert_subset_correct(new, old, indices):
    assert len(new) == len(indices)
    # new is correctly sorted
    identical = (
        old[old_index] == new[new_index] for new_index, old_index in enumerate(indices)
    )
    # in case of arrays, assert that all values are identical
    assert all(all(x) if isinstance(x, Iterable) else x for x in identical)


@given(st.data(), st.one_of(data_st(), processed_data_st(), bow_data_st()))
def test_subset_data_points(data: st.DataObject, data_base: Data):
    if len(data_base.ids) == 0:
        indices = []
    else:
        indices = data.draw(
            st.lists(
                st.integers(min_value=0, max_value=len(data_base.ids) - 1), unique=True
            )
        )

    data_subset = subset_data_points(data_base, indices)

    for field in ["raw_texts", "ids", "redaktion_arr", "processed_texts", "bows"]:
        if not hasattr(data_base, field):
            continue

        assert_subset_correct(
            getattr(data_subset, field), getattr(data_base, field), indices
        )

    for target_data_new, target_data_old in zip(
        data_subset.target_data.values(), data_base.target_data.values()
    ):
        for field in ["arr", "in_test_set"]:
            assert_subset_correct(
                getattr(target_data_new, field),
                getattr(target_data_old, field),
                indices,
            )

        # unchanged data
        for field in ["uris", "labels"]:
            new_value = getattr(target_data_new, field)
            old_value = getattr(target_data_old, field)
            assert new_value is old_value or new_value == old_value


@given(st.data(), st.one_of(data_st(), processed_data_st(), bow_data_st()))
def test_subset_categories(data: st.DataObject, data_base: Data):
    if len(data_base.target_data) == 0:
        return
    chosen_field = data.draw(st.sampled_from(list(data_base.target_data.keys())))
    if len(data_base.target_data[chosen_field].uris) == 0:
        indices = []
    else:
        indices = data.draw(
            st.lists(
                st.integers(
                    min_value=0,
                    max_value=len(data_base.target_data[chosen_field].uris) - 1,
                ),
                unique=True,
            )
        )

    data_subset = subset_categories(data_base, indices=indices, field=chosen_field)

    # unchanged
    for field in ["raw_texts", "ids", "redaktion_arr", "processed_texts", "bows"]:
        if not hasattr(data_base, field):
            continue

        new_value = getattr(data_base, field)
        old_value = getattr(data_subset, field)
        assert new_value is old_value or new_value == old_value

    for key in set(data_base.target_data.keys()) | set(data_subset.target_data.keys()):
        target_data_new = data_subset.target_data[key]
        target_data_old = data_base.target_data[key]

        # assert that the categories were adjusted correctly
        if key == chosen_field:
            assert_subset_correct(
                target_data_new.labels, target_data_old.labels, indices
            )
            assert_subset_correct(target_data_new.uris, target_data_old.uris, indices)
            assert target_data_new.arr.shape[-1] == len(indices)
            assert all(
                (
                    target_data_new.arr[:, new_index]
                    == target_data_old.arr[:, old_index]
                ).all()
                for new_index, old_index in enumerate(indices)
            )

        # unchanged
        else:
            assert (
                target_data_new is target_data_old or target_data_new == target_data_old
            )


@given(data_st())
def test_processed_data(data: Data):
    processed_data = Processed_Data.from_data(data)

    assert "processed_texts" in processed_data._1d_data_fields

    # unchanged
    for field in ["raw_texts", "ids", "redaktion_arr", "target_data"]:
        new, old = getattr(processed_data, field), getattr(data, field)
        identical = new == old
        if isinstance(identical, Iterable):
            assert all(identical)
        else:
            assert identical


@given(data_st())
def test_bow_data(data: Data):
    processed_data = Processed_Data.from_data(data)
    bow_data = BoW_Data.from_processed_data(processed_data)

    assert "processed_texts" in bow_data._1d_data_fields
    assert "bows" in bow_data._1d_data_fields

    # unchanged
    for field in [
        "raw_texts",
        "ids",
        "redaktion_arr",
        "target_data",
        "processed_texts",
    ]:
        new, old = getattr(bow_data, field), getattr(processed_data, field)
        identical = new == old
        if isinstance(identical, Iterable):
            assert all(identical)
        else:
            assert identical

    # check document lengths
    lens_old = [len(doc) for doc in processed_data.processed_texts]
    lens_new = bow_data.bows.sum(-1).tolist()

    assert len(lens_old) == len(lens_new)
    assert all(old == new for old, new in zip(lens_old, lens_new))

    # check ids
    for bow, doc in zip(bow_data.bows, bow_data.processed_texts):
        doc_set = set(doc)
        bow_word_ids = np.where(bow > 0)[0]
        assert all(bow_data.id_to_word[id] in doc_set for id in bow_word_ids)


def test_internal_sets():
    data = Data(
        raw_texts=np.array(["foo"]),
        ids=np.array(["bar"]),
        redaktion_arr=np.array([True]),
        target_data=dict(),
    )

    assert "processed_texts" not in data._1d_data_fields
    assert "bows" not in data._1d_data_fields

    processed_data = Processed_Data.from_data(data)

    assert "processed_texts" not in data._1d_data_fields
    assert "bows" not in data._1d_data_fields
    assert "processed_texts" in processed_data._1d_data_fields
    assert "bows" not in processed_data._1d_data_fields

    bow_data = BoW_Data.from_processed_data(processed_data)

    assert "processed_texts" not in data._1d_data_fields
    assert "bows" not in data._1d_data_fields
    assert "processed_texts" in processed_data._1d_data_fields
    assert "bows" not in processed_data._1d_data_fields
    assert "processed_texts" in bow_data._1d_data_fields
    assert "bows" in bow_data._1d_data_fields
