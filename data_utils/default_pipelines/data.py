from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np


class Target_Data(NamedTuple):
    arr: np.ndarray[Any, np.dtypes.BoolDType]
    uris: list[str]
    labels: list[str | None]
    in_test_set: np.ndarray[Any, np.dtypes.BoolDType]


class Data(NamedTuple):
    raw_texts: np.ndarray[Any, np.dtypes.StrDType]
    ids: np.ndarray[Any, np.dtypes.StrDType]
    redaktion_arr: np.ndarray[Any, np.dtypes.BoolDType]
    target_data: dict[str, Target_Data]


def subset_data_points(data: Data, subset_mask: Sequence[bool]) -> Data:
    new_target_data = dict()
    for field, target_data in data.target_data.items():
        target_data = Target_Data(
            arr=target_data.arr[subset_mask],
            uris=target_data.uris,
            labels=target_data.labels,
            in_test_set=target_data.in_test_set[subset_mask],
        )
        new_target_data[field] = target_data

    return Data(
        raw_texts=data.raw_texts,
        ids=data.ids[subset_mask],
        redaktion_arr=data.redaktion_arr[subset_mask],
        target_data=new_target_data,
    )


def subset_categories(data: Data, subset_mask: Sequence[bool], field: str) -> Data:
    subset_indices = np.where(subset_mask)[0]
    new_target_data = data.target_data | {
        field: Target_Data(
            arr=data.target_data[field].arr[:, subset_mask],
            in_test_set=data.target_data[field].in_test_set,
            uris=[
                uri
                for i, uri in enumerate(data.target_data[field].uris)
                if i in subset_indices
            ],
            labels=[
                label
                for i, label in enumerate(data.target_data[field].labels)
                if i in subset_indices
            ],
        )
    }

    return Data(
        raw_texts=data.raw_texts,
        ids=data.ids,
        redaktion_arr=data.redaktion_arr,
        target_data=new_target_data,
    )


def sorted_by_indices(data: Data, indices: np.ndarray) -> Data:
    ids = data.ids[indices]
    raw_texts = data.raw_texts[indices]
    redaktion_arr = data.redaktion_arr[indices]

    new_target_data = dict()
    for target, target_data in data.target_data.items():
        new_target_data[target] = Target_Data(
            arr=target_data.arr[indices],
            in_test_set=target_data.in_test_set[indices],
            uris=target_data.uris,
            labels=target_data.labels,
        )

    return Data(
        raw_texts=raw_texts,
        ids=ids,
        redaktion_arr=redaktion_arr,
        target_data=new_target_data,
    )
