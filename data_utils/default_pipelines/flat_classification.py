import operator as op
from collections.abc import Collection, Iterable, Sequence
from functools import partial, reduce
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional

import data_utils.default_pipelines.defaults as defaults
import data_utils.fetch as fetch
import data_utils.filters as filt
import data_utils.transform as trans
import numpy as np
import pandas as pd


class Target_Data(NamedTuple):
    arr: np.ndarray[Any, np.dtypes.BoolDType]
    uris: list[str]
    labels: list[str | None]


class Data(NamedTuple):
    raw_texts: np.ndarray[Any, np.dtypes.StrDType]
    ids: np.ndarray[Any, np.dtypes.StrDType]
    target_data: dict[str, Target_Data]
    redaktion_arr: np.ndarray[Any, np.dtypes.BoolDType]


def generate_data(
    json_file: Path,
    target_fields: Collection[str],
    dropped_values: dict[str, Collection[str]] = dict(),
    remapped_values: dict[str, dict[str, str]] = dict(),
    skos_urls: dict[str, str] = dict(),
    filters: Collection[filt.Filter] = tuple(),
    use_defaults: bool = True,
    min_category_support: Optional[int] = None,
    **kwargs,
) -> Data:
    if use_defaults:
        dropped_values = defaults.dropped_values | dropped_values
        remapped_values = defaults.remapped_values | remapped_values
        skos_urls = defaults.skos_urls | skos_urls
        filters = {
            filt.kibana_basic_filter,
            filt.kibana_publicly_visible,
        } | set(filters)

    df = get_basic_df(
        json_file=json_file,
        target_fields=target_fields,
        dropped_values=dropped_values,
        remapped_values=remapped_values,
        filters=filters,
        **kwargs,
    )

    # drop entries with no description or no title
    def empty_str(x: str | None) -> bool:
        return x is None or len(x) == 0

    df: pd.DataFrame = df[
        df.apply(
            lambda x: not empty_str(x["description"]) and not empty_str(x["title"]),
            axis=1,
        )
    ]  # type: ignore

    # drop any entry that contains a language that is not allowed
    # if allowed_languages is not None:
    #     kept_docs = (
    #         df["properties.cclom:general_language"]
    #         .apply(lambda x: not x - frozenset(allowed_languages))
    #         .to_numpy()
    #     )
    #     df = df.iloc[kept_docs]

    # get data for targets
    target_data = {
        target: _values_to_target_data(
            df[target],
            skos_url=skos_urls.get(target, None),
            min_category_support=min_category_support,
        )
        for target in target_fields
    }

    # drop any document that contains no data for any / all target(s)
    # kept_docs: np.ndarray[Any, np.dtypes.BoolDType] = reduce(
    #     np.logical_or if keep_docs_that_contain == "any_target" else np.logical_and,
    #     (data.arr.sum(-1) > 0 for data in target_data.values()),
    # )
    # df = df.iloc[kept_docs]
    # for target in target_fields:
    #     target_data[target] = subset_target_array(
    #         target_data[target],
    #         kept_docs,  # type: ignore
    #     )

    # concatenate title and description
    raw_texts = df.apply(lambda x: x["title"] + "\n" + x["description"], axis=1)
    # determine whether documents belong to the redaktionsbuffet
    redaktion_arr = df["collections"].apply(lambda x: "Redaktionsbuffet" in x)

    return Data(
        raw_texts=raw_texts.to_numpy(),
        ids=df["id"].to_numpy(),
        redaktion_arr=redaktion_arr.to_numpy(dtype=bool),
        target_data=target_data,
    )


def get_basic_df(
    json_file: Path, target_fields: Collection[str], **kwargs
) -> pd.DataFrame:
    df = fetch.df_from_json_file(
        json_file,
        columns={
            "title": "properties.cclom:title",
            "description": "properties.cclom:general_description",
            "id": "nodeRef.id",
            "collections": "collections.properties.cm:title",
            "properties.cclom:general_language": "properties.cclom:general_language",
        }
        | {target: target for target in target_fields},
        **kwargs,
    )

    # transform multi value fields to sets
    for column in ["collections", "properties.cclom:general_language"] + list(
        target_fields
    ):
        df[column] = df[column].fillna("").apply(list).apply(frozenset)

    # unnest description field
    df["description"] = df["description"].apply(
        lambda x: x[0] if x is not None else None
    )

    return df


def _values_to_target_data(
    values: Collection[str],
    skos_url: Optional[str] = None,
    min_category_support: Optional[int] = None,
) -> Target_Data:
    # transform the entries into boolean arrays
    arr, uris = trans.as_boolean_array(values, sort_fn=lambda x: sorted(x))

    # get readable labels for the targets, if available
    if skos_url is not None:
        labels: list[str | None] = fetch.labels_from_skos(
            uris, url=skos_url, multi_value=False
        )  # type: ignore
    else:
        labels = fetch.labels_from_uris(uris=uris, multi_value=False)  # type: ignore

    data = Target_Data(arr, uris, labels)

    # drop all categories with too little support
    if min_category_support:
        data = subset_target_categories(
            data,
            arr.sum(-2) > min_category_support,
        )

    return Target_Data(arr, uris, labels)


def subset_target_array(data: Target_Data, subset_mask: Sequence[bool]) -> Target_Data:
    data = Target_Data(arr=data.arr[subset_mask], uris=data.uris, labels=data.labels)

    return data


def subset_target_categories(
    data: Target_Data, subset_mask: Sequence[bool]
) -> Target_Data:
    subset_indices = np.where(subset_mask)[0]
    data = Target_Data(
        arr=data.arr[:, subset_mask],
        uris=[uri for i, uri in enumerate(data.uris) if i in subset_indices],
        labels=[label for i, label in enumerate(data.labels) if i in subset_indices],
    )

    assert len(data.uris) == len(data.labels) == data.arr.shape[-1]

    return data
