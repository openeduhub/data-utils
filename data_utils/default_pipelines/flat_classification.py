import operator as op
from collections.abc import Collection, Iterable, Sequence
from functools import partial, reduce
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional, TypeVar

import data_utils.fetch as fetch
import data_utils.filters as filters
import data_utils.transform as transform
import data_utils.default_pipelines.defaults as defaults
import numpy as np
import pandas as pd
from data_utils.default_pipelines.utils import fix_entry_multi_value


def get_basic_df(
    path: Path,
    target_fields: Collection[str],
    dropped_values: dict[str, Collection[str]] = dict(),
    merged_values: dict[str, dict[str, str]] = dict(),
    dropped_languages: Collection[str] = tuple(),
    merged_languages: dict[str, str] = dict(),
    username: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    df = fetch.df_from_json_file(
        path=fetch.fetch(
            base_url="https://elasticdump.prod.openeduhub.net",
            target_file="workspace_data-public-only.json",
            output_dir=path,
            username=username,
            password=password,
            skip_if_exists=True,
            delete_compressed_archive=True,
        ),
        columns={
            "title": "properties.cclom:title",
            "description": "properties.cclom:general_description",
            "id": "nodeRef.id",
            "collections": "collections.properties.cm:title",
            "properties.cclom:general_language": "properties.cclom:general_language",
        }
        | {target: target for target in target_fields},
        filters=[filters.kibana_basic_filter, filters.kibana_publicly_visible],
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

    # fix target values
    for target in target_fields:
        df[target] = df[target].apply(
            partial(
                fix_entry_multi_value,
                to_drop=dropped_values.get(target, tuple()),
                to_merge=merged_values.get(target, dict()),
            )
        )

    # fix incorrect language IDs
    df["properties.cclom:general_language"] = df[
        "properties.cclom:general_language"
    ].apply(
        partial(
            fix_entry_multi_value,
            to_drop=dropped_languages,
            to_merge=merged_languages,
        )
    )

    return df


class Target_Data(NamedTuple):
    arr: np.ndarray[Any, np.dtypes.BoolDType]
    uris: list[str]
    labels: list[str | None]


def _get_target_data(
    values: Collection[str], skos_url: Optional[str] = None
) -> Target_Data:
    # transform the entries into boolean arrays
    arr, uris = transform.as_boolean_array(values, sort_fn=lambda x: sorted(x))

    # get readable labels for the targets, if available
    if skos_url is not None:
        labels: list[str | None] = fetch.labels_from_skos(
            uris, url=skos_url, multi_value=False
        )  # type: ignore
    else:
        labels = fetch.labels_from_uris(uris=uris, multi_value=False)  # type: ignore

    data = Target_Data(arr, uris, labels)

    # drop all categories with fewer than five documents
    kept_categories = arr.sum(-2) > 4
    data = subset_target_categories(data, kept_categories)

    assert len(labels) == len(uris) == arr.shape[-1]

    return Target_Data(arr, uris, labels)


def subset_target_array(data: Target_Data, subset_mask: Sequence[bool]) -> Target_Data:
    data = Target_Data(arr=data.arr[subset_mask], uris=data.uris, labels=data.labels)

    assert len(data.uris) == len(data.labels) == data.arr.shape[-1]

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


class Data(NamedTuple):
    raw_texts: np.ndarray[Any, np.dtypes.StrDType]
    ids: np.ndarray[Any, np.dtypes.StrDType]
    target_data: dict[str, Target_Data]
    redaktion_arr: np.ndarray[Any, np.dtypes.BoolDType]


def generate_data(
    path: Path,
    target_fields: Collection[str],
    dropped_targets: dict[str, Collection[str]] = dict(),
    merged_targets: dict[str, dict[str, str]] = dict(),
    dropped_languages: Collection[str] = tuple(),
    merged_languages: dict[str, str] = dict(),
    use_defaults: bool = True,
    skos_urls: dict[str, str] = dict(),
    username: Optional[str] = None,
    password: Optional[str] = None,
    allowed_languages: Optional[Iterable[str]] = None,
    keep_docs_that_contain: Literal["any_target", "all_targets"] = "any_target",
    **kwargs,
) -> Data:
    if use_defaults:
        dropped_targets = defaults.dropped_values | dropped_targets
        merged_targets = defaults.merged_values | merged_targets
        dropped_languages = set(
            defaults.dropped_values["properties.cclom:general_language"]
        ) | set(dropped_languages)
        merged_languages = (
            defaults.merged_values["properties.cclom:general_language_drop_region"]
            | merged_languages
        )
        skos_urls = defaults.skos_urls | skos_urls

    df = get_basic_df(
        path,
        target_fields=target_fields,
        dropped_values=dropped_targets,
        merged_values=merged_targets,
        dropped_languages=dropped_languages,
        merged_languages=merged_languages,
        username=username,
        password=password,
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
    if allowed_languages is not None:
        kept_docs = (
            df["properties.cclom:general_language"]
            .apply(lambda x: not x - frozenset(allowed_languages))
            .to_numpy()
        )
        df = df.iloc[kept_docs]

    # get data for targets
    target_data = {
        target: _get_target_data(df[target], skos_url=skos_urls.get(target, None))
        for target in target_fields
    }

    # drop any document that contains no data for any / all target(s)
    kept_docs: np.ndarray[Any, np.dtypes.BoolDType] = reduce(
        np.logical_or if keep_docs_that_contain == "any_target" else np.logical_and,
        (data.arr.sum(-1) > 0 for data in target_data.values()),
    )
    df = df.iloc[kept_docs]
    for target in target_fields:
        target_data[target] = subset_target_array(
            target_data[target],
            kept_docs,  # type: ignore
        )

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
