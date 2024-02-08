from __future__ import annotations

import operator as op
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Optional

import data_utils.defaults as defaults
import data_utils.fetch as fetch
import data_utils.filters as filt
import data_utils.transform as trans
import numpy as np
import pandas as pd
from data_utils.data import Basic_Value, Basic_Value_Not_None
from data_utils.default_pipelines.data import Data, Target_Data
from data_utils.defaults import Fields


def generate_data(
    json_file: Path,
    target_fields: Collection[str],
    dropped_values: dict[str, Collection[Basic_Value_Not_None]] = dict(),
    remapped_values: dict[str, dict[Basic_Value_Not_None, Basic_Value]] = dict(),
    skos_urls: dict[str, str] = dict(),
    uri_label_fields: dict[str, str] = dict(),
    filters: Collection[filt.Filter] = tuple(),
    use_defaults: bool = True,
    skip_labels: bool = False,
    **kwargs,
) -> Data:
    """
    Turn the raw json data into a representation more suitable for classification tasks.

    Currently, this only supports \"flat\" categorical data,
    i.e. non-hierarchical structures.

    :param json_file: The path to the raw json file to process, e.g. from
        :func:`data_utils.fetch.fetch`.
    :param target_fields: The data fields that shall be contained.
        Any other data (except text) will be discarded.
    :param dropped_values: Map from data field to the categories that
        shall be dropped.
    :param remapped_values: Map from data field to the categories that
        shall be renamed to different ones.
    :param skos_urls: Map from data field to the SKOS vocabulary to use
        for looking up human-readable labels for the categories.
    :param uri_label_fields: Map from the data field to the (dot-separated)
        fields to look up in the SKOS vocabulary when looking up
        human-readable labels.
    :param filters: Additional filters to apply to drop data during importing.
    :param use_defaults: Whether to apply defaults (:ref:`data_utils.defaults`).
        If defaults and arguments given above conflict, the given arguments
        will be preferred.
    :param skip_labels: Whether to skip automatic label generation.
    :param kwargs: Additional keyword-arguments to pass onto
        :func:`data_utils.fetch.df_from_json_file`.
    """
    if use_defaults:
        dropped_values = defaults.dropped_values | dropped_values
        remapped_values = defaults.remapped_values | remapped_values
        skos_urls = defaults.skos_urls | skos_urls
        filters = {
            filt.kibana_publicly_visible,
            filt.kibana_basic_filter,
            filt.existing_text_filter,
        } | set(filters)

    df = _get_basic_df(
        json_file=json_file,
        target_fields=target_fields,
        dropped_values=dropped_values,
        remapped_values=remapped_values,
        filters=filters,
        **kwargs,
    )

    # get data for targets
    target_data = {
        target: _values_to_target_data(
            df,
            field=target,
            skos_url=skos_urls.get(target, None),
            uri_label_field=uri_label_fields.get(target, ("prefLabel", "de")),
            skip_labels=skip_labels,
        )
        for target in target_fields
    }

    # concatenate title and description
    raw_texts = df.apply(
        lambda x: (x[Fields.TITLE.value] or "")
        + "\n"
        + (x[Fields.DESCRIPTION.value] or ""),
        axis=1,
    )

    # determine whether documents belong to the redaktionsbuffet
    redaktion_series = df[Fields.COLLECTIONS_TITLE.value].apply(
        lambda x: x is not None and "Redaktionsbuffet" in x
    )

    return Data(
        raw_texts=raw_texts.to_numpy(),
        ids=df[Fields.ID.value].to_numpy(),
        editor_arr=redaktion_series.to_numpy(dtype=bool),
        target_data=target_data,
    )


def _get_basic_df(
    json_file: Path, target_fields: Collection[str], **kwargs
) -> pd.DataFrame:
    df = fetch.df_from_json_file(
        json_file,
        columns={
            Fields.TITLE.value,
            Fields.DESCRIPTION.value,
            Fields.ID.value,
            Fields.COLLECTIONS_TITLE.value,
            Fields.LANGUAGE.value,
            Fields.TEST_DATA.value,
        }
        | set(target_fields),
        **kwargs,
    )

    # unnest description field
    df[Fields.DESCRIPTION.value] = df[Fields.DESCRIPTION.value].apply(
        lambda x: x[0] if x is not None else None
    )

    return df


def _values_to_target_data(
    df: pd.DataFrame,
    field: str,
    skos_url: Optional[str],
    uri_label_field: Sequence[str],
    skip_labels: bool,
) -> Target_Data:
    values = df[field]

    # transform the entries into boolean arrays
    arr, uris = trans.as_boolean_array(values, sort_fn=lambda x: sorted(x))

    # get readable labels for the targets, if available
    if not skip_labels:
        if skos_url is not None:
            labels: list[str | None] = fetch.labels_from_skos(
                uris, url=skos_url, label_seq=uri_label_field, multi_value=False
            )  # type: ignore
        else:
            labels = fetch.labels_from_uris(
                uris=uris, label_seq=uri_label_field, multi_value=False
            )  # type: ignore
    else:
        labels = [None for _ in uris]

    return Target_Data(
        arr=arr,
        uris=np.array(uris),
        labels=np.array(labels),
        in_test_set=np.array(
            [
                test_ids is not None and field in test_ids
                for test_ids in df[Fields.TEST_DATA.value]
            ]
        ),
    )
