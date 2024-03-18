from __future__ import annotations

import operator as op
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Optional

import its_data.defaults as defaults
import its_data.fetch as fetch
import its_data.filters as filt
import its_data.transform as trans
import numpy as np
import pandas as pd
from its_data.default_pipelines.data import Data, Target_Data
from its_data.defaults import Fields


def generate_data(
    json_file: Path,
    target_fields: Collection[str],
    uri_label_fields: dict[str, str],
    skos_urls: dict[str, str],
    skip_labels: bool,
    **kwargs,
) -> Data:
    """
    Turn the raw json data into a representation more suitable for classification tasks.

    Currently, this only supports \"flat\" categorical data,
    i.e. non-hierarchical structures.

    :param json_file: The path to the raw json file to process, e.g. from
        :func:`its_data.fetch.fetch`.
    :param target_fields: The data fields that shall be contained.
        Any other data (except text) will be discarded.
    :param skos_urls: Map from data field to the SKOS vocabulary to use
        for looking up human-readable labels for the categories.
    :param uri_label_fields: Map from the data field to the (dot-separated)
        fields to look up in the SKOS vocabulary when looking up
        human-readable labels.
    :param skip_labels: Whether to skip automatic label generation.
    :param kwargs: Additional keyword-arguments to pass onto
        :func:`its_data.fetch._get_basic_df`.
    """
    df = _get_basic_df(
        json_file=json_file,
        target_fields=target_fields,
        **kwargs,
    )

    # get data for targets
    skos_urls = defaults.skos_urls | skos_urls
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
    arr, uris = trans.as_boolean_array(values)

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
