from __future__ import annotations

import operator as op
from collections.abc import Collection
from pathlib import Path

import its_data.defaults as defaults
import its_data.filters as filt
from its_data.data import Basic_Value, Basic_Value_Not_None
from its_data.default_pipelines import basic
from its_data.default_pipelines.data import Data


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
    Turn the raw json data into a representation more suitable for
    classification tasks.

    Currently, this only supports \"flat\" categorical data,
    i.e. non-hierarchical structures.

    :param json_file: The path to the raw json file to process, e.g. from
        :func:`its_data.fetch.fetch`.
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
    :param use_defaults: Whether to apply defaults (:ref:`its_data.defaults`).
        If defaults and arguments given above conflict, the given arguments
        will be preferred.
    :param skip_labels: Whether to skip automatic label generation.
    :param kwargs: Additional keyword-arguments to pass onto
        :func:`its_data.fetch.df_from_json_file`.
    """
    if use_defaults:
        dropped_values = defaults.dropped_values | dropped_values
        remapped_values = defaults.remapped_values | remapped_values
        filters = {
            filt.kibana_publicly_visible,
            filt.kibana_basic_filter,
            filt.existing_text_filter,
        } | set(filters)

    return basic.generate_data(
        json_file=json_file,
        target_fields=target_fields,
        uri_label_fields=uri_label_fields,
        skos_urls=skos_urls,
        skip_labels=skip_labels,
        filters=filters,
        dropped_values=dropped_values,
        remapped_values=remapped_values,
        **kwargs,
    )
