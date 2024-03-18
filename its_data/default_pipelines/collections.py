from collections.abc import Collection, Iterable
from pathlib import Path

import its_data.filters as filt
from its_data.data import Basic_Value, Basic_Value_Not_None
from its_data.default_pipelines import basic
from its_data.default_pipelines.data import Data


def generate_data(
    json_file: Path,
    target_fields: Collection[str],
    skos_urls: dict[str, str] = dict(),
    uri_label_fields: dict[str, str] = dict(),
    skip_labels: bool = False,
    filters: Iterable[filt.Filter] = tuple(),
    **kwargs,
) -> Data:
    filters = list(filters)
    # only include collections, obviously
    filters.append(filt.collections_filter)

    data = basic.generate_data(
        json_file=json_file,
        target_fields=target_fields,
        uri_label_fields=uri_label_fields,
        skos_urls=skos_urls,
        skip_labels=skip_labels,
        filters=filters,
        **kwargs,
    )

    return data
