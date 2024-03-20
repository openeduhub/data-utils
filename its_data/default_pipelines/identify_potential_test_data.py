from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional

import its_data.filters as filt
import its_prep.spacy as nlp
import numpy as np
from its_data import defaults, fetch
from its_data.default_pipelines import collections, flat_classification
from its_data.default_pipelines.data import (
    Data,
    Processed_Data,
    subset_categories,
    subset_data_points,
)
from its_data.defaults import Fields
from its_prep import Pipeline_Generator
import its_prep.specs.filters as nlp_filters


def generate_data(
    json_file: Path,
    field: str,
    filters: Iterable[filt.Filter] = tuple(),
    cache_dir: Optional[Path] = None,
    **kwargs,
) -> Processed_Data:
    additional_filters: list[filt.Filter] = list()
    # only keep data that contains at least one target
    additional_filters.append(filt.get_labeled_filter([field]))
    # drop materials with too little text.
    # this is relatively small and will be further filtered later.
    additional_filters.append(
        filt.get_len_filter(
            fields=[Fields.DESCRIPTION.value, Fields.TITLE.value],
            min_lengths=50,
            multi_field_semantics=any,
        )
    )

    data = flat_classification.generate_data(
        json_file=json_file,
        target_fields=[field, Fields.COLLECTIONS_UUID.value, Fields.TOPIC.value],
        filters=set(additional_filters) | set(filters),
        **kwargs,
    )

    # we filter out small sentences here in order to deal with sections such as
    # "further links" or "my equipment", which are especially prevalent on
    # YouTube
    data = Processed_Data.from_data(
        data,
        pipeline_generators=[
            lambda docs, **kwargs: [
                nlp_filters.get_filter_by_subset_len(nlp.into_sentences, min_len=6)
            ]
        ],
        cache_dir=cache_dir,
    )

    doc_lens = np.array([len(doc) for doc in data.processed_texts])
    data = subset_data_points(data, np.where(doc_lens >= 20)[0])

    # get data on collections, which we will use later on
    collections_data = collections.generate_data(json_file, target_fields=[field])

    # drop all collections that do not have a corresponding topic label
    topic_uris = [
        f"http://w3id.org/openeduhub/vocabs/oeh-topics/{id}"
        for id in collections_data.ids
    ]
    topic_labels = fetch.labels_from_skos(
        topic_uris, url=defaults.skos_urls[Fields.TOPIC.value], multi_value=False
    )
    existing_topic_label = [label is not None for label in topic_labels]
    collections_data = subset_data_points(
        collections_data, np.where(existing_topic_label)[0]
    )

    # drop all associated collections that are not included in our new collection
    collections_uris = set(collections_data.ids)
    data = subset_categories(
        data,
        np.where(
            [
                uri in collections_uris
                for uri in data.target_data[Fields.COLLECTIONS_UUID.value].uris
            ]
        )[0],
        Fields.COLLECTIONS_UUID.value,
    )

    # sort the materials by the number of collections they have been assigned
    # to, as being assigned to more collections indicates that the material is
    # of higher quality
    data = subset_data_points(
        data,
        np.flip(
            np.argsort(data.target_data[Fields.COLLECTIONS_UUID.value].arr.sum(-1))
        ),
    )

    # also sort by the number of included words
    data = subset_data_points(
        data,
        np.flip(np.argsort([len(doc) for doc in data.processed_texts], kind="stable")),
    )

    return data
