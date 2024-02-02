from collections.abc import Collection
from functools import reduce
from pathlib import Path
from typing import Optional

import data_utils.default_pipelines.flat_classification as base_pipeline
import data_utils.filters as filters
import numpy as np
from data_utils.default_pipelines.data import (
    BoW_Data,
    Processed_Data,
    subset_categories,
    subset_data_points,
)
from nlprep import pipelines


def generate_data(
    json_file: Path,
    target_fields: Collection[str],
    cache_dir: Optional[Path] = None,
    **kwargs,
) -> BoW_Data:
    print("Reading data...")
    # only keep data that contains at least one target
    labeled_filter = filters.get_labeled_filter(
        target_fields, multi_field_semantics=any
    )
    # only keep German data
    german_filter = filters.get_language_filter(["de"])
    # drop materials with too little text
    text_len_filter = filters.get_len_filter(
        fields=[filters.Fields.DESCRIPTION.value, filters.Fields.TITLE.value],
        min_lengths=30,
        multi_field_semantics=any,
    )

    base_data = base_pipeline.generate_data(
        json_file=json_file,
        target_fields=target_fields,
        filters=[labeled_filter, german_filter, text_len_filter],
        **kwargs,
    )

    # drop all target categories with fewer than 10 associated materials
    for target in target_fields:
        support = base_data.target_data[target].arr.sum(-2)
        base_data = subset_categories(
            base_data,
            indices=np.where(support >= 10)[0],  # type: ignore
            field=target,
        )

    # drop all data that now no longer has any targets
    to_keep = np.ones_like(base_data.ids, dtype=bool)
    for target in target_fields:
        to_keep = np.logical_and(to_keep, base_data.target_data[target].arr.sum(-1) > 0)

    base_data = subset_data_points(base_data, np.where(to_keep)[0])

    # pre-process texts
    print("Pre-processing texts...")
    processed_data = Processed_Data.from_data(
        base_data,
        pipeline_generators=pipelines.get_poc_topic_modeling_pipelines(
            ignored_upos_tags={
                "PUNCT",
                "SPACE",
                "X",
                "SCONJ",
                "PRON",
                "PART",
                "INTJ",
                "DET",
                "CCONJ",
                "AUX",
                "ADP",
            },
            required_df_interval={
                "min_num": 10,
                "max_rate": 0.25,
                "interval_open": False,
                "count_only_selected": True,
            },
        ),
        cache_dir=cache_dir,
    )
    del base_data  # now redundant

    # calculate bag of words
    print("Transforming into bag of words...")
    bow_data = BoW_Data.from_processed_data(processed_data)
    del processed_data  # now redundant

    # drop all tokens that make up more than 0.5% of all tokens
    keep_tokens = (bow_data.bows.sum(-2) / bow_data.bows.sum()) < 0.005
    bow_data = subset_categories(bow_data, np.where(keep_tokens)[0], field="bows")

    # extremely long tokens are usually invalid, so drop them
    keep_tokens = np.array([len(word) <= 30 for word in bow_data.words])
    bow_data = subset_categories(bow_data, np.where(keep_tokens)[0], field="bows")

    # ensure that all docs have at least ten tokens,
    # each token is in at least five docs,
    # and each category has at least ten documents
    print("Dropping values and categories that do not fit quality criteria...")
    lens = bow_data.bows.sum(-1)
    support = (bow_data.bows > 0).sum(-2)
    target_supports = {
        target: target_data.arr.sum(-2)
        for target, target_data in bow_data.target_data.items()
    }
    while (
        lens.min() < 10
        or support.min() < 5
        or any(target_support.min() < 10 for target_support in target_supports.values())
    ):
        print(f"Shape of bag of words: {bow_data.bows.shape}")
        # drop tokens that have too small support
        bow_data = subset_categories(bow_data, np.where(support >= 5)[0], field="bows")

        # drop docs that are too small
        lens = bow_data.bows.sum(-1)
        bow_data = subset_data_points(bow_data, np.where(lens >= 10)[0])

        # drop disciplines that have too low support
        target_supports = {
            target: target_data.arr.sum(-2)
            for target, target_data in bow_data.target_data.items()
        }
        for target, target_support in target_supports.items():
            bow_data = subset_categories(
                bow_data, np.where(target_support >= 10)[0], field=target
            )

        # drop all documents that have no categories assigned
        to_keep = reduce(
            np.logical_or,
            (
                target_data.arr.sum(1) > 0
                for target_data in bow_data.target_data.values()
            ),
            np.zeros_like(bow_data.ids, dtype=bool),
        )
        bow_data = subset_data_points(bow_data, np.where(to_keep)[0])

        # re-calculate all metrics
        lens = bow_data.bows.sum(-1)
        support = (bow_data.bows > 0).sum(-2)
        target_supports = {
            target: target_data.arr.sum(-2)
            for target, target_data in bow_data.target_data.items()
        }

    return bow_data
