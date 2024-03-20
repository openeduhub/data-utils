import numpy as np
from its_prep.types import Document, Filter


def get_repetition_filter(min_rep_count: int = 3, post_filter_count: int = 1) -> Filter:
    """
    Filter out repeated tokens (e.g. "abc abc abc" -> "abc")
    """

    if min_rep_count < 2:
        raise ValueError(f"min_rep_count must be at least 2! Was: {min_rep_count}")

    if post_filter_count > min_rep_count:
        raise ValueError("post_filter_count must not be larger than min_rep_count!")

    def filter_fun(doc: Document) -> Document:
        tokens_arr = np.array(doc.selected_tokens)
        # this array contains information about whether the token at index i is
        # identical to the token at index i+1
        identical_arr = tokens_arr[:-1] == tokens_arr[1:]

        # because i == i+1 and i+1 == i+2 => i == i+2, we now just need to find
        # all sequences of contiguious True's.
        # find the start- and end-points of sequences of True's
        ranges = np.where(np.diff(identical_arr, prepend=False, append=False))[0]
        # reshape into 2-tuples
        ranges = ranges.reshape(ranges.shape[0] // 2, 2)

        # discard sequences that are too short
        range_lens = ranges[:, 1] - ranges[:, 0] + 1
        ranges = ranges[range_lens >= min_rep_count]

        # drop all tokens of the sequences except for the first
        # post_filter_count
        keep = np.ones_like(tokens_arr, dtype=bool)
        for range_spec in ranges:
            keep[range_spec[0] + post_filter_count : range_spec[1] + 1] = False

        return doc.sub_doc(set(np.where(keep)[0]))

    return filter_fun
