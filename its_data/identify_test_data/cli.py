import argparse
import csv
import math
import pickle
from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd
import questionary
from its_data import fetch
from its_data._version import __version__
from its_data.data import get_leaves
from its_data.default_pipelines import identify_potential_test_data
from its_data.default_pipelines.data import Data
from tqdm import tqdm


def main() -> None:
    # define CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_file",
        action="store",
        help="The path to the line-separated json file storing the data. If you are missing this file, please run the download-data script.",
    )
    parser.add_argument(
        "--initial-parse-rate",
        action="store",
        type=float,
        default=0.01,
        help="The relative amount of data points to scan through in order to find possible metadata fields.",
    )
    parser.add_argument(
        "--cache-dir",
        action="store",
        type=str,
        default=None,
        help="The directory to use for caching.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # read passed CLI values
    args = parser.parse_args()
    json_file = Path(args.json_file)

    cache_dir = (
        Path.cwd() / ".cache" if args.cache_dir is None else Path(args.cache_dir)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    # prompt the user on which metadata field to generate test data for
    try:
        with open(cache_dir / "fields.pkl", "rb") as f:
            fields = pickle.load(f)
    except FileNotFoundError:
        print("Finding (most) metadata fields...")
        fields = _get_metadata_fields(json_file, rate=args.initial_parse_rate)
        fields = sorted(fields, key=lambda x: len(".".join(x)))

        with open(cache_dir / "fields.pkl", "wb+") as f:
            pickle.dump(fields, f)

    chosen_field: str = questionary.autocomplete(
        message="Please select a metadata field to generate test-data for:",
        choices=[".".join(field) for field in fields],
    ).ask()

    # generate potential test data
    print(f"Finding potential candidates for {chosen_field}...")
    data = identify_potential_test_data.generate_data(
        json_file, chosen_field, cache_dir=cache_dir
    )
    print(f"Total number of potential candidates: {len(data.ids)}")
    print()
    totals = data.target_data[chosen_field].arr.sum(-2)
    print(f"Distribution of potential candidates:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(
            pd.DataFrame(
                {"count": totals, "name": data.target_data[chosen_field].labels}
            ).set_index("name")
        )

    # ask for wanted number of hits per category
    while True:
        try:
            max_hits = int(
                questionary.text(
                    message="Please specify the target number of materials per category:",
                ).ask()
            )
            if max_hits < 1:
                raise ValueError()

            break

        except ValueError:
            print("Please provide a positive integer")

    try:
        with open(cache_dir / "visited.csv", "r") as f:
            visited: set[str] = {line[0] for line in csv.reader(f)}
    except FileNotFoundError:
        visited = set()

    try:
        with open(Path.cwd() / "accepted.csv", "r") as f:
            accepted: set[str] = {line[0] for line in csv.reader(f)}
    except FileNotFoundError:
        accepted = set()

    next_candidate_fun = lambda: next_potential_candidate(
        data,
        visited=visited,
        accepted=accepted,
        field=chosen_field,
        max_hits=max_hits,
    )

    # continue presenting candidates as long as there are any
    i = next_candidate_fun()
    try:
        while i is not None:
            target_data = data.target_data[chosen_field]
            print("".join("-" for _ in range(79)))
            print("Processed text:")
            print(data.processed_texts[i])
            print("Text:")
            print(data.raw_texts[i])
            print("URIs of assignments:")
            print(target_data.uris[target_data.arr[i]])
            print("Labels of assignments:")
            print(target_data.labels[target_data.arr[i]])

            accept = questionary.confirm("Accept this material?").unsafe_ask()
            visited.add(data.ids[i])
            if accept:
                accepted.add(data.ids[i])

            i = next_candidate_fun()
    except KeyboardInterrupt:
        pass
    finally:
        with open(Path.cwd() / "accepted.csv", "w") as f:
            csv.writer(f).writerows([[uuid] for uuid in accepted])
        with open(cache_dir / "visited.csv", "w") as f:
            csv.writer(f).writerows([[uuid] for uuid in visited])


def next_potential_candidate(
    data: Data,
    visited: Collection[str],
    accepted: Collection[str],
    field: str,
    max_hits: int,
) -> int | None:
    visited_index = np.where([id in visited for id in data.ids])[0]
    accepted_index = np.where([id in accepted for id in data.ids])[0]

    if len(accepted_index) == 0:
        current_counts = np.zeros_like(data.target_data[field].uris, dtype=int)
    else:
        current_counts = data.target_data[field].arr[accepted_index].sum(-2)

    print("Current number of accepted_index materials per category:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(
            pd.DataFrame(
                {"category": data.target_data[field].labels, "count": current_counts}
            )
        )

    categories_to_try = np.argsort(current_counts, kind="stable")

    for category_id in categories_to_try:
        # we have already found enough test data for this category
        if current_counts[category_id] > max_hits:
            continue

        relevant = np.where(data.target_data[field].arr[:, category_id])[0]
        relevant = np.setdiff1d(relevant, visited_index, assume_unique=True)
        if len(relevant) == 0:
            continue

        return relevant[0]

    return None


def _get_metadata_fields(
    json_file: Path,
    rate: float,
    key_separator: str = ".",
    prefix: str = "_source",
) -> set[tuple[str, ...]]:
    n = fetch.num_entries(json_file)

    rate_inv = math.ceil(1 / rate)

    fields: set[tuple[str, ...]] = set()
    for index, entry in enumerate(
        tqdm(
            fetch.raw_entry_generator(
                json_file, key_separator=key_separator, prefix=prefix, max_len=None
            ),
            total=n,
        )
    ):
        if index % rate_inv == 0:
            fields |= get_leaves(entry)

    return fields


if __name__ == "__main__":
    main()
