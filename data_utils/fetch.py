import gzip
import json
import shutil
import urllib.request
from collections.abc import Sequence, Iterator
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd


def fetch_data(
    base_url: str,
    target_file: str = "workspace_data-public-only.json.gz",
    username: Optional[str] = None,
    password: Optional[str] = None,
    encoded_auth: Optional[str] = None,
    target_dir: Path = Path("/tmp"),
    skip_if_exists: bool = True,
    delete_compressed_archive: bool = True,
) -> Path:
    """
    Download the latest data dump and save it to the given directory.

    Note: for our production data dump, supplying username and password
      does not work for some reason. Instead, the already correctly encoded
      authentication value (after 'Basic') has to be used.
      See the dev console after having connected to the server for this value.
    """
    path = target_dir / target_file
    if ".gz" == path.__str__()[-3:]:
        final_path = path.with_suffix(path.suffix[:-3])
    else:
        final_path = path

    if skip_if_exists and final_path.exists():
        print(f"File at {final_path} already exists. Skipping...")
        print("Set skip_if_exists to False to force re-download.")
        return final_path

    if "/" == base_url[-1]:
        base_url = base_url[:-1]

    url = "/".join([base_url, target_file])
    headers: Optional[dict[str, str]] = None

    # handle authentication, if supplied
    if username is not None and password is not None:
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, base_url, username, password)
        handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        opener = urllib.request.build_opener(handler)
        urllib.request.install_opener(opener)

    elif encoded_auth is not None:
        headers = {"Authorization": f"Basic {encoded_auth}"}

    print("Downloading data...")
    with urllib.request.urlopen(
        urllib.request.Request(
            url,
            headers=headers,
        )
        if headers is not None
        else url
    ) as r:
        with open(path, "wb") as f:
            shutil.copyfileobj(r, f)

    # if the file was gzipped, decompress it
    if ".gz" == path.__str__()[-3:]:
        print("Decompressing data...")
        with gzip.open(path) as f_in:
            with open(final_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        if delete_compressed_archive:
            path.unlink()

    return final_path


Nested_Dict = Union[dict[str, "Nested_Dict"], Any]


def get_in(nested_dict: Nested_Dict, keys: Sequence[str]) -> Nested_Dict:
    if not keys:
        return nested_dict

    try:
        return get_in(nested_dict[keys[0]], keys[1:])
    except KeyError:
        return None


def raw_json_entry_to_dict(
    raw_entry: Nested_Dict,
    columns: Sequence[str] | dict[str, str],
    key_separator: str = ".",
) -> dict[str, Any]:
    entry: dict[str, Any] = dict()
    if isinstance(columns, Sequence):
        for column in columns:
            entry[column] = get_in(raw_entry, column.split(key_separator))
    else:
        for key, column in columns.items():
            entry[key] = get_in(raw_entry, column.split(key_separator))

    return entry


def json_file_to_dicts(
    path: Path,
    columns: Sequence[str] | dict[str, str],
    key_separator: str = ".",
    prefix: Sequence[str] = ("_source",),
    max_len: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        # because we are dealing with line-separated jsons,
        # read the file one line at a time
        for index, line in enumerate(f):
            if max_len is not None and index >= max_len:
                return

            raw_entry = get_in(json.loads(line), prefix)
            yield (
                raw_json_entry_to_dict(
                    raw_entry, columns=columns, key_separator=key_separator
                )
            )


def json_file_to_pd(
    path: Path,
    columns: Sequence[str] | dict[str, str],
    key_separator: str = ".",
    prefix: Sequence[str] = ("_source",),
    max_len: Optional[int] = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        json_file_to_dicts(
            path, columns, key_separator=key_separator, prefix=prefix, max_len=max_len
        ),
        columns=list(columns.keys() if isinstance(columns, dict) else columns),
    )
