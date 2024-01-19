import gzip
import json
import shutil
import urllib.request
from collections.abc import Sequence, Iterator, Iterable
from pathlib import Path
from typing import Any, Optional
import data_utils.filters as filters
from data_utils.utils import get_in, Terminal_Value, Nested_Dict, get_terminal_in

import pandas as pd


def _download(url: str, target_path: Path, headers: Optional[dict[str, str]] = None):
    print("Downloading data...")
    with urllib.request.urlopen(
        urllib.request.Request(
            url,
            headers=headers,
        )
        if headers is not None
        else url
    ) as r:
        with open(target_path, "wb") as f:
            # use shutil.copyfileobj to avoid loading the entire file to RAM
            shutil.copyfileobj(r, f)


def fetch(
    base_url: str,
    target_file: str = "workspace_data-public-only.json.gz",
    target_dir: str | Path = Path("/tmp"),
    username: Optional[str] = None,
    password: Optional[str] = None,
    encoded_auth: Optional[str] = None,
    skip_if_exists: bool = True,
    delete_compressed_archive: bool = True,
) -> Path:
    """Download the latest data dump and save it to the given directory"""
    # convert to pathlib.Path if the path was given as a string
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    # expand the home user directory path, if it was given symbolically
    target_dir = target_dir.expanduser()

    download_path = target_dir / target_file
    if ".gz" == download_path.__str__()[-3:]:
        final_path = download_path.with_suffix(download_path.suffix[:-3])
    else:
        final_path = download_path

    if skip_if_exists and final_path.exists():
        print(f"File at {final_path} already exists.")
        print("Set skip_if_exists to False to force re-download.")
        return final_path

    if skip_if_exists and download_path.exists():
        print(f"File at {download_path} already exists.")
        print("Set skip_if_exists to False to force re-download.")
    else:
        if "/" == base_url[-1]:
            base_url = base_url[:-1]

        url = "/".join([base_url, target_file])
        headers = None

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
        _download(url=url, target_path=download_path, headers=headers)

    # if the file was gzipped, decompress it
    # act on the file on disk to avoid loading the entire file to RAM
    if final_path != download_path:
        print("Decompressing data...")
        with gzip.open(download_path, "rb") as f_in:
            with open(final_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        if delete_compressed_archive:
            download_path.unlink()

    return final_path


def json_entry_to_dict(
    raw_entry: Nested_Dict,
    columns: Iterable[str] | dict[str, str],
    key_separator: str = ".",
) -> dict[str, Terminal_Value]:
    entry: dict[str, Terminal_Value] = dict()
    if isinstance(columns, dict):
        for key, column in columns.items():
            entry[key] = get_terminal_in(raw_entry, column.split(key_separator))
    else:
        for column in columns:
            entry[column] = get_terminal_in(raw_entry, column.split(key_separator))

    return entry


def json_file_to_dicts(
    path: Path,
    columns: Iterable[str] | dict[str, str],
    key_separator: str = ".",
    prefix: str = "_source",
    filters: Iterable[Filter] = tuple(),
    max_len: Optional[int] = None,
    **kwargs,
) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        # because we are dealing with line-separated jsons,
        # read the file one line at a time
        for index, line in enumerate(f):
            if max_len is not None and index >= max_len:
                return

            raw_entry = get_in(json.loads(line), prefix.split(key_separator))
            if not isinstance(raw_entry, dict):
                raise ValueError(
                    f"The given prefix key {prefix} does not exists for all entries or does not point to a map!"
                )

            if all(fun(raw_entry) for fun in filters):
                yield json_entry_to_dict(
                    raw_entry, columns=columns, key_separator=key_separator, **kwargs
                )


def json_file_to_df(
    path: Path,
    columns: Sequence[str] | dict[str, str],
    key_separator: str = ".",
    prefix: str = "_source",
    filters: Iterable[filters.Filter] = tuple(),
    max_len: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    return pd.DataFrame(
        json_file_to_dicts(
            path.expanduser(),
            columns,
            key_separator=key_separator,
            prefix=prefix,
            filters=filters,
            max_len=max_len,
            **kwargs,
        ),
        columns=list(columns.keys() if isinstance(columns, dict) else columns),
    )
