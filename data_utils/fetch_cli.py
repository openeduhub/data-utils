import argparse
import urllib
import urllib.error
import data_utils.fetch as fetch
from data_utils._version import __version__
from pathlib import Path


def main():
    # define CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url",
        action="store",
        help="The (base) URL from which to download the data dump.",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        action="store",
        help="The name of the file from the URL to be downloaded. It is assumed that this file is accessible through <url/target-file>.",
    )
    parser.add_argument(
        "-u",
        "--username",
        action="store",
        default=None,
        help="The username to use when providing authentication details. Optional unless a password is provided.",
    )
    parser.add_argument(
        "-p",
        "--password",
        action="store",
        default=None,
        help="The password to use when providing authentication details. Optional unless a username is provided.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        action="store",
        default=".",
        help="The path to the output file. If a directory, save the (decompressed) target file to this directory.",
    )
    parser.add_argument(
        "--skip-if-exists", action="store_true", help="Skip files that already exist."
    )
    parser.add_argument(
        "--no-delete-archive",
        action="store_false",
        help="Do not delete the original archive if it was compressed.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )

    # read passed CLI values
    args = parser.parse_args()

    # differentiate between output directory and file
    output_dir = Path(args.output_file)
    output_file = None
    if not output_dir.exists() or output_dir.is_file():
        output_file = output_dir
        output_dir = output_dir.parent

    # download dump
    try:
        output_file = fetch.fetch(
            base_url=args.url,
            target_file=args.input_file,
            output_dir=output_dir,
            output_file=output_file,
            username=args.username,
            password=args.password,
            skip_if_exists=args.skip_if_exists,
            delete_compressed_archive=args.no_delete_archive,
        )
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(
                "The domain you are trying to download data from requires authorization!\nPlease provide a username and password through -u and -p, respectively."
            )

        else:
            raise e

        return

    print(f"Done! Result is located at:\n{output_file}")
