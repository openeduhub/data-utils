import argparse
from pathlib import Path

from data_utils._version import __version__
from data_utils.default_pipelines import its_jointprobability
from data_utils.default_pipelines.data import balanced_split, publish
from data_utils.defaults import Fields

PIPELINES = {"its-jointprobability": its_jointprobability.generate_data}


def main():
    # define CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_file",
        help="The path to the json file fetched using the download utility",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        choices=PIPELINES.keys(),
        help="The pipeline to run",
    )
    parser.add_argument(
        "--targets",
        "-t",
        nargs="+",
        # type=Fields,
        choices=[i.name for i in Fields],
        help="The metadata fields to export",
    )
    parser.add_argument(
        "--split",
        # type=Fields,
        choices=[i.name for i in Fields],
        default=None,
        help="The metadata field to balance the split in train / test around. Defaults to the first given target field",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="The path to the cache directory to use. Defaults to the parent of the json file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="The path to file in which to store the results. Defaults to the parent of the json file",
    )
    parser.add_argument(
        "--name",
        default="",
        help="An additional identifier to give to the output files",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # read CLI argument values
    args = parser.parse_args()
    json_file = Path(args.json_file)
    if args.cache_dir is None:
        cache_dir = json_file.parent
    else:
        cache_dir = Path(args.cache_dir)
    if args.output_dir is None:
        output_dir = json_file.parent
    else:
        output_dir = Path(args.output_dir)

    # ensure that the cache & output directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_fields = [Fields[field].value for field in args.targets]
    data = PIPELINES[args.pipeline](
        json_file,
        target_fields=target_fields,
        cache_dir=cache_dir,
    )

    split_field = target_fields[0]
    train, test = balanced_split(data, split_field)

    publish(train, output_dir, name=f"{args.name}_train")
    publish(test, output_dir, name=f"{args.name}_test")


if __name__ == "__name__":
    main()
