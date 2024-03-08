from __future__ import annotations
import csv
from math import dist
import ast

import operator as op
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, Optional, TypeVar, overload

import numpy as np
import pandas as pd
from nlprep import Pipeline_Generator, apply_filters, tokenize_documents
from numpy.typing import NDArray


@dataclass(frozen=True)
class _Base_Data:
    """
    The basic information contained within all data objects.

    This in intended entirely for internal use within the subset functions,
    so that they can detect which fields to act on and in what way.
    """

    _nested_data_fields: frozenset[str] = field(init=False, repr=False)
    _category_fields: frozenset[str] = field(init=False, repr=False)
    _1d_data_fields: frozenset[str] = field(init=False, repr=False)
    _2d_data_fields: frozenset[str] = field(init=False, repr=False)


@dataclass(frozen=True)
class Target_Data(_Base_Data):
    """
    Data for one particular field, including assignments and metadata.

    :param arr: The assignments to the field at hand.

        - For categorical fields, this is a Boolean matrix.
        - For bag-of-words data, this is an Integer matrix.
    :param in_test_set: Boolean array indicating whether each data point
        belongs to the test data set for this field.
    :param uris: String array indicating the URI of each category
        (i.e. the right-most dimension of the assignment array).
    :param labels: String array indicating the (human-readable) label
        of each category.
    """

    _nested_data_fields = frozenset()
    _category_fields = frozenset({"uris", "labels"})
    _1d_data_fields = frozenset({"in_test_set"})
    _2d_data_fields = frozenset({"arr"})

    #:The assignments to the field at hand.
    #:
    #: - For categorical fields, this is a Boolean matrix.
    #: - For bag-of-words data, this is an Integer matrix.
    arr: np.ndarray
    #: Boolean array indicating whether each data point
    #: belongs to the test data set for this field.
    in_test_set: np.ndarray[Any, np.dtypes.BoolDType]
    #: String array indicating the URI of each category
    #: (i.e. the right-most dimension of the assignment array).
    uris: np.ndarray[Any, np.dtypes.StrDType]
    #: String array indicating the (human-readable) label
    #: of each category.
    labels: np.ndarray[Any, np.dtypes.StrDType]


@dataclass(frozen=True)
class Data(_Base_Data):
    """
    Data on an entire text corpus with multiple possible metadata fields.

    :param raw_texts: The unprocessed texts contained within each document.
    :param ids: The unique IDs of each document.
    :param editor_arr: A Boolean array indicating whether each document
        is editorially confirmed.
    :param target_data: Map from field names to their data.
    """

    #: The unprocessed texts contained within each document.
    raw_texts: np.ndarray[Any, np.dtypes.StrDType]
    #: The unique IDs of each document.
    ids: np.ndarray[Any, np.dtypes.StrDType]
    #: A Boolean array indicating whether each document
    #: is editorially confirmed.
    editor_arr: np.ndarray[Any, np.dtypes.BoolDType]
    #: Map from field names to their data.
    target_data: dict[str, Target_Data]

    _nested_data_fields = frozenset({"target_data"})
    _category_fields = frozenset()
    _1d_data_fields = frozenset({"raw_texts", "ids", "editor_arr"})
    _2d_data_fields = frozenset()


@dataclass(frozen=True)
class Processed_Data(Data):
    """
    A text corpus where each document's text has been tokenized and pre-processed.

    :param processed_texts: The tokenized representations of the texts.
    """

    #: The language of each text.
    languages: np.ndarray[Any, np.dtypes.StrDType]
    #: The tokenized representations of the texts.
    processed_texts: list[tuple[str, ...]] = field(default_factory=list)

    _nested_data_fields = Data._nested_data_fields
    _category_fields = Data._category_fields
    _1d_data_fields = Data._1d_data_fields | {"processed_texts"}
    _2d_data_fields = Data._2d_data_fields

    @classmethod
    def from_data(
        cls,
        data: Data,
        pipeline_generators: Iterable[Pipeline_Generator] = tuple(),
        cache_dir: Optional[Path] = None,
    ) -> "Processed_Data":
        """
        Turn an unprocessed corpus into a processed one.

        This utilizes the nlprep library for tokenization and pre-processing.
        Note that this may take a long time for decently large corpuses.

        :param data: The text corpus to pre-process.
        :param pipeline_generators: Pipelines to iteratively apply to the
            corpus, in order to filter out unwanted tokens.
            See the nlprep library for more details.
        :param cache_dir: The path to the directory that shall contain a cache
            of all the pre-processed texts. Useful when identical texts may be
            processed multiple times.
        """
        import nlprep.spacy as nlp

        try:
            nlp.utils.load_caches(
                cache_dir,  # type: ignore
                file_prefix="nlp_cache",
            )
        except (FileNotFoundError, TypeError):
            pass

        docs = list(tokenize_documents(data.raw_texts.tolist(), nlp.tokenize_as_lemmas))

        # apply the given pipelines one by one
        for pipeline_generator in pipeline_generators:
            pipeline = pipeline_generator(docs)
            docs = list(apply_filters(docs, pipeline))

        if cache_dir is not None:
            nlp.utils.save_caches(directory=cache_dir, file_prefix="nlp_cache")

        return cls(
            raw_texts=data.raw_texts,
            ids=data.ids,
            editor_arr=data.editor_arr,
            target_data=data.target_data,
            processed_texts=[doc.selected_tokens for doc in docs],
            languages=np.array([doc.language for doc in docs]),
        )


@dataclass(frozen=True)
class BoW_Data(Processed_Data):
    """
    A tokenized corpus where each document has additionally been turned into
    its bag-of-words representation.
    """

    _virtual_bow_dict: dict[str, Target_Data] = field(default_factory=dict, repr=False)

    _nested_data_fields = Processed_Data._nested_data_fields | {"_virtual_bow_dict"}
    _category_fields = Processed_Data._category_fields
    _1d_data_fields = Processed_Data._1d_data_fields
    _2d_data_fields = Processed_Data._2d_data_fields

    @property
    def bows(self) -> np.ndarray[Any, np.dtypes.UInt8DType]:
        """
        A two-dimensional array representing each document's bag-of-words
        representation.
        """
        return self._virtual_bow_dict["bows"].arr

    @property
    def words(self) -> np.ndarray[Any, np.dtypes.StrDType]:
        """
        The words that are represented by each column of the bag-of-words
        representation.
        """
        return self._virtual_bow_dict["bows"].labels

    @classmethod
    def from_processed_data(
        cls, data: Processed_Data, words: Optional[Collection[str]] = None
    ) -> "BoW_Data":
        """
        Create bag-of-words representations from a tokenized corpus.

        Note that for space efficiency, word counts within one document
        are capped at 255.

        :args words: If non-None, use these words instead of all unique ones
            found in the documents. This can be useful if the documents do not
            contain all tokens that we expect to see. Note that all tokens
            within the documents must still be present here.
        """
        words = (
            list(reduce(op.or_, (set(doc) for doc in data.processed_texts), set()))
            if words is None
            else list(words)
        )
        word_to_id = {word: index for index, word in enumerate(words)}

        def doc_to_bow(doc: Collection[str]):
            res = np.zeros(len(words), dtype=np.uint8)
            for word, count in Counter(doc).items():
                index = word_to_id[word]
                # avoid overflows, as we are using bytes here
                res[index] = min(count, 255)

            return res

        bows = np.stack(
            [doc_to_bow(doc) for doc in data.processed_texts], dtype=np.uint8
        )
        bows_data = Target_Data(
            arr=bows,
            in_test_set=np.zeros_like(data.editor_arr, dtype=np.dtypes.BoolDType),
            uris=np.array(words),
            labels=np.array(words),
        )

        return cls(
            raw_texts=data.raw_texts,
            ids=data.ids,
            editor_arr=data.editor_arr,
            target_data=data.target_data,
            processed_texts=data.processed_texts,
            languages=data.languages,
            _virtual_bow_dict={"bows": bows_data},
        )

    @classmethod
    def from_data(
        cls, data: Data, words: Optional[Collection[str]] = None, **kwargs
    ) -> "BoW_Data":
        """
        Get bag-of-words representations for unprocessed data.

        This simply applies the pre-processing step from
        :func:`data_utils.default_pipelines.data.Processed_Data.from_data`
        before calculating the bag-of-words representations.

        :param kwargs: Additional keyword-arguments to be passed onto
            :func:`data_utils.default_pipelines.data.Processed_Data.from_data`.
        """
        return cls.from_processed_data(
            Processed_Data.from_data(data, **kwargs), words=words
        )

    def __post_init__(self) -> None:
        # ensure that the processed texts and the bag-of-words contain the same
        # tokens by removing from the former all that is not contained in the
        # latter.
        words_set: set[str] = set(self.words)
        # because we cannot assign to the processed texts directly, we simply
        # change the entries in-place
        for i, doc in enumerate(self.processed_texts):
            new_doc = tuple(word for word in doc if word in words_set)
            self.processed_texts[i] = new_doc


Base_Data_Subtype = TypeVar("Base_Data_Subtype", bound=_Base_Data)
Data_Subtype = TypeVar("Data_Subtype", bound=Data)


def _copy_with_changed_values(_obj: Base_Data_Subtype, **kwargs) -> Base_Data_Subtype:
    original_dict = {
        key: value
        for key, value in asdict(_obj).items()
        # skip hidden properties
        if "_" != key[0]
    }

    return _obj.__class__(**(original_dict | kwargs))


def subset_data_points(
    data: Base_Data_Subtype, indices: Sequence[int] | NDArray[np.intp]
) -> Base_Data_Subtype:
    """
    Return a subset of the given corpus that only contains data points
    as indicated by the given sequence of indices.
    """

    def subset_arr_or_list(val: list | np.ndarray) -> list | np.ndarray:
        if isinstance(val, list):
            return [val[int(index)] for index in indices]

        return val[indices]

    changed_values = {
        key: subset_arr_or_list(getattr(data, key))
        for key in data._1d_data_fields | data._2d_data_fields
    }
    changed_nested_data = {
        key: {
            nested_key: subset_data_points(nested_data, indices)
            for nested_key, nested_data in getattr(data, key).items()
        }
        for key in data._nested_data_fields
    }
    return _copy_with_changed_values(data, **(changed_values | changed_nested_data))


def subset_categories(
    data: Base_Data_Subtype,
    indices: Sequence[int] | NDArray[np.intp],
    field: Optional[str],
) -> Base_Data_Subtype:
    """
    Return a modified version of the corpus such that the data field,
    will only contain the categories as determined by the given indices.
    """

    if field is not None:
        # if the given field is not present, raise an exception, as this is
        # almost always an error
        if all(
            field not in getattr(data, nested_field)
            for nested_field in data._nested_data_fields
        ):
            raise ValueError(
                f"The field {field} could not be found! Perhaps this is a typo?"
            )

        nested_value_changes = {
            key: {
                nested_key: subset_categories(nested_data, indices, None)
                for nested_key, nested_data in getattr(data, key).items()
                if nested_key == field
            }
            for key in data._nested_data_fields
        }
        changed_data = {
            key: getattr(data, key) | changes
            for key, changes in nested_value_changes.items()
        }

    else:
        changed_data = {
            key: getattr(data, key)[:, indices] for key in data._2d_data_fields
        } | {key: getattr(data, key)[indices] for key in data._category_fields}

    return _copy_with_changed_values(data, **changed_data)


def publish(data: Data, target_dir: Path, name: str) -> tuple[Path, Path, Path | None]:
    """
    Publish relevant data fields as csv files.

    This results in two to three files, depending on the type of ``data``:

    1. The csv containing the basic data. Name: `{name}_data.csv`
    2. The csv containing some metadata about the target categories.
       Name: `{name}_metadata.csv`
    3. (If applicable), the csv containing the processed texts.
       Name: `{name}_processed_text.csv`

    All assignments to targets are encoded as binary arrays; the URIs / labels
    can be found in the metadata file.
    """
    # ensure that the parent directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    data_file = target_dir / f"{name}_data.csv"
    metadata_file = target_dir / f"{name}_metadata.csv"

    def to_binary_strings(x: np.ndarray) -> list[str]:
        return ["".join(assignment) for assignment in x.astype(int).astype(str)]

    columns = (
        {
            "uuid": data.ids,
            "text": data.raw_texts,
            "editorially_confirmed": data.editor_arr.astype(int),
        }
        | {key: to_binary_strings(value.arr) for key, value in data.target_data.items()}
        | {
            key + "_test-set": value.in_test_set.astype(int)
            for key, value in data.target_data.items()
        }
    )

    # export the data
    with open(data_file, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(columns.keys())
        for values in zip(*list(columns.values())):
            writer.writerow(values)

    # export metadata
    with open(metadata_file, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "index", "uri", "label"])
        for field, target_data in data.target_data.items():
            for index, (uri, label) in enumerate(
                zip(target_data.uris, target_data.labels)
            ):
                writer.writerow([field, index, uri, label])

    # export processed texts, if available
    processed_text_file = None
    if isinstance(data, Processed_Data):
        processed_text_file = target_dir / f"{name}_processed_text.csv"
        with open(processed_text_file, "w+") as f:
            writer = csv.writer(f)
            writer.writerow(["uuid", "processed_text", "language"])
            for uuid, processed_text, lang in zip(
                data.ids, data.processed_texts, data.languages
            ):
                writer.writerow([uuid, processed_text, lang])

    return data_file, metadata_file, processed_text_file


@overload
def import_published(data_file: Path, metadata_file: Path) -> Data:
    ...


@overload
def import_published(
    data_file: Path, metadata_file: Path, processed_text_file: Path
) -> Processed_Data:
    ...


def import_published(
    data_file: Path, metadata_file: Path, processed_text_file: Optional[Path] = None
) -> Data | Processed_Data:
    """Read a published set of csv files."""
    raw_data = pd.read_csv(data_file, sep=",", dtype=str)
    target_fields = set(raw_data.columns) - {"uuid", "text", "editorially_confirmed"}
    target_test_fields = {field for field in target_fields if "_test-set" in field}
    target_fields = target_fields - target_test_fields

    ids = raw_data["uuid"].to_numpy()
    texts = raw_data["text"].to_numpy()
    editor_arr = raw_data["editorially_confirmed"].to_numpy(dtype=int).astype(bool)

    target_data_test_arrs = {
        field: getattr(raw_data, field).to_numpy(dtype=int).astype(bool)
        for field in target_test_fields
    }

    # convert binary arrays to numpy arrays
    target_data_arr: dict[str, np.ndarray[Any, np.dtypes.BoolDType]] = dict()
    for field in target_fields:
        target_data_arr[field] = np.array(
            [
                list(bool_array) if isinstance(bool_array, Iterable) else bool_array
                for bool_array in raw_data[field]
            ],
            dtype=int,
        ).astype(bool)

    # read the metadata to fill in remaining information on targets
    raw_metadata = pd.read_csv(metadata_file, sep=",", dtype=str)
    target_data: dict[str, Target_Data] = dict()
    for field in target_fields:
        relevant_metadata = raw_metadata[raw_metadata["field"] == field]
        uris = relevant_metadata["uri"].to_numpy()
        labels = relevant_metadata["label"].to_numpy()
        target_data[field] = Target_Data(
            arr=target_data_arr[field],
            in_test_set=target_data_test_arrs[field + "_test-set"],
            uris=uris,
            labels=labels,
        )

    data = Data(
        raw_texts=texts, ids=ids, editor_arr=editor_arr, target_data=target_data
    )

    # read the processed texts if the corresponding file was given, and
    # integrate them
    if processed_text_file is not None:
        processed_texts_df = pd.read_csv(
            processed_text_file,
            dtype={"uuid": str, "language": str},
            # note: while this line evaluates raw strings, it is relatively
            # "safe", because no actual code is executed; instead, the string
            # is evaluated as a python data structure.
            # however, a malicious actor can still cause the interpreter to
            # crash or force high CPU usage
            converters={"processed_text": ast.literal_eval},
        ).set_index("uuid")

        processed_texts_list = [
            processed_texts_df["processed_text"].loc[uuid] for uuid in data.ids
        ]
        languages = processed_texts_df["language"].to_numpy()

        data = Processed_Data(
            raw_texts=data.raw_texts,
            ids=data.ids,
            editor_arr=data.editor_arr,
            target_data=data.target_data,
            processed_texts=processed_texts_list,
            languages=languages,
        )

    return data


def balanced_split(
    data: Data_Subtype,
    field: str,
    ratio: float = 0.3,
    randomize: bool = True,
    seed: int = 0,
) -> tuple[Data_Subtype, Data_Subtype]:
    """
    Split the data into two, keeping the overall distribution for the given
    field.
    """
    if randomize:
        indices = np.arange(data.editor_arr.shape[0])
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(indices)
        data = subset_data_points(data, indices)

    target_arr = data.target_data[field].arr
    # calculate the number of target docs per category
    counts: np.ndarray = target_arr.sum(-2)
    target_counts = np.rint(counts * ratio)

    # for each target category, find the minimum number of
    # data points such that we have selected at least the target
    # number
    def find_kept(
        values: np.ndarray, target: int
    ) -> np.ndarray[Any, np.dtypes.BoolDType]:
        split = np.where(values)[0][target]
        kept = np.zeros_like(values, dtype=bool)
        kept[:split] = True
        kept = np.logical_and(values, kept)

        return kept

    # naively combine all such split points, such that each category
    # has support of at least the target
    kept = reduce(
        np.logical_or,
        [
            find_kept(target_arr[:, i], int(target_count))
            for i, target_count in enumerate(target_counts)
        ],
        np.zeros_like(data.editor_arr, dtype=bool),
    )

    # because documents can belong to multiple categories, greedily
    # try to remove each and check if all targets are still met
    for index in np.where(kept)[0]:
        kept[index] = False
        selected_cats = target_arr[index]
        if all(
            target_arr[:, selected_cats][kept].sum(-2) >= target_counts[selected_cats]
        ):
            continue
        kept[index] = True

    return (
        subset_data_points(data, np.where(~kept)[0]),
        subset_data_points(data, np.where(kept)[0]),
    )
