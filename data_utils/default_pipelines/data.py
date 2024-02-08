from __future__ import annotations

import operator as op
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, Optional, TypeVar

import numpy as np
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
    def from_processed_data(cls, data: Processed_Data) -> "BoW_Data":
        """
        Create bag-of-words representations from a tokenized corpus.

        Note that for space efficiency, word counts within one document
        are capped at 255.
        """
        words: list[str] = list(
            reduce(op.or_, (set(doc) for doc in data.processed_texts), set())
        )
        word_to_id = {word: index for index, word in enumerate(words)}

        def doc_to_bow(doc: Collection[str]):
            res = np.zeros(len(words), dtype=np.uint8)
            for word, count in Counter(doc).items():
                index = word_to_id[word]
                # avoid overflows, as we are using bytes here
                res[index] = min(count, 255)

            return res

        bows = np.stack([doc_to_bow(doc) for doc in data.processed_texts])
        bows_data = Target_Data(
            arr=bows,
            in_test_set=np.zeros_like(bows, dtype=np.dtypes.BoolDType),
            uris=np.array(words),
            labels=np.array(words),
        )

        return cls(
            raw_texts=data.raw_texts,
            ids=data.ids,
            editor_arr=data.editor_arr,
            target_data=data.target_data,
            processed_texts=data.processed_texts,
            _virtual_bow_dict={"bows": bows_data},
        )

    @classmethod
    def from_data(cls, data: Data, **kwargs) -> "BoW_Data":
        """
        Get bag-of-words representations for unprocessed data.

        This simply applies the pre-processing step from
        :func:`data_utils.default_pipelines.data.Processed_Data.from_data`
        before calculating the bag-of-words representations.

        :param kwargs: Additional keyword-arguments to be passed onto
            :func:`data_utils.default_pipelines.data.Processed_Data.from_data`.
        """
        return cls.from_processed_data(Processed_Data.from_data(data, **kwargs))


Base_Data_Subtype = TypeVar("Base_Data_Subtype", bound=_Base_Data)


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
        changed_data = {
            key: getattr(data, key)
            | {
                nested_key: subset_categories(nested_data, indices, None)
                for nested_key, nested_data in getattr(data, key).items()
                if nested_key == field
            }
            for key in data._nested_data_fields
        }

    else:
        changed_data = {
            key: getattr(data, key)[:, indices] for key in data._2d_data_fields
        } | {key: getattr(data, key)[indices] for key in data._category_fields}

    return _copy_with_changed_values(data, **changed_data)
