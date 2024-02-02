import operator as op
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial, reduce
from pathlib import Path
from typing import Any, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray
from nlprep import Pipeline_Generator, apply_filters, pipelines, tokenize_documents
from copy import copy


@dataclass
class _Base_Data:
    _nested_data_fields: set[str] = field(init=False, default_factory=set)
    _category_fields: set[str] = field(init=False, default_factory=set)
    _1d_data_fields: set[str] = field(init=False, default_factory=set)
    _2d_data_fields: set[str] = field(init=False, default_factory=set)


@dataclass
class Target_Data(_Base_Data):
    arr: np.ndarray
    in_test_set: np.ndarray[Any, np.dtypes.BoolDType]
    uris: np.ndarray[Any, np.dtypes.StrDType]
    labels: np.ndarray[Any, np.dtypes.StrDType]

    def __post_init__(self):
        self._nested_data_fields = set()
        self._category_fields = {"uris", "labels"}
        self._1d_data_fields = {"in_test_set"}
        self._2d_data_fields = {"arr"}


@dataclass
class Data(_Base_Data):
    raw_texts: np.ndarray[Any, np.dtypes.StrDType]
    ids: np.ndarray[Any, np.dtypes.StrDType]
    redaktion_arr: np.ndarray[Any, np.dtypes.BoolDType]
    target_data: dict[str, Target_Data]

    def __post_init__(self):
        self._nested_data_fields = {"target_data"}
        self._category_fields = set()
        self._1d_data_fields = {"raw_texts", "ids", "redaktion_arr"}
        self._2d_data_fields = set()


@dataclass
class Processed_Data(Data):
    processed_texts: list[tuple[str, ...]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self._1d_data_fields.add("processed_texts")

    @classmethod
    def from_data(
        cls,
        data: Data,
        pipeline_generators: Iterable[Pipeline_Generator] = tuple(),
        cache_dir: Optional[Path] = None,
    ) -> "Processed_Data":
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
            redaktion_arr=data.redaktion_arr,
            target_data=data.target_data,
            processed_texts=[doc.selected_tokens for doc in docs],
        )


@dataclass
class BoW_Data(Processed_Data):
    _virtual_bow_dict: dict[str, Target_Data] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self._nested_data_fields.add("_virtual_bow_dict")

    @property
    def bows(self) -> np.ndarray[Any, np.dtypes.IntDType]:
        """Alias for accessing the bag of words representations more conveniently."""
        return self._virtual_bow_dict["bows"].arr

    @property
    def words(self) -> np.ndarray[Any, np.dtypes.StrDType]:
        """Alias for accessing the id-to-word map more conveniently."""
        return self._virtual_bow_dict["bows"].labels

    @classmethod
    def from_processed_data(cls, data: Processed_Data) -> "BoW_Data":
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
            redaktion_arr=data.redaktion_arr,
            target_data=data.target_data,
            processed_texts=data.processed_texts,
            _virtual_bow_dict={"bows": bows_data},
        )

    @classmethod
    def from_data(
        cls,
        data: Data,
        pipeline_generators: Iterable[Pipeline_Generator] = tuple(),
        cache_dir: Optional[Path] = None,
    ) -> "BoW_Data":
        return cls.from_processed_data(
            Processed_Data.from_data(
                data, pipeline_generators=pipeline_generators, cache_dir=cache_dir
            )
        )


T = TypeVar("T")
Base_Data_Subtype = TypeVar("Base_Data_Subtype", bound=_Base_Data)
Data_Subtype = TypeVar("Data_Subtype", bound=Data)
BoW_Subtype = TypeVar("BoW_Subtype", bound=BoW_Data)


def _copy_with_changed_values(_obj: T, **kwargs) -> T:
    obj = copy(_obj)
    for key, value in kwargs.items():
        setattr(obj, key, value)

    return obj


def subset_data_points(
    data: Base_Data_Subtype, indices: Sequence[int] | NDArray[np.intp]
) -> Base_Data_Subtype:
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
    data: Data_Subtype, indices: Sequence[int] | NDArray[np.intp], field: str
) -> Data_Subtype:
    changed_nested_data = {
        key: getattr(data, key)
        | {
            nested_key: subset_categories(nested_data, indices, field)
            for nested_key, nested_data in getattr(data, key).items()
            if nested_key == field
        }
        for key in data._nested_data_fields
    }
    changed_values = {
        key: getattr(data, key)[:, indices] for key in data._2d_data_fields
    } | {key: getattr(data, key)[indices] for key in data._category_fields}

    return _copy_with_changed_values(data, **(changed_nested_data | changed_values))
