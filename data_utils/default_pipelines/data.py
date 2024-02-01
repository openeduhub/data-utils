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
from tqdm import tqdm
from copy import copy


@dataclass
class _Base_Data:
    _nested_data_fields: set[str] = field(init=False)
    _category_fields: set[str] = field(init=False)
    _1d_data_fields: set[str] = field(init=False)
    _2d_data_fields: set[str] = field(init=False)


@dataclass
class Target_Data(_Base_Data):
    arr: np.ndarray[Any, np.dtypes.BoolDType]
    in_test_set: np.ndarray[Any, np.dtypes.BoolDType]
    uris: np.ndarray[Any, np.dtypes.StrDType]
    labels: np.ndarray[Any, np.dtypes.StrDType]

    _nested_data_fields = set()
    _category_fields = {"uris", "labels"}
    _1d_data_fields = {"in_test_set"}
    _2d_data_fields = {"arr"}


@dataclass
class Data(_Base_Data):
    raw_texts: np.ndarray[Any, np.dtypes.StrDType]
    ids: np.ndarray[Any, np.dtypes.StrDType]
    redaktion_arr: np.ndarray[Any, np.dtypes.BoolDType]
    target_data: dict[str, Target_Data]

    _nested_data_fields = {"target_data"}
    _category_fields = set()
    _1d_data_fields = {"raw_texts", "ids", "redaktion_arr"}
    _2d_data_fields = set()


@dataclass
class Processed_Data(Data):
    processed_texts: list[tuple[str, ...]] = field(default_factory=list)

    def __post_init__(self):
        self._1d_data_fields.add("processed_texts")

    @classmethod
    def from_data(
        cls,
        data: Data,
        pipeline_generators: Iterable[Pipeline_Generator] = tuple(),
        cache_dir: Optional[Path] = None,
    ) -> "Processed_Data":
        import nlprep.spacy as nlp

        print("Pre-processing texts...")
        try:
            nlp.utils.load_caches(
                cache_dir,  # type: ignore
                file_prefix="nlp_cache",
            )
        except (FileNotFoundError, TypeError):
            print("No NLP cache found. This may take a while...")

        docs = list(
            tokenize_documents(tqdm(data.raw_texts.tolist()), nlp.tokenize_as_lemmas)
        )

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
    bows: np.ndarray[Any, np.dtypes.UInt8DType] = field(
        default_factory=lambda: np.array([])
    )
    id_to_word: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        self._1d_data_fields.add("bows")

    @classmethod
    def from_processed_data(cls, data: Processed_Data) -> "BoW_Data":
        print("Converting texts to bag of words representations...")
        words = list(reduce(op.or_, (set(doc) for doc in data.processed_texts), set()))
        word_to_id = {word: index for index, word in enumerate(words)}

        def doc_to_bow(doc: Collection[str]):
            res = np.zeros(len(words), dtype=np.uint8)
            for word, count in Counter(doc).items():
                index = word_to_id[word]
                # avoid overflows, as we are using bytes here
                res[index] = min(count, 255)

            return res

        return cls(
            raw_texts=data.raw_texts,
            ids=data.ids,
            redaktion_arr=data.redaktion_arr,
            target_data=data.target_data,
            processed_texts=data.processed_texts,
            bows=np.stack([doc_to_bow(doc) for doc in tqdm(data.processed_texts)]),
            id_to_word={index: word for index, word in enumerate(words)},
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
    changed_values = {
        key: getattr(data, key)[indices]
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
    new_target_data = data.target_data | {
        field: Target_Data(
            arr=data.target_data[field].arr[:, indices],
            in_test_set=data.target_data[field].in_test_set,
            uris=data.target_data[field].uris[indices],
            labels=data.target_data[field].labels[indices],
        )
    }

    return _copy_with_changed_values(data, target_data=new_target_data)


def subset_words(
    data: BoW_Subtype, indices: Sequence[int] | NDArray[np.intp]
) -> BoW_Subtype:
    bows = data.bows[:, indices]
    id_to_word = {
        new_index: data.id_to_word[int(old_index)]
        for new_index, old_index in enumerate(indices)
    }

    return _copy_with_changed_values(data, bows=bows, id_to_word=id_to_word)
