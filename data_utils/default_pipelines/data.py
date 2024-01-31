import operator as op
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Any, Optional

import numpy as np
from nlprep import Pipeline_Generator, apply_filters, pipelines, tokenize_documents
from tqdm import tqdm


@dataclass
class Target_Data:
    arr: np.ndarray[Any, np.dtypes.BoolDType]
    in_test_set: np.ndarray[Any, np.dtypes.BoolDType]
    uris: list[str]
    labels: list[str | None]


@dataclass
class Data:
    raw_texts: np.ndarray[Any, np.dtypes.StrDType]
    ids: np.ndarray[Any, np.dtypes.StrDType]
    redaktion_arr: np.ndarray[Any, np.dtypes.BoolDType]
    target_data: dict[str, Target_Data]

    def subset_data_points(self, indices: Sequence[int]) -> "Data":
        ids = self.ids[indices]
        raw_texts = self.raw_texts[indices]
        redaktion_arr = self.redaktion_arr[indices]

        new_target_data = dict()
        for field, target_data in self.target_data.items():
            target_data = Target_Data(
                arr=target_data.arr[indices],
                in_test_set=target_data.in_test_set[indices],
                uris=target_data.uris,
                labels=target_data.labels,
            )
            new_target_data[field] = target_data

        return Data(
            ids=ids,
            raw_texts=raw_texts,
            redaktion_arr=redaktion_arr,
            target_data=new_target_data,
        )

    def subset_categories(self, indices: Sequence[int], field: str) -> "Data":
        new_target_data = self.target_data | {
            field: Target_Data(
                arr=self.target_data[field].arr[:, indices],
                in_test_set=self.target_data[field].in_test_set,
                uris=[self.target_data[field].uris[i] for i in indices],
                labels=[self.target_data[field].labels[i] for i in indices],
            )
        }

        return Data(
            ids=self.ids.copy(),
            raw_texts=self.raw_texts.copy(),
            redaktion_arr=self.redaktion_arr.copy(),
            target_data=new_target_data,
        )


class Processed_Data(Data):
    processed_texts: list[tuple[str, ...]]

    @classmethod
    def __from_data_incomplete(cls, data: Data) -> "Processed_Data":
        return Processed_Data(
            raw_texts=data.raw_texts,
            ids=data.ids,
            redaktion_arr=data.redaktion_arr,
            target_data=data.target_data,
        )

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
            nlp.utils.load_caches(cache_dir, file_prefix="nlp_cache")
        except (FileNotFoundError, TypeError):
            print("No NLP cache found. This may take a while...")

        docs = list(tokenize_documents(tqdm(data.raw_texts), nlp.tokenize_as_lemmas))

        # apply the given pipelines one by one
        for pipeline_generator in pipeline_generators:
            pipeline = pipeline_generator(docs)
            docs = list(apply_filters(docs, pipeline))

        if cache_dir is not None:
            nlp.utils.save_caches(directory=cache_dir, file_prefix="nlp_cache")

        obj = cls.__from_data_incomplete(data)

        obj.processed_texts = [doc.selected_tokens for doc in docs]
        return obj

        # # drop all words that are longer than 25 characters
        # results = [[token for token in doc if len(token) <= 25] for doc in results]

    def subset_data_points(self, indices: Sequence[int]) -> "Processed_Data":
        data = super().subset_data_points(indices)
        processed_texts = [self.processed_texts[i] for i in indices]

        obj = self.__from_data_incomplete(data)
        obj.processed_texts = processed_texts

        return obj

    def subset_categories(self, indices: Sequence[int], field: str) -> "Processed_Data":
        data = super().subset_categories(indices, field)
        obj = self.__from_data_incomplete(data)
        obj.processed_texts = self.processed_texts.copy()

        return obj


class BoW_Data(Processed_Data):
    bows: np.ndarray[Any, np.dtypes.UInt8DType]

    @classmethod
    def __from_processed_data_incomplete(cls, data: Processed_Data) -> "BoW_Data":
        obj = BoW_Data(
            raw_texts=data.raw_texts,
            ids=data.ids,
            redaktion_arr=data.redaktion_arr,
            target_data=data.target_data,
        )
        obj.processed_texts = data.processed_texts

        return obj

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
                res[index] = max(count, 255)

            return res

        bows = np.stack([doc_to_bow(doc) for doc in tqdm(data.processed_texts)])

        obj = cls.__from_processed_data_incomplete(data)
        obj.bows = bows

        return obj

    @classmethod
    def from_data(
        cls,
        data: Data,
        pipeline_generators: Iterable[Pipeline_Generator] = tuple(),
        cache_dir: Optional[Path] = None,
    ) -> "BoW_Data":
        return BoW_Data.from_processed_data(
            Processed_Data.from_data(
                data, pipeline_generators=pipeline_generators, cache_dir=cache_dir
            )
        )

    def subset_data_points(self, indices: Sequence[int]) -> "BoW_Data":
        data = super().subset_data_points(indices)
        bows = self.bows[indices]

        obj = self.__from_processed_data_incomplete(data)
        obj.bows = bows

        return obj

    def subset_categories(self, indices: Sequence[int], field: str) -> "BoW_Data":
        data = super().subset_categories(indices, field)

        obj = self.__from_processed_data_incomplete(data)
        obj.bows = self.bows.copy()

        return obj
