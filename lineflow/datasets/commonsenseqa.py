import io
import json
import os
import pickle
from functools import lru_cache
from typing import Dict, List

import gdown

from lineflow import download
from lineflow.text import Dataset


def get_commonsenseqa() -> Dict[str, List[str]]:
    train_url = "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl"
    dev_url = "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl"
    test_url = "https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl"
    root = download.get_cache_directory(os.path.join("datasets", "commonsenseqa"))

    def creator(path):
        train_path = gdown.cached_download(train_url)
        dev_path = gdown.cached_download(dev_url)
        test_path = gdown.cached_download(test_url)

        dataset = {}
        for split in ("train", "dev", "test"):
            data_path = {"train": train_path, "dev": dev_path, "test": test_path}[split]
            with io.open(data_path, "rt", encoding="utf-8") as f:
                data = [json.loads(line) for line in f.readlines()]
            temp = []
            for x in data:
                answer_key = x["answerKey"] if split != "test" else ""
                options = {choice["label"]: choice["text"] for choice in x["question"]["choices"]}
                stem = x["question"]["stem"]
                temp.append({
                    "id": x["id"],
                    "answer_key": answer_key,
                    "options": options,
                    "stem": stem
                })
            dataset[split] = temp

        with io.open(path, "wb") as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, "rb") as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, "commonsenseqa.pkl")
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_commonsenseqa = lru_cache()(get_commonsenseqa)


class CommonsenseQA(Dataset):

    def __init__(self, split: str = "train") -> None:

        if split not in {"train", "dev", "test"}:
            raise ValueError(f"only 'train' and 'dev' are valid for 'split', but '{split}' is given.")

        raw = cached_get_commonsenseqa()

        super().__init__(raw[split])
