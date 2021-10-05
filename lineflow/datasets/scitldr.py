import json
import os
import pickle
from functools import lru_cache
from typing import Any, Dict

import gdown

from lineflow import Dataset, download


def get_scitldr(mode: str = "a") -> Dict[str, Any]:

    url = {
        "a": "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-A/{}.jsonl",
        "aic": "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-AIC/{}.jsonl",
        "full": "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-FullText/{}.json",
    }[mode]

    root = download.get_cache_directory(os.path.join("datasets", "scitldr"))

    def creator(path):
        dataset = {}
        for split in ("train", "test", "dev"):
            d_path = gdown.cached_download(url.format(split))
            dataset[split] = []
            with open(d_path, "r") as _f:
                for line in _f.readlines():
                    dataset[split].append(json.loads(line))

        with open(path, "wb") as _f:
            pickle.dump(dataset, _f)
        return dataset

    def loader(path):
        with open(path, "rb") as _f:
            return pickle.load(_f)

    pkl_path = os.path.join(root, "scitldr.pkl")
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_scitldr = lru_cache()(get_scitldr)


class SciTLDR(Dataset):
    def __init__(self, mode: str = "a", split: str = "train") -> None:
        if mode not in ("a", "aic", "full"):
            raise ValueError(
                f"only 'a', 'aic' and 'full' are for valid for 'mode', but '{mode}' is given."
            )
        if split not in ("train", "dev", "test"):
            raise ValueError(
                f"only 'train', 'dev' and 'test' are for valid for 'split', but '{split}' is given."
            )
        raw = cached_get_scitldr(mode)
        super().__init__(raw[split])
