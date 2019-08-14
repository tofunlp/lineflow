from typing import Dict, Tuple
import os
import io
import tarfile
from functools import lru_cache
import pickle

import easyfile

from lineflow.core import ZipDataset
from lineflow import download


def get_cnn_dailymail() -> Dict[str, Tuple[easyfile.TextFile]]:

    url = 'https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz'
    root = download.get_cache_directory(os.path.join('datasets', 'cnndm'))

    def creator(path):
        archive_path = download.cached_download(url)
        with tarfile.open(archive_path, 'r') as archive:
            print(f'Extracting to {root}')
            archive.extractall(root)

        dataset = {}
        for split in ('train', 'dev', 'test'):
            src_path = f'{split if split != "dev" else "val"}.txt.src'
            tgt_path = f'{split if split != "dev" else "val"}.txt.tgt.tagged'
            dataset[split] = (
                easyfile.TextFile(os.path.join(root, src_path)),
                easyfile.TextFile(os.path.join(root, tgt_path))
            )

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, 'cnndm.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_cnn_dailymail = lru_cache()(get_cnn_dailymail)


class CnnDailymail(ZipDataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in ('train', 'dev', 'test'):
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_cnn_dailymail()
        super(CnnDailymail, self).__init__(*raw[split])
