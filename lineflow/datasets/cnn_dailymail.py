import io
import os
import pickle
import tarfile
from functools import lru_cache
from typing import Dict, Tuple

import arrayfiles
import gdown

from lineflow import download
from lineflow.core import ZipDataset


def get_cnn_dailymail() -> Dict[str, Tuple[arrayfiles.TextFile]]:

    url = 'https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz'
    root = download.get_cache_directory(os.path.join('datasets', 'cnn_dailymail'))

    def creator(path):
        archive_path = gdown.cached_download(url)
        target_path = os.path.join(root, 'raw')
        with tarfile.open(archive_path, 'r') as archive:
            print(f'Extracting to {target_path}')
            archive.extractall(target_path)

        dataset = {}
        for split in ('train', 'dev', 'test'):
            src_path = f'{split if split != "dev" else "val"}.txt.src'
            tgt_path = f'{split if split != "dev" else "val"}.txt.tgt.tagged'
            dataset[split] = (
                arrayfiles.TextFile(os.path.join(target_path, src_path)),
                arrayfiles.TextFile(os.path.join(target_path, tgt_path))
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
        if split not in {'train', 'dev', 'test'}:
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_cnn_dailymail()
        super(CnnDailymail, self).__init__(*raw[split])
