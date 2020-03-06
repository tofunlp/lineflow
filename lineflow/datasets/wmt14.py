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


def get_wmt14() -> Dict[str, Tuple[arrayfiles.TextFile]]:

    url = 'https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8'
    root = download.get_cache_directory(os.path.join('datasets', 'wmt14'))

    def creator(path):
        archive_path = gdown.cached_download(url)
        target_path = os.path.join(root, 'raw')
        with tarfile.open(archive_path, 'r') as archive:
            print(f'Extracting to {target_path}')
            archive.extractall(target_path)

        split2filename = {'train': 'train.tok.clean.bpe.32000',
                          'dev': 'newstest2013.tok.bpe.32000',
                          'test': 'newstest2014.tok.bpe.32000'}
        dataset = {}
        for split, filename in split2filename.items():
            src_path = f'{filename}.en'
            tgt_path = f'{filename}.de'
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


cached_get_wmt14 = lru_cache()(get_wmt14)


class Wmt14(ZipDataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'dev', 'test'}:
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_wmt14()
        super(Wmt14, self).__init__(*raw[split])
