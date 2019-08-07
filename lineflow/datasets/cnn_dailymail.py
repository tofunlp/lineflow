from typing import Dict, Tuple
import os
import io
import tarfile
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
            archive.extractall(root)

        dataset = {}
        for split in ('train', 'dev', 'test'):
            dataset[split] = tuple(easyfile.TextFile(
                os.path.join(root, filename.format(split if split != 'dev' else 'val')))
                for filename in ('{}.txt.src', '{}.txt.tgt.tagged'))

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, 'cnndm.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


class CnnDailymail(ZipDataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in ('train', 'dev', 'test'):
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = get_cnn_dailymail()
        super(CnnDailymail, self).__init__(*raw[split])
