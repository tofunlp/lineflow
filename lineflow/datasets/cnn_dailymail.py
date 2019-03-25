import tarfile
from pathlib import Path

from ..download import cached_download
from ..download import get_cache_directory
from ..datasets import Seq2SeqDataset


CNN_DAILYMAIL_URL = 'https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz'

TRAIN_SOURCE_NAME = 'train.txt.src'
TRAIN_TARGET_NAME = 'train.txt.tgt.tagged'

VAL_SOURCE_NAME = 'val.txt.src'
VAL_TARGET_NAME = 'val.txt.tgt.tagged'

TEST_SOURCE_NAME = 'test.txt.src'
TEST_TARGET_NAME = 'test.txt.tgt.tagged'

ALL = (TRAIN_SOURCE_NAME, TRAIN_TARGET_NAME,
       VAL_SOURCE_NAME, VAL_TARGET_NAME,
       TEST_SOURCE_NAME, TEST_TARGET_NAME)


class CnnDailymail(Seq2SeqDataset):
    def __init__(self, split: str = 'train') -> None:
        path = cached_download(CNN_DAILYMAIL_URL)
        tf = tarfile.open(path, 'r')
        cache_dir = Path(get_cache_directory('cnndm'))
        if not all((cache_dir / p).exists() for p in ALL):
            print(f'Extracting from {path}...')
            tf.extractall(cache_dir)

        if split == 'train':
            src_path = cache_dir / TRAIN_SOURCE_NAME
            tgt_path = cache_dir / TRAIN_TARGET_NAME
        elif split == 'dev':
            src_path = cache_dir / VAL_SOURCE_NAME
            tgt_path = cache_dir / VAL_TARGET_NAME
        elif split == 'test':
            src_path = cache_dir / TEST_SOURCE_NAME
            tgt_path = cache_dir / TEST_TARGET_NAME
        else:
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        super().__init__(source_file_path=src_path, target_file_path=tgt_path)
