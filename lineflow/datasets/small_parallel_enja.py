from ..download import cached_download
from ..datasets import Seq2SeqDataset


TRAIN_EN_URL = 'https://raw.githubusercontent.com/odashi/small_parallel_enja/master/train.en'
TRAIN_JA_URL = 'https://raw.githubusercontent.com/odashi/small_parallel_enja/master/train.ja'

DEV_EN_URL = 'https://raw.githubusercontent.com/odashi/small_parallel_enja/master/dev.en'
DEV_JA_URL = 'https://raw.githubusercontent.com/odashi/small_parallel_enja/master/dev.ja'

TEST_EN_URL = 'https://raw.githubusercontent.com/odashi/small_parallel_enja/master/test.en'
TEST_JA_URL = 'https://raw.githubusercontent.com/odashi/small_parallel_enja/master/test.ja'


class SmallParallelEnJa(Seq2SeqDataset):
    def __init__(self, split: str = 'train') -> None:
        if split == 'train':
            en_path = cached_download(TRAIN_EN_URL)
            ja_path = cached_download(TRAIN_JA_URL)
        elif split == 'dev':
            en_path = cached_download(DEV_EN_URL)
            ja_path = cached_download(DEV_JA_URL)
        elif split == 'test':
            en_path = cached_download(TEST_EN_URL)
            ja_path = cached_download(TEST_JA_URL)
        else:
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        super().__init__(source_file_path=en_path, target_file_path=ja_path)
