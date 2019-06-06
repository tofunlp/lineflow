from lineflow.download import cached_download
from lineflow import Dataset
from lineflow.text import CsvDataset


TRAIN_URL = 'https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_train.txt'  # NOQA
TEST_URL = 'https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_test.txt'  # NOQA


class MsrParaphrase(Dataset):
    def __init__(self,
                 split: str = 'train') -> None:
        if split == 'train':
            path = cached_download(TRAIN_URL)
        elif split == 'test':
            path = cached_download(TEST_URL)
        else:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        dataset = CsvDataset(path, header=True, delimiter='\t')
        dataset._header = ('quality', '#1id', '#2id', '#1string', '#2string')

        super().__init__(dataset)
