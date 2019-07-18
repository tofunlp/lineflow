import easyfile

from lineflow import Dataset
from lineflow.download import cached_download


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

        fieldnames = ('quality', 'id1', 'id2', 'string1', 'string2')
        dataset = easyfile.CsvFile(path,
                                   encoding='utf-8',
                                   header=True,
                                   delimiter='\t',
                                   fieldnames=fieldnames)

        super().__init__(dataset)
