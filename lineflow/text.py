from typing import Union, List

import easyfile

from lineflow import Dataset
from lineflow.core import RandomAccessConcat, RandomAccessZip


class TextDataset(Dataset):
    def __init__(self,
                 paths: Union[str, List[str]],
                 encoding: str = 'utf-8',
                 mode: str = 'zip') -> None:
        if isinstance(paths, str):
            dataset = easyfile.TextFile(paths, encoding)
        elif isinstance(paths, list):
            if mode == 'zip':
                dataset = RandomAccessZip(*[easyfile.TextFile(p, encoding) for p in paths])
            elif mode == 'concat':
                dataset = RandomAccessConcat(*[easyfile.TextFile(p, encoding) for p in paths])
            else:
                raise ValueError(f"only 'zip' and 'concat' are valid for 'mode', but '{mode}' is given.")

        super().__init__(dataset)


class CsvDataset(Dataset):
    def __init__(self,
                 path: str,
                 encoding: str = 'utf-8',
                 delimiter: str = ',',
                 header: bool = False) -> None:

        super().__init__(
            easyfile.CsvFile(path=path, encoding=encoding, delimiter=delimiter, header=header))
