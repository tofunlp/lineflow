from typing import Union, List, Dict, Iterator
import os
import os.path as osp
import csv
import mmap
import linecache

from lineflow import Dataset
from lineflow.core import RandomAccessConcat, RandomAccessZip


class RandomAccessText:
    def __init__(self, path: str, encoding: str = 'utf-8') -> None:
        assert osp.exists(path)

        self._path = path
        self._encoding = encoding
        self._length = None
        self._offset = 1

    def _get_length(self) -> int:
        with open(self._path, 'r+', encoding=self._encoding) as f:
            mm = mmap.mmap(f.fileno(), 0)
            return sum(1 for _ in iter(mm.readline, b''))

    def __iter__(self) -> Iterator[str]:
        with open(self._path, encoding=self._encoding) as f:
            for line in f:
                yield line.rstrip(os.linesep)

    def __getitem__(self, index: int) -> str:
        if index < 0 or len(self) <= index:
            raise IndexError('RandomAccessText object index out of range')
        return linecache.getline(self._path, index + self._offset).rstrip(os.linesep)

    def __len__(self) -> int:
        if self._length is None:
            self._length = self._get_length()
        return self._length


class RandomAccessCsv(RandomAccessText):
    def __init__(self,
                 path: str,
                 encoding: str = 'utf-8',
                 delimiter: str = ',',
                 header: bool = False) -> None:
        super().__init__(path, encoding)

        self._delimiter = delimiter
        self._reader = csv.DictReader if header else csv.reader
        if header:
            with open(path, encoding=encoding) as f:
                self._header = next(csv.reader(f, delimiter=delimiter))
            self._offset = 2
        else:
            self._header = None

    def _get_length(self) -> int:
        length = super()._get_length()
        if self._header is None:
            return length
        else:
            return length - 1

    def __iter__(self) -> Iterator[Union[List[str], Dict[str, str]]]:
        with open(self._path, encoding=self._encoding) as f:
            if self._header is None:
                yield from self._reader(f, delimiter=self._delimiter)
            else:
                reader = self._reader(f, delimiter=self._delimiter, fieldnames=self._header)
                next(reader)
                yield from reader

    def __getitem__(self, i: int) -> Union[List[str], Dict[str, str]]:
        x = super().__getitem__(i)
        if self._header is None:
            row = self._reader([x], delimiter=self._delimiter)
        else:
            row = self._reader([x], delimiter=self._delimiter, fieldnames=self._header)
        return next(row)


class TextDataset(Dataset):
    def __init__(self,
                 paths: Union[str, List[str]],
                 encoding: str = 'utf-8',
                 mode: str = 'zip') -> None:
        if isinstance(paths, str):
            dataset = RandomAccessText(paths, encoding)
        elif isinstance(paths, list):
            if mode == 'zip':
                dataset = RandomAccessZip(*[RandomAccessText(p, encoding) for p in paths])
            elif mode == 'concat':
                dataset = RandomAccessConcat(*[RandomAccessText(p, encoding) for p in paths])
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
            RandomAccessCsv(path=path, encoding=encoding, delimiter=delimiter, header=header))
