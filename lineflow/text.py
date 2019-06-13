from typing import Union, List, Dict, Iterator
import os
import os.path as osp
import csv
import mmap

from lineflow import Dataset
from lineflow.core import RandomAccessConcat, RandomAccessZip


class RandomAccessText:
    def __init__(self, path: str, encoding: str = 'utf-8') -> None:
        assert osp.exists(path)

        self._path = path
        self._encoding = encoding
        self._ready = False
        self._length = None
        self._offsets = None
        self._mm = None

    def _prepare_reading(self) -> None:
        if self._ready:
            return
        with open(self._path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._offsets = [0] + [mm.tell() for _ in iter(mm.readline, b'')]
        self._mm = mm
        self._ready = True

    def __iter__(self) -> Iterator[str]:
        with open(self._path, encoding=self._encoding) as f:
            for line in f:
                yield line.rstrip(os.linesep)

    def __getitem__(self, index: int) -> str:
        self._prepare_reading()

        if index < 0 or len(self) <= index:
            raise IndexError('RandomAccessText object index out of range')

        start = self._offsets[index]
        end = self._offsets[index + 1]
        return self._mm[start:end].decode(self._encoding).rstrip(os.linesep)

    def __len__(self) -> int:
        self._prepare_reading()
        if self._length is None:
            self._length = len(self._offsets) - 1
        return self._length

    def __del__(self) -> None:
        if self._mm is not None:
            self._mm.close()


class RandomAccessCsv(RandomAccessText):
    def __init__(self,
                 path: str,
                 encoding: str = 'utf-8',
                 delimiter: str = ',',
                 header: bool = False) -> None:
        super().__init__(path, encoding)

        self._delimiter = delimiter
        self._reader = csv.DictReader if header else csv.reader
        self._header = header
        if header:
            with open(path, encoding=encoding) as f:
                self._header = next(csv.reader(f, delimiter=delimiter))
        else:
            self._header = None

    def _prepare_reading(self) -> None:
        if not self._ready:
            super()._prepare_reading()
            if self._header is not None:
                self._offsets.pop(0)

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
