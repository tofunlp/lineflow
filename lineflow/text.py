from typing import Union, List, Dict, Iterator
import os
import os.path as osp
import csv
import mmap

from lineflow import Dataset
from lineflow.core import ConcatDataset, ZipDataset


class RandomAccessFile:
    def __init__(self, path: str, encoding: str = 'utf-8') -> None:
        assert osp.exists(path)

        self._path = path
        self._encoding = encoding
        self._offsets = None
        self._length = None

    def _initialize_offsets(self) -> None:
        with open(self._path, 'r+', encoding=self._encoding) as f:
            mm = mmap.mmap(f.fileno(), 0)
            offsets = [0] + [mm.tell() for _ in iter(mm.readline, b'')]
        self._offsets = offsets

    def __iter__(self) -> Iterator[str]:
        with open(self._path, encoding=self._encoding) as f:
            for line in f:
                yield line.rstrip(os.linesep)

    def __getitem__(self, index: int) -> str:
        if self._offsets is None:
            self._initialize_offsets()
        if index < 0 or len(self) <= index:
            raise IndexError('RandomAccessTextFile object index out of range')
        with open(self._path, encoding=self._encoding) as f:
            f.seek(self._offsets[index])
            return f.readline().rstrip(os.linesep)

    def __len__(self) -> int:
        if self._offsets is None:
            self._initialize_offsets()
        if self._length is None:
            self._length = len(self._offsets) - 1
        return self._length


class SingleTextDataset(Dataset):
    def __init__(self,
                 path: str,
                 encoding: str = 'utf-8') -> None:
        super().__init__(RandomAccessFile(path, encoding))

        self._path = path
        self._encoding = encoding


class ConcatTextDataset(ConcatDataset):
    def __init__(self,
                 paths: List[str],
                 encoding: str = 'utf-8') -> None:
        super().__init__(
            *[SingleTextDataset(path, encoding) for path in paths]
        )


class ZipTextDataset(ZipDataset):
    def __init__(self,
                 paths: List[str],
                 encoding: str = 'utf-8') -> None:
        super().__init__(
            *[SingleTextDataset(path, encoding) for path in paths]
        )


class TextDataset(Dataset):
    def __init__(self,
                 paths: Union[str, List[str]],
                 encoding: str = 'utf-8',
                 mode: str = 'zip') -> None:
        if isinstance(paths, str):
            dataset = SingleTextDataset(paths, encoding)
        elif isinstance(paths, list):
            if mode == 'zip':
                dataset = ZipTextDataset(paths, encoding)
            elif mode == 'concat':
                dataset = ConcatTextDataset(paths, encoding)
            else:
                raise ValueError(f"only 'zip' and 'concat' are valid for 'mode', but '{mode}' is given.")

        super().__init__(dataset)


class CsvDataset(SingleTextDataset):
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
                self._header = next(csv.reader(f))
        else:
            self._header = None

    def __iter__(self) -> Iterator[Union[List[str], Dict[str, str]]]:
        with open(self._path, encoding=self._encoding) as f:
            if self._header is None:
                yield from self._reader(f, delimiter=self._delimiter)
            else:
                reader = self._reader(f, delimiter=self._delimiter, fieldnames=self._header)
                next(reader)
                yield from reader

    def get_example(self, i: int) -> Union[List[str], Dict[str, str]]:
        if self._header is None:
            row = self._reader([self._dataset[i]],
                               delimiter=self._delimiter)
        else:
            row = self._reader([self._dataset[i + 1]],
                               delimiter=self._delimiter,
                               fieldnames=self._header)
        return next(row)

    def get_length(self) -> int:
        length = len(self._dataset)
        if self._header is None:
            return length
        else:
            return length - 1
