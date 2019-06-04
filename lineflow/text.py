from typing import Union, List, Tuple, Dict, Iterator
import os
import os.path as osp
import linecache
import csv
import mmap

from lineflow import Dataset
from lineflow.core import ConcatDataset, ZipDataset


def count_lines(filepath: str, encoding: str = 'utf-8') -> int:
    count = 0
    with open(filepath, mode='r+', encoding=encoding) as f:
        mm = mmap.mmap(f.fileno(), 0)
        while mm.readline():
            count += 1
    return count


class SingleTextDataset(Dataset):
    def __init__(self,
                 filepath: str,
                 encoding: str = 'utf-8') -> None:
        self._filepath = filepath

        assert osp.exists(filepath)

        self._encoding = encoding
        self._length = None

    def __iter__(self) -> Iterator[str]:
        with open(self._filepath, encoding=self._encoding) as f:
            for line in f:
                yield line.rstrip(os.linesep)

    def get_example(self, i: int) -> str:
        return linecache.getline(self._filepath, i + 1).rstrip(os.linesep)

    def get_length(self) -> int:
        return count_lines(self._filepath, self._encoding)

    @property
    def _dataset(self) -> 'SingleTextDataset':
        return self


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
                 filepath: Union[str, List[str]],
                 encoding: str = 'utf-8',
                 mode: str = 'zip') -> None:
        if isinstance(filepath, str):
            self.__dataset = SingleTextDataset(filepath, encoding)
        elif isinstance(filepath, list):
            if mode == 'zip':
                self.__dataset = ZipTextDataset(filepath, encoding)
            elif mode == 'concat':
                self.__dataset = ConcatTextDataset(filepath, encoding)
            else:
                raise ValueError(f"only 'zip' and 'concat' are valid for 'mode', but '{mode}' is given.")

        self._filepath = filepath
        self._length = self.__dataset._length

    def __iter__(self) -> Iterator[Union[str, Tuple[str]]]:
        yield from self.__dataset

    def get_example(self, i: int) -> Union[str, Tuple[str]]:
        return self.__dataset[i]

    def get_length(self) -> int:
        return len(self.__dataset)

    @property
    def _dataset(self) -> Union[SingleTextDataset, ZipTextDataset, ConcatTextDataset]:
        return self.__dataset


class CsvDataset(SingleTextDataset):
    def __init__(self,
                 filepath: str,
                 encoding: str = 'utf-8',
                 delimiter: str = ',',
                 header: bool = False) -> None:

        super().__init__(filepath, encoding)

        self._delimiter = delimiter
        self._reader = csv.DictReader if header else csv.reader
        if header:
            with open(filepath, encoding=encoding) as f:
                self._header = next(csv.reader(f))
        else:
            self._header = None

    def __iter__(self) -> Iterator[Union[List[str], Dict[str, str]]]:
        with open(self._filepath, encoding=self._encoding) as f:
            yield from self._reader(f, delimiter=self._delimiter)

    def get_example(self, i: int) -> Union[List[str], Dict[str, str]]:
        if self._header is None:
            row = self._reader([super().get_example(i)],
                               delimiter=self._delimiter)
        else:
            row = self._reader([super().get_example(i + 1)],
                               delimiter=self._delimiter,
                               fieldnames=self._header)
        return next(row)

    def get_length(self) -> int:
        count = count_lines(self._filepath, self._encoding)
        if self._header is None:
            return count
        else:
            return count - 1
