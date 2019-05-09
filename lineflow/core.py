from typing import Sequence, Any, Union, Callable, List, Tuple, Dict, Iterator, Iterable
import warnings
import os
import pickle
import linecache
import csv
import copy
import mmap
from pathlib import Path
from collections import OrderedDict
from itertools import accumulate, chain, islice
from bisect import bisect


class Dataset:
    def __init__(self,
                 dataset: Sequence[Any]) -> None:
        if isinstance(dataset, Dataset):
            self._dataset = dataset._dataset
        else:
            self._dataset = dataset

        self._length = None

    def __iter__(self) -> Iterator[Any]:
        yield from self._dataset

    def __getitem__(self, index: Union[int, slice]) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in range(start, stop, step)]
        return self.get_example(index)

    def __len__(self) -> int:
        if self._length is None:
            self._length = self.get_length()
        return self._length

    def __add__(self, other: 'Dataset') -> 'ConcatDataset':
        return ConcatDataset(self, other)

    def get_example(self, i: int) -> Any:
        return self._dataset[i]

    def get_length(self) -> int:
        return len(self._dataset)

    def map(self, map_func: Callable[[Any], Any]) -> 'MapDataset':
        return MapDataset(self, map_func)

    def all(self) -> List[Any]:
        return list(self)

    def take(self, n: int) -> List[Any]:
        return list(islice(self, n))

    def first(self) -> Any:
        return next(iter(self))

    def save(self, filename: str) -> 'CacheDataset':
        path = Path(filename)
        if path.exists():
            print(f'Loading data from {filename}...')
            with path.open('rb') as f:
                cache = pickle.load(f)
        else:
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
            print(f'Saving data to {filename}...')
            cache = list(self)
            with path.open('wb') as f:
                pickle.dump(cache, f)
        return CacheDataset(self, cache)

    @staticmethod
    def load(filename: str) -> 'Dataset':
        warnings.warn(
            'lineflow.Dataset.load is deprecated. Please refer to '
            'lineflow.load.',
            DeprecationWarning,
            stacklevel=2)
        return lineflow_load(filename)


class ConcatDataset(Dataset):
    def __init__(self, *datasets: List[Dataset]) -> None:
        assert all(isinstance(d, Dataset) for d in datasets)

        self._datasets = datasets
        self._lengths = list(accumulate(len(d) for d in datasets))
        self._length = self._lengths[-1]
        self._offsets = [0] + self._lengths[:-1]

    def __iter__(self) -> Iterator[Any]:
        for d in self._datasets:
            yield from d

    def get_example(self, i: int) -> Any:
        if i >= self._length:
            raise IndexError('ConcatDataset object index out of range')
        j = bisect(self._lengths, i)
        return self._datasets[j][i - self._offsets[j]]

    @property
    def _dataset(self) -> 'ConcatDataset':
        return self


class ZipDataset(Dataset):
    def __init__(self, *datasets: List[Dataset]) -> None:
        assert all(isinstance(d, Dataset) for d in datasets)

        self._datasets = datasets
        self._length = min(len(d) for d in datasets)

    def __iter__(self) -> Iterator[Any]:
        for x in zip(*self._datasets):
            yield tuple(x)

    def get_example(self, i: int) -> Any:
        if i >= self._length:
            raise IndexError('ZipDataset object index out of range')
        return tuple(d[i] for d in self._datasets)

    @property
    def _dataset(self) -> 'ZipDataset':
        return self


class MapDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 map_func: Callable[[Any], Any]) -> None:
        assert callable(map_func)

        if isinstance(dataset, MapDataset):
            funcs = copy.deepcopy(dataset._funcs)
            funcs.append(map_func)
            processed_funcs = copy.deepcopy(dataset._processed_funcs)
        else:
            funcs = [map_func]
            processed_funcs = []

        self._funcs = funcs
        self._processed_funcs = processed_funcs

        super().__init__(dataset)

    def __iter__(self) -> Iterator[Any]:
        for x in self._dataset:
            for f in self._funcs:
                x = f(x)
            yield x

    def get_example(self, i: int) -> Any:
        x = self._dataset[i]
        for f in self._funcs:
            x = f(x)
        return x


class CacheDataset(MapDataset):
    def __init__(self,
                 dataset: Dataset,
                 cache: List[Any]) -> None:
        if isinstance(dataset, MapDataset):
            funcs = copy.deepcopy(dataset._funcs)
            processed_funcs = funcs + copy.deepcopy(dataset._processed_funcs)
        else:
            processed_funcs = []

        self._funcs = []
        self._processed_funcs = processed_funcs
        self._cache = cache
        self._length = len(self._cache)

        super(MapDataset, self).__init__(cache)


class TextDataset(Dataset):
    def __init__(self,
                 filepath: Union[str, List[str]],
                 encoding: str = 'utf-8',
                 mode: str = 'zip') -> None:
        self._encoding = encoding
        self._length = None

        if isinstance(filepath, str):
            filepath = Path(filepath)
            assert filepath.exists()
            self.get_iterator = self._iterate_sinle_file
            self.get_example = self._getline_from_single_file
        else:
            filepath = [Path(p) for p in filepath]
            assert all(p.is_file() for p in filepath)
            if mode == 'zip':
                self.get_iterator = self._iterate_multiple_files_zip
                self.get_example = self._getlines_from_multiple_files_zip
            elif mode == 'concat':
                self._lengths = list(accumulate(self._count_lines(p) for p in filepath))
                self._length = self._lengths[-1]
                self._offsets = [0] + self._lengths[:-1]
                self.get_iterator = self._iterate_multiple_files_concat
                self.get_example = self._getlines_from_multiple_files_concat
            else:
                raise ValueError(f"only 'zip' and 'concat' are valid for 'mode', but '{mode}' is given.")

        self._filepath = filepath

    def __iter__(self) -> Iterator[Union[str, Tuple[str]]]:
        yield from self.get_iterator()

    def _iterate_sinle_file(self) -> Iterator[str]:
        with self._filepath.open(encoding=self._encoding) as f:
            for line in f:
                yield line.rstrip(os.linesep)

    def _iterate_multiple_files_zip(self) -> Iterator[Tuple[str]]:
        fps = [p.open(encoding=self._encoding) for p in self._filepath]
        for lines in zip(*fps):
            yield tuple(l.rstrip(os.linesep) for l in lines)
        for fp in fps:
            fp.close()

    def _iterate_multiple_files_concat(self) -> Iterator[str]:
        for p in self._filepath:
            with p.open(encoding=self._encoding) as f:
                for line in f:
                    yield line.rstrip(os.linesep)

    def _getline_from_single_file(self, i: int) -> str:
        return linecache.getline(str(self._filepath), i + 1).rstrip(os.linesep)

    def _getlines_from_multiple_files_zip(self, i: int) -> Tuple[str]:
        return tuple(linecache.getline(str(p), i + 1).rstrip(os.linesep)
                     for p in self._filepath)

    def _getlines_from_multiple_files_concat(self, i: int) -> str:
        if i >= self._length:
            return linecache.getline(str(self._filepath[-1]), self._lengths[-1]).rstrip(os.linesep)
        j = bisect(self._lengths, i)
        return linecache.getline(str(self._filepath[j]),
                                 i - self._offsets[j] + 1).rstrip(os.linesep)

    def get_length(self) -> int:
        if isinstance(self._filepath, list):
            return self._count_lines(self._filepath[0])
        else:
            return self._count_lines(self._filepath)

    def _count_lines(self, filepath: Path) -> int:
        count = 0
        with filepath.open(mode='r+', encoding=self._encoding) as f:
            mm = mmap.mmap(f.fileno(), 0)
            while mm.readline():
                count += 1
        return count

    @property
    def _dataset(self) -> 'TextDataset':
        return self


class CsvDataset(TextDataset):
    def __init__(self,
                 filepath: str,
                 encoding: str = 'utf-8',
                 delimiter: str = ',',
                 header: bool = False) -> None:
        filepath = Path(filepath)
        assert filepath.exists()

        self._filepath = filepath
        self._encoding = encoding
        self._length = None

        self._delimiter = delimiter
        self._reader = csv.DictReader if header else csv.reader
        if header:
            with filepath.open(encoding=encoding) as f:
                self._header = next(csv.reader(f))
        else:
            self._header = None

    def __iter__(self) -> Iterator[Union[List[str], Dict[str, str]]]:
        with self._filepath.open(encoding=self._encoding) as f:
            yield from self._reader(f, delimiter=self._delimiter)

    def get_example(self, i: int) -> Union[List[str], Dict[str, str]]:
        if self._header is None:
            line = self._getline_from_single_file(i)
            return next(csv.reader([line], delimiter=self._delimiter))
        else:
            line = self._getline_from_single_file(i + 1)
            row = next(csv.reader([line], delimiter=self._delimiter))
            return OrderedDict(zip(self._header, row))

    def get_length(self) -> int:
        count = self._count_lines(self._filepath)
        if self._header is None:
            return count
        else:
            return count - 1


def lineflow_concat(*datasets: List[Dataset]) -> ConcatDataset:
    return ConcatDataset(*datasets)


def lineflow_zip(*datasets: List[Dataset]) -> ZipDataset:
    return ZipDataset(*datasets)


def lineflow_filter(
        predicate: Callable[[Any], bool],
        dataset: Dataset,
        lazy: bool = False) -> Union[Iterator[Any], List[Any]]:
    iterator = filter(predicate, dataset)
    if lazy:
        return iterator
    else:
        return list(iterator)


def lineflow_flat_map(
        map_func: Callable[[Iterable[Any]], Any],
        dataset: Dataset,
        lazy: bool = False) -> Union[Iterator[Any], List[Any]]:
    iterator = chain.from_iterable(map(map_func, dataset))
    if lazy:
        return iterator
    else:
        return list(iterator)


def lineflow_load(filename: str) -> Dataset:
    print(f'Loading data from {filename}...')
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return Dataset(dataset)
