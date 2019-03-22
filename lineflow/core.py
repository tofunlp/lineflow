from typing import Sequence, Any, Union, Callable, List, Tuple, Iterator, Iterable
import os
import pickle
import linecache
import copy
import mmap
from pathlib import Path
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
        print(f'Loading data from {filename}...')
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        return Dataset(dataset)


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
                 filepaths: Union[str, List[str]],
                 encoding: str = 'utf-8') -> None:
        if isinstance(filepaths, str):
            filepaths = Path(filepaths)
            assert filepaths.exists()
            self.get_iterator = self._iterate_sinle_file
            self.get_example = self._getline_from_single_file
        else:
            filepaths = [Path(p) for p in filepaths]
            assert all(p.is_file() for p in filepaths)
            self.get_iterator = self._iterate_multiple_files
            self.get_example = self._getlines_from_multiple_files

        self._filepaths = filepaths
        self._encoding = encoding
        self._length = None

    def __iter__(self) -> Iterator[Union[str, Tuple[str]]]:
        yield from self.get_iterator()

    def _iterate_sinle_file(self) -> Iterator[str]:
        with self._filepaths.open(encoding=self._encoding) as f:
            for line in f:
                yield line.rstrip(os.linesep)

    def _iterate_multiple_files(self) -> Iterator[Tuple[str]]:
        fps = [p.open(encoding=self._encoding) for p in self._filepaths]
        for lines in zip(*fps):
            yield tuple(l.rstrip(os.linesep) for l in lines)
        for fp in fps:
            fp.close()

    def _getline_from_single_file(self, i: int) -> str:
        return linecache.getline(str(self._filepaths), i + 1).rstrip(os.linesep)

    def _getlines_from_multiple_files(self, i: int) -> Tuple[str]:
        return tuple(linecache.getline(str(p), i + 1).rstrip(os.linesep)
                     for p in self._filepaths)

    def get_length(self) -> int:
        if isinstance(self._filepaths, list):
            return self._count_lines(self._filepaths[0])
        else:
            return self._count_lines(self._filepaths)

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
