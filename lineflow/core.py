import os
import pickle
import linecache
import copy
import mmap
from pathlib import Path
from itertools import accumulate, chain, islice
from bisect import bisect


class Dataset:
    def __init__(self, dataset):
        if isinstance(dataset, Dataset):
            self._dataset = dataset._dataset
        else:
            self._dataset = dataset

        self._length = None

    def __iter__(self):
        yield from self._dataset

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in range(start, stop, step)]
        return self.get_example(index)

    def __len__(self):
        if self._length is None:
            self._length = self.get_length()
        return self._length

    def __add__(self, other):
        return ConcatDataset(self, other)

    def get_example(self, i):
        return self._dataset[i]

    def get_length(self):
        return len(self._dataset)

    def map(self, map_func):
        return MapDataset(self, map_func)

    def all(self):
        return list(self)

    def take(self, n):
        return list(islice(self, n))

    def first(self):
        return next(iter(self))

    def save(self, filename):
        if os.path.exists(filename):
            print(f'Loading data from {filename}...')
            with open(filename, 'rb') as f:
                cache = pickle.load(f)
        else:
            print(f'Saving data to {filename}...')
            cache = list(self)
            with open(filename, 'wb') as f:
                pickle.dump(cache, f)
        return CacheDataset(self, cache)

    @staticmethod
    def load(filename):
        print(f'Loading data from {filename}...')
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        return Dataset(dataset)


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        assert all(isinstance(d, Dataset) for d in datasets)

        self._datasets = datasets
        self._lengths = list(accumulate(len(d) for d in datasets))
        self._length = self._lengths[-1]
        self._offsets = [0] + self._lengths[:-1]

    def __iter__(self):
        for d in self._datasets:
            yield from d

    def get_example(self, i):
        if i >= self._length:
            raise IndexError('ConcatDataset object index out of range')
        j = bisect(self._lengths, i)
        return self._datasets[j][i - self._offsets[j]]

    @property
    def _dataset(self):
        return self


class ZipDataset(Dataset):
    def __init__(self, *datasets):
        assert all(isinstance(d, Dataset) for d in datasets)

        self._datasets = datasets
        self._length = min(len(d) for d in datasets)

    def __iter__(self):
        for x in zip(*self._datasets):
            yield tuple(x)

    def get_example(self, i):
        if i >= self._length:
            raise IndexError('ZipDataset object index out of range')
        return tuple(d[i] for d in self._datasets)

    @property
    def _dataset(self):
        return self


class MapDataset(Dataset):
    def __init__(self, dataset, map_func):
        assert callable(map_func)

        if isinstance(dataset, MapDataset):
            map_func_list = copy.deepcopy(dataset._map_func_list)
            map_func_list.append(map_func)
        else:
            map_func_list = [map_func]

        self._map_func_list = map_func_list

        super().__init__(dataset)

    def __iter__(self):
        for x in self._dataset:
            for map_func in self._map_func_list:
                x = map_func(x)
            yield x

    def get_example(self, i):
        x = self._dataset[i]
        for map_func in self._map_func_list:
            x = map_func(x)
        return x


class CacheDataset(MapDataset):
    def __init__(self, dataset, cache):
        if isinstance(dataset, MapDataset):
            map_func_list = copy.deepcopy(dataset._map_func_list)
        else:
            map_func_list = []

        self._map_func_list = map_func_list
        self._cache = cache
        self._length = len(self._cache)

        super(MapDataset, self).__init__(dataset)

    def __iter__(self):
        yield from self._cache

    def get_example(self, i):
        return self._cache[i]


class TextDataset(Dataset):
    def __init__(self, filepaths, encoding='utf-8'):
        if isinstance(filepaths, str):
            filepaths = Path(filepaths)
            assert filepaths.exists()
            self._iterate = self._iterate_sinle_file
            self.get_example = self._getline_from_single_file
        else:
            filepaths = [Path(p) for p in filepaths]
            assert all(p.is_file() for p in filepaths)
            self._iterate = self._iterate_multiple_files
            self.get_example = self._getlines_from_multiple_files

        self._filepaths = filepaths
        self._encoding = encoding
        self._length = None

    def __iter__(self):
        yield from self._iterate()

    def _iterate_sinle_file(self):
        with self._filepaths.open(encoding=self._encoding) as f:
            for line in f:
                yield line.rstrip(os.linesep)

    def _iterate_multiple_files(self):
        fps = [p.open(encoding=self._encoding) for p in self._filepaths]
        for lines in zip(*fps):
            yield tuple(l.rstrip(os.linesep) for l in lines)
        for fp in fps:
            fp.close()

    def _getline_from_single_file(self, i):
        return linecache.getline(str(self._filepaths), i + 1).rstrip(os.linesep)

    def _getlines_from_multiple_files(self, i):
        return tuple(linecache.getline(str(p), i + 1).rstrip(os.linesep)
                     for p in self._filepaths)

    def get_length(self):
        if isinstance(self._filepaths, list):
            return self._count_lines(self._filepaths[0])
        else:
            return self._count_lines(self._filepaths)

    def _count_lines(self, filepath):
        count = 0
        with filepath.open(mode='r+', encoding=self._encoding) as f:
            mm = mmap.mmap(f.fileno(), 0)
            while mm.readline():
                count += 1
        return count

    @property
    def _dataset(self):
        return self


def lineflow_concat(*datasets):
    return ConcatDataset(*datasets)


def lineflow_zip(*datasets):
    return ZipDataset(*datasets)


def lineflow_filter(predicate, dataset, lazy=False):
    iterator = filter(predicate, dataset)
    if lazy:
        return iterator
    else:
        return list(iterator)


def lineflow_flat_map(map_func, dataset, lazy=False):
    iterator = chain.from_iterable(map(map_func, dataset))
    if lazy:
        return iterator
    else:
        return list(iterator)
