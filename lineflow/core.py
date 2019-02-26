import os
import pickle
import linecache
import copy
import mmap
from pathlib import Path
from itertools import islice


class Dataset:
    def __init__(self, dataset):
        if isinstance(dataset, Dataset):
            self._dataset = dataset._dataset
        else:
            self._dataset = dataset

    def __iter__(self):
        yield from self._dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
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
        cache = list(self)
        with open(filename, 'wb') as f:
            pickle.dump(cache, f)
        return CacheDataset(self, cache)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        return Dataset(dataset)


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

    def __getitem__(self, index):
        x = self._dataset[index]
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

        super(MapDataset, self).__init__(dataset)

    def __iter__(self):
        yield from self._cache

    def __getitem__(self, index):
        return self._cache[index]

    def __len__(self):
        return len(self._cache)


class TextDataset(Dataset):
    def __init__(self, filepath, encoding='utf-8'):
        filepath = Path(filepath)
        assert filepath.is_file()

        self._filepath = filepath
        self._encoding = encoding

    def __iter__(self):
        with self._filepath.open(encoding=self._encoding) as f:
            for line in f:
                yield line.rstrip(os.linesep)

    def __getitem__(self, index):
        return linecache.getline(
            str(self._filepath), index + 1).rstrip(os.linesep)

    def __len__(self):
        count = 0
        with self._filepath.open(mode='r+', encoding=self._encoding) as f:
            buf = mmap.mmap(f.fileno(), 0)
            while buf.readline():
                count += 1
        return count

    @property
    def _dataset(self):
        return self
