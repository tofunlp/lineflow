import random
import itertools

from torch.utils.data import IterableDataset


class Dataset(IterableDataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __iter__(self):
        yield from self._dataset

    def all(self):
        return list(self)

    def first(self):
        return next(iter(self))

    def take(self, n):
        return list(itertools.islice(self, n))

    def map(self, map_func):
        return MapDataset(self, map_func)

    def flat_map(self, map_func):
        return FlatMapDataset(self, map_func)

    def filter(self, predicate):
        return FilterDataset(self, predicate)

    def shuffle(self, buffer_size=None):
        return ShuffleDataset(self, buffer_size)


class MapDataset(Dataset):
    def __init__(self, dataset, map_func):
        assert callable(map_func)

        self._dataset = dataset
        self._map_func = map_func

    def __iter__(self):
        yield from map(self._map_func, self._dataset)


class FlatMapDataset(Dataset):
    def __init__(self, dataset, map_func):
        assert callable(map_func)

        self._dataset = dataset
        self._map_func = map_func

    def __iter__(self):
        yield from itertools.chain.from_iterable(map(self._map_func, self._dataset))


class FilterDataset(Dataset):
    def __init__(self, dataset, predicate):
        assert callable(predicate)

        self._dataset = dataset
        self._predicate = predicate

    def __iter__(self):
        yield from filter(self._predicate, self._dataset)


class ShuffleDataset(Dataset):
    def __init__(self, dataset, buffer_size=None):
        self._dataset = dataset
        self._buffer_size = buffer_size

    def __iter__(self):
        chunk = []

        if self._buffer_size is None:
            for x in self._dataset:
                chunk.append(x)
            random.shuffle(chunk)
            yield from chunk
            return

        for x in self._dataset:
            chunk.append(x)
            if len(chunk) >= self._buffer_size:
                random.shuffle(chunk)
                yield from chunk
                chunk = []
        if chunk:
            random.shuffle(chunk)
            yield from chunk
