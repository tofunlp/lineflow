from typing import Sequence, Any, Union, Callable, List, Tuple, Iterator, Iterable
import warnings
import pickle
import copy
from pathlib import Path
from itertools import accumulate, chain, islice
import bisect


class RandomAccessConcat:
    def __init__(self, *datasets: List[Sequence[Any]]) -> None:
        self._datasets = datasets
        self._offsets = None
        self._length = None

    def _initialize_offsets(self) -> None:
        self._lengths = list(accumulate(len(d) for d in self._datasets))
        self._offsets = [0] + self._lengths[:-1]

    def __iter__(self) -> Iterator[Any]:
        for d in self._datasets:
            yield from d

    def __getitem__(self, index: int) -> Any:
        if self._offsets is None:
            self._initialize_offsets()
        if index < 0 or len(self) <= index:
            raise IndexError('RandomAccessConcat object index out of range')
        j = bisect.bisect_right(self._lengths, index)
        return self._datasets[j][index - self._offsets[j]]

    def __len__(self) -> int:
        if self._offsets is None:
            self._initialize_offsets()
        if self._length is None:
            self._length = self._lengths[-1]
        return self._length


class RandomAccessZip:
    def __init__(self, *datasets: List[Sequence[Any]]) -> None:
        self._datasets = datasets
        self._length = None

    def __iter__(self) -> Iterator[Tuple[Any]]:
        yield from zip(*self._datasets)

    def __getitem__(self, index: int) -> Tuple[Any]:
        if index < 0 or len(self) <= index:
            raise IndexError('RandomAccessZip object index out of range')
        return tuple(d[index] for d in self._datasets)

    def __len__(self) -> int:
        if self._length is None:
            self._length = min(len(d) for d in self._datasets)
        return self._length


class Dataset:
    def __init__(self,
                 dataset: Sequence[Any]) -> None:
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

        super().__init__(RandomAccessConcat(*datasets))


class ZipDataset(Dataset):
    def __init__(self, *datasets: List[Dataset]) -> None:
        assert all(isinstance(d, Dataset) for d in datasets)

        super().__init__(RandomAccessZip(*datasets))


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

        if isinstance(dataset, Dataset):
            dataset = dataset._dataset

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
