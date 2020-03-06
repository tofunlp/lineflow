import bisect
import pickle
from _collections_abc import Sequence, _check_methods
from abc import ABCMeta, abstractmethod
from collections import deque
from functools import lru_cache
from itertools import accumulate, chain, islice, tee
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Tuple, Union

import arrayfiles


class DatasetMixin(metaclass=ABCMeta):

    __slots__ = ()

    def __getitem__(self, index: Union[int, slice]) -> Union[Any, List[Any]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in range(start, stop, step)]

        if index >= 0:
            if index >= len(self):
                raise IndexError(f'{self.__class__.__name__} object index out of range')
        else:
            if index < - len(self):
                raise IndexError(f'{self.__class__.__name__} object index out of range')
            index += len(self)

        return self.get_example(index)

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        while False:
            yield None

    @abstractmethod
    def get_example(self, i: int) -> Any:
        raise IndexError

    @abstractmethod
    def __len__(self) -> int:
        return 0

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DatasetMixin:
            return _check_methods(C, '__iter__', 'get_example', '__len__')
        return NotImplemented


DatasetMixin.register(Sequence)
DatasetMixin.register(arrayfiles.TextFile)


class Dataset(DatasetMixin):
    """Dataset wrapping ``DatasetMixin`` object.

    Args:
        dataset (DatasetMixin): ``Sequence``, ``arrayfiles.TextFile``, or ``arrayfiles.CsvFile`` object.
    """

    def __init__(self,
                 dataset: DatasetMixin) -> None:
        assert isinstance(dataset, DatasetMixin)

        self._dataset = dataset
        self._length = None

    def __iter__(self) -> Iterator[Any]:
        yield from self._dataset

    def get_example(self, i: int) -> Any:
        return self._dataset[i]

    def __len__(self) -> int:
        if self._length is None:
            self._length = len(self._dataset)
        return self._length

    def __add__(self, other: 'Dataset') -> 'ConcatDataset':
        return ConcatDataset(self, other)

    def map(self, map_func: Callable[[Any], Any]) -> 'MapDataset':
        """Applies a function across the examples of this dataset.

        Args:
            map_func (Callable[[Any], Any]): A function to apply.

        Returns ('MapDataset'):
            The dataset applied the function.
        """
        return MapDataset(self, map_func)

    def flat_map(self, map_func: Callable[[Any], Any]) -> 'IterableDataset':
        """Applies a function across the examples of this dataset and then flattens the result.

        Args:
            map_func (Callable[[Any], Any]): A function to apply.

        Returns ('IterableDataset'):
            The dataset applied the function and flattened.
        """
        return IterableDataset(lineflow_flat_map(map_func, self, lazy=True))

    def filter(self, predicate: Callable[[Any], bool]) -> 'IterableDataset':
        """Filters this dataset by a predicate function.

        Args:
            predicate (Callable[[Any], bool]): A predicate function.

        Returns ('IterableDataset'):
            The dataset containing the examples for which ``predicate`` returns ``True``.
        """
        return IterableDataset(lineflow_filter(predicate, self, lazy=True))

    def window(self, window_size: int, shift: int = None) -> 'IterableDataset':
        """Combines input examples into a dataset of windows.

        Args:
            window_size (int): the number of examples of the input dataset to combine into a window.
            shift (int, optional): The forward shift of the sliding window in each iteration.

        Returns ('IterableDataset'):
            The dataset of windows.
        """
        return IterableDataset(lineflow_window(self, window_size, shift, lazy=True))

    def all(self) -> List[Any]:
        """Takes all examples from the dataset.

        Returns (List[Any]):
            The list of the examples in the dataset.
        """
        return list(self)

    def take(self, n: int) -> List[Any]:
        """Takes the first n examples from the dataset.

        Args:
            n (int): the number of examples to take.

        Returns (List[Any]):
            The list of the ``n`` examples.
        """
        return list(islice(self, n))

    def first(self) -> Any:
        """Takes the first example from the dataset.

        Returns (Any):
            The first example in the dataset.
        """
        return next(iter(self))

    def save(self, filename: str) -> 'CacheDataset':
        """Evaluates the datasets and save it as pickle.

        Args:
            filename (str): The name of the pickle file.

        Returns ('CacheDataset'):
            The evaluated dataset.
        """
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
        return CacheDataset(cache)


class IterableDataset(Dataset):
    def __init__(self, iterable: Iterable) -> None:
        self._length = None
        self._iterable = iterable
        self._computed = False

    @lru_cache()
    def _get_dataset(self) -> List[Any]:
        self._computed = True
        return list(self._iterable)

    @property
    def _dataset(self) -> List[Any]:
        return self._get_dataset()

    def __iter__(self) -> Iterator[Any]:
        if self._computed:
            yield from self._dataset
        else:
            iterable, self._iterable = tee(self._iterable)
            yield from iterable

    def get_example(self, i: int) -> Any:
        return super(IterableDataset, self).get_example(i)

    def __len__(self) -> int:
        return super(IterableDataset, self).__len__()


class ConcatDataset(Dataset):
    def __init__(self, *datasets: List[DatasetMixin]) -> None:
        assert all(isinstance(d, DatasetMixin) for d in datasets)

        self._datasets = datasets
        self._length = None

    @lru_cache()
    def _get_lengths(self) -> List[int]:
        return list(accumulate(len(d) for d in self._datasets))

    @property
    def _lengths(self) -> List[int]:
        return self._get_lengths()

    @lru_cache()
    def _get_offsets(self) -> List[int]:
        offsets = [0]
        offsets.extend(self._lengths[:-1])
        return offsets

    @property
    def _offsets(self) -> List[int]:
        return self._get_offsets()

    def __iter__(self) -> Iterator[Any]:
        for d in self._datasets:
            yield from d

    def get_example(self, i: int) -> Any:
        j = bisect.bisect_right(self._lengths, i)
        return self._datasets[j][i - self._offsets[j]]

    def __len__(self) -> int:
        if self._length is None:
            self._length = self._lengths[-1]
        return self._length


class ZipDataset(Dataset):
    def __init__(self, *datasets: List[DatasetMixin]) -> None:
        assert all(isinstance(d, DatasetMixin) for d in datasets)
        self._datasets = datasets
        self._length = None

    def __iter__(self) -> Iterator[Tuple[Any]]:
        yield from zip(*self._datasets)

    def get_example(self, i: int) -> Tuple[Any]:
        return tuple(d[i] for d in self._datasets)

    def __len__(self) -> int:
        if self._length is None:
            self._length = min(len(d) for d in self._datasets)
        return self._length


class MapDataset(Dataset):
    def __init__(self,
                 dataset: DatasetMixin,
                 map_func: Callable[[Any], Any]) -> None:
        assert callable(map_func)

        self._map_func = map_func

        super(MapDataset, self).__init__(dataset)

    def __iter__(self) -> Iterator[Any]:
        yield from map(self._map_func, self._dataset)

    def get_example(self, i: int) -> Any:
        return self._map_func(self._dataset[i])


class CacheDataset(Dataset):
    def __init__(self, cache: List[Any]) -> None:
        super(CacheDataset, self).__init__(cache)

        self._length = len(cache)


def lineflow_concat(*datasets: List[DatasetMixin]) -> ConcatDataset:
    return ConcatDataset(*datasets)


def lineflow_zip(*datasets: List[DatasetMixin]) -> ZipDataset:
    return ZipDataset(*datasets)


def lineflow_filter(
        predicate: Callable[[Any], bool],
        dataset: DatasetMixin,
        lazy: bool = False) -> Union[Iterator[Any], List[Any]]:
    iterator = filter(predicate, dataset)
    if lazy:
        return iterator
    else:
        return list(iterator)


def lineflow_flat_map(
        map_func: Callable[[Iterable[Any]], Any],
        dataset: DatasetMixin,
        lazy: bool = False) -> Union[Iterator[Any], List[Any]]:
    iterator = chain.from_iterable(map(map_func, dataset))
    if lazy:
        return iterator
    else:
        return list(iterator)


def lineflow_window(
        dataset: DatasetMixin,
        window_size: int,
        shift: int = None,
        lazy: bool = False) -> Union[Iterator[Any], List[Any]]:
    shift = shift or window_size

    def generator(dataset, window_size, shift):
        iterator = iter(dataset)
        window = deque([], window_size)
        append = window.append

        for _, x in zip(range(window_size), iterator):
            append(x)
        yield tuple(window)

        i = 0
        for x in iterator:
            append(x)
            i = (i + 1) % shift
            if i % shift == 0:
                yield tuple(window)

        if (i % shift) and (shift - i < window_size):
            popleft = window.popleft
            for _ in range(shift - i):
                popleft()
            yield tuple(window)

    iterator = generator(dataset, window_size, shift)
    if lazy:
        return iterator
    else:
        return list(iterator)


def lineflow_load(filename: str) -> Dataset:
    print(f'Loading data from {filename}...')
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return Dataset(dataset)
