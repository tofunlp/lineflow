from typing import List, Tuple, Any, Iterator
import random

from lineflow import Dataset


class SubDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 start: int,
                 end: int,
                 indices: List[int] = None) -> None:
        if start < 0 or end > len(dataset):
            raise ValueError('subset overruns the base dataset.')
        self._dataset = dataset
        self._start = start
        self._end = end
        self._size = end - start
        if indices is not None and len(indices) != len(dataset):
            msg = (f'indices option must have the same length as the base '
                   'dataset: len(indices) = {len(indices)} while len(dataset) = {len(dataset)}')
            raise ValueError(msg)
        self._indices = indices or list(range(len(dataset)))

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        for index in self._indices[self._start: self._end]:
            yield self._dataset[index]

    def get_example(self, i: int) -> Any:
        return self._dataset[self._indices[self._start + i]]


def split_dataset(dataset: Dataset,
                  split_at: int,
                  indices: List[int] = None) -> Tuple[SubDataset]:
    n_examples = len(dataset)
    if not isinstance(split_at, int):
        raise TypeError(f'split_at must be int, got {type(split_at)} instead')
    if split_at < 0:
        raise ValueError('split_at must be non-negative')
    if split_at > n_examples:
        raise ValueError('split_at exceeds the dataset size')
    subset1 = SubDataset(dataset, 0, split_at, indices)
    subset2 = SubDataset(dataset, split_at, n_examples, indices)
    return subset1, subset2


def split_dataset_random(dataset: Dataset,
                         first_size: int,
                         seed=None) -> Tuple[SubDataset]:
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return split_dataset(dataset, first_size, indices)


def split_dataset_n(dataset: Dataset,
                    n: int,
                    indices: List[int] = None) -> List[SubDataset]:
    n_examples = len(dataset)
    sub_size = n_examples // n
    return [SubDataset(dataset, sub_size * i, sub_size * (i + 1), indices)
            for i in range(n)]


def split_dataset_n_random(dataset: Dataset,
                           n: int,
                           seed=None) -> List[SubDataset]:
    n_examples = len(dataset)
    sub_size = n_examples // n
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return [SubDataset(dataset, sub_size * i, sub_size * (i + 1), indices)
            for i in range(n)]


def get_cross_validation_datasets(dataset: Dataset,
                                  n_fold: int,
                                  indices: List[int] = None) -> List[Tuple[SubDataset]]:
    if indices is None:
        indices = list(range(len(dataset)))
    else:
        indices = indices.copy()

    whole_size = len(dataset)
    borders = [whole_size * i // n_fold for i in range(n_fold + 1)]
    test_sizes = [borders[i + 1] - borders[i] for i in range(n_fold)]

    splits = []
    for test_size in reversed(test_sizes):
        size = whole_size - test_size
        splits.append(split_dataset(dataset, size, indices))
        new_indices = [None] * len(indices)
        new_indices[:test_size] = indices[-test_size:]
        new_indices[test_size:] = indices[:-test_size]
        indices = new_indices

    return splits


def get_cross_validation_datasets_random(dataset: Dataset,
                                         n_fold: int,
                                         seed=None) -> List[Tuple[SubDataset]]:
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return get_cross_validation_datasets(dataset, n_fold, indices)
