from typing import Iterator
import os
import io

from torch.utils.data import get_worker_info

from lineflow.torch import Dataset


class TextDataset(Dataset):
    def __init__(self, path: str, encoding: str = 'utf-8') -> None:
        assert os.path.exists(path)

        self._path = path
        self._encoding = encoding

    def __iter__(self) -> Iterator[str]:
        worker_info = get_worker_info()
        if worker_info is None:
            with io.open(self._path, 'r', encoding=self._encoding) as fp:
                for line in fp:
                    yield line.rstrip(os.linesep)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            with io.open(self._path, 'rb') as fp:
                for i, line in enumerate(fp):
                    if i % num_workers == worker_id:
                        yield line.decode(self._encoding).rstrip(os.linesep)
