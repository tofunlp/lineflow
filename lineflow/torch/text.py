import os
import io

from torch.utils.data import get_worker_info

from lineflow.torch import Dataset


class TextDataset(Dataset):
    def __init__(self, path, encoding='utf-8'):
        self._path = path
        self._encoding = encoding
        self._total_size = os.stat(path).st_size

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            with io.open(self._path, 'r', encoding=self._encoding) as fp:
                for line in fp:
                    yield line.rstrip(os.linesep)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            fp, end = self._read_block(worker_id, num_workers)
            for line in fp:
                yield line.decode(self._encoding).rstrip(os.linesep)
                if fp.tell() >= end:
                    break
            fp.close()

    def _read_block(self, worker_id, num_workers):
        chunk_size = self._total_size // num_workers
        fp = io.open(self._path, 'rb')
        # end position
        if (worker_id + 1) == num_workers:
            end = self._total_size
        else:
            offset = chunk_size * (worker_id + 1)
            fp.seek(offset)
            end = offset + len(fp.readline())
        # start position
        if worker_id == 0:
            fp.seek(0)
        else:
            offset = chunk_size * worker_id
            fp.seek(offset)
            fp.readline()
        return fp, end
