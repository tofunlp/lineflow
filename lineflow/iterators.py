import threading
import queue


class PrefetchIterator:
    def __init__(self, dataset, n_prefetch=1):
        self._dataset = dataset
        self._queue = queue.Queue(maxsize=n_prefetch)
        self._thread = self._launch_thread()

    def _launch_thread(self):
        thread = threading.Thread(target=self._task,
                                  args=(self._dataset, self._queue))
        thread.daemon = True
        thread.start()
        return thread

    @staticmethod
    def _task(dataset, queue):
        for x in dataset:
            queue.put(x)
        queue.put(StopIteration)

    def __iter__(self):
        return self

    def __next__(self):
        if self._thread is None:
            self._thread = self._launch_thread()

        x = self._queue.get()

        if x is StopIteration:
            self._thread = None
            raise x
        else:
            return x
