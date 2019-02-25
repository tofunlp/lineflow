from itertools import chain


class Reduced:
    def __init__(self, iterable):
        self._iterable = iterable
        self._cache = []
        self._done = False

    def __iter__(self):
        if not self._done:
            for x in self._iterable:
                yield x
                self._cache.append(x)
            self._done = True
        else:
            yield from self._cache


class Reducer:
    def __call__(self, dataset):
        raise NotImplementedError


class Filter(Reducer):
    def __init__(self, predicate):
        self._predicate = predicate

    def __call__(self, dataset):
        return Reduced(filter(self._predicate, dataset))


class FlatMap(Reducer):
    def __init__(self, map_func):
        self._map_func = map_func

    def __call__(self, dataset):
        return Reduced(
            chain.from_iterable(map(self._map_func, dataset)))


class Concat(Reducer):
    def __call__(self, *dataset):
        return Reduced(chain(*dataset))


class Zip(Reducer):
    def __call__(self, *dataset):
        return Reduced(zip(*dataset))
