from typing import Union, Callable, Tuple, List, Dict
from functools import wraps


class MapFunction:
    def __init__(self,
                 key: Union[int, str],
                 func: Callable[[Union[Tuple, List, Dict]], Union[Tuple, List, Dict]]) -> None:
        self._queue = [key]
        self._func = func

    def append(self, key: Union[int, str]) -> None:
        self._queue.append(key)

    def __call__(self, x: Union[Tuple, List, Dict]) -> Union[Tuple, List, Dict]:
        for key in self._queue:
            if isinstance(x, tuple):
                x = list(x)
                x[key] = self._func(x[key])
                x = tuple(x)
            elif isinstance(x, (dict, list)):
                x[key] = self._func(x[key])
            else:
                raise TypeError(f'Passed argument should be tuple, list or dict',
                                'but {type(x)} is passed.')
        return x


def apply(key: Union[int, str]) -> Callable[[Callable], Callable]:
    def decorator(
        func: Callable[[Union[Tuple, List, Dict]], Union[Tuple, List, Dict]]
    ) -> Callable[[Union[Tuple, List, Dict]], Union[Tuple, List, Dict]]:
        if isinstance(func, MapFunction):
            func.append(key)
        else:
            func = wraps(func)(MapFunction(key, func))
        return func
    return decorator


def apply_all(*ignores: List[Union[int, str]]) -> Callable[[Callable], Callable]:
    ignores = set(ignores) if ignores else {}

    def decorator(
        func: Callable[[Union[Tuple, List, Dict]], Union[Tuple, List, Dict]]
    ) -> Callable[[Union[Tuple, List, Dict]], Union[Tuple, List, Dict]]:
        if isinstance(func, MapFunction):
            raise ValueError('lineflow.apply_all cannot use with lineflow.apply.')

        @wraps(func)
        def wrapper(x):
            if isinstance(x, tuple):
                x = tuple(func(item) if i not in ignores else item for i, item in enumerate(x))
            elif isinstance(x, list):
                x = [func(item) if i not in ignores else item for i, item in enumerate(x)]
            elif isinstance(x, dict):
                x = {k: func(v) if k not in ignores else v for k, v in x.items()}
            else:
                raise TypeError(f'Passed argument should be tuple, list or dict',
                                'but {type(x)} is passed.')
            return x
        return wrapper
    return decorator
