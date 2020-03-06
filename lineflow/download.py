import contextlib
import os
import shutil
import tempfile

_cache_root = os.environ.get(
    'LINEFLOW_ROOT',
    os.path.join(os.path.expanduser('~'), '.cache', 'lineflow'))


@contextlib.contextmanager
def tempdir(**kwargs):
    ignore_errors = kwargs.pop('ignore_errors', False)

    temp_dir = tempfile.mkdtemp(**kwargs)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=ignore_errors)


def get_cache_root() -> str:
    return _cache_root


def set_cache_root(path: str) -> None:
    global _cache_root
    _cache_root = path


def get_cache_directory(cache_name: str,
                        create_directory: bool = True) -> str:
    path = os.path.join(_cache_root, cache_name)
    if create_directory:
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


def cache_or_load_file(path, creator, loader):
    if os.path.exists(path):
        return loader(path)

    try:
        os.makedirs(_cache_root)
    except OSError:
        if not os.path.isdir(_cache_root):
            raise RuntimeError('cannot create cache directory')

    with tempdir() as temp_dir:
        filename = os.path.basename(path)
        temp_path = os.path.join(temp_dir, filename)
        content = creator(temp_path)
        if not os.path.exists(path):
            shutil.move(temp_path, path)

    return content
