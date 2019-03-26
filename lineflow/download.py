import hashlib
import os
import shutil
import sys
from urllib import request

from .utils import tempdir


_cache_root = os.environ.get(
    'LINEFLOW_CACHE_ROOT',
    os.path.join(os.path.expanduser('~'), '.lineflow', 'cache'))


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


def cached_download(url: str) -> str:
    cache_root = os.path.join(_cache_root, '_dl_cache')
    try:
        os.makedirs(cache_root)
    except OSError:
        if not os.path.isdir(cache_root):
            raise

    urlhash = hashlib.md5(url.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_root, urlhash)

    if os.path.exists(cache_path):
        return cache_path

    with tempdir(dir=cache_root) as temp_root:
        temp_path = os.path.join(temp_root, 'dl')
        sys.stderr.write('Downloading from {}...\n'.format(url))
        sys.stderr.flush()
        request.urlretrieve(url, temp_path)
        shutil.move(temp_path, cache_path)

    return cache_path
