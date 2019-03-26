import os
import shutil
import tempfile
import hashlib
import unittest
from unittest import mock


from lineflow import download


class TestSetCacheRoot(unittest.TestCase):

    def test_set_cache_root(self):
        orig_root = download.get_cache_root()
        new_root = '/tmp/cache'
        try:
            download.set_cache_root(new_root)
            self.assertEqual(download.get_cache_root(), new_root)
        finally:
            download.set_cache_root(orig_root)


class TestGetCacheDirectory(unittest.TestCase):

    def test_get_cache_directory(self):
        root = download.get_cache_root()
        path = download.get_cache_directory('test', False)
        self.assertEqual(path, os.path.join(root, 'test'))

    def test_fails_to_make_directory(self):
        with mock.patch('os.makedirs') as f:
            f.side_effect = OSError()
            with self.assertRaises(OSError):
                download.get_cache_directory('/lineflow_test_cache', True)


class TestCachedDownload(unittest.TestCase):

    def setUp(self):
        self.default_cache_root = download.get_cache_root()
        self.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(self.temp_dir)

    def tearDown(self):
        download.set_cache_root(self.default_cache_root)
        shutil.rmtree(self.temp_dir)

    def test_fails_to_make_directory(self):
        with mock.patch('os.makedirs') as f:
            f.side_effect = OSError()
            with self.assertRaises(OSError):
                download.cached_download('https://example.com')

    def test_file_exists(self):
        # Make an empty file which has the same name as the cache directory
        with open(os.path.join(self.temp_dir, '_dl_cache'), 'w'):
            pass
        with self.assertRaises(OSError):
            download.cached_download('https://example.com')

    def test_cache_exists(self):
        with mock.patch('os.path.exists') as f:
            f.return_value = True
            url = 'https://example.com'
            path = download.cached_download(url)
            self.assertEqual(path, f'{self.temp_dir}/_dl_cache/{hashlib.md5(url.encode("utf-8")).hexdigest()}')

    def test_cached_download(self):
        with mock.patch('urllib.request.urlretrieve') as f:
            def urlretrieve(url, path):
                with open(path, 'w') as f:
                    f.write('test')
            f.side_effect = urlretrieve

            cache_path = download.cached_download('https://example.com')

        self.assertEqual(f.call_count, 1)
        args, kwargs = f.call_args
        self.assertEqual(kwargs, {})
        self.assertEqual(len(args), 2)
        # The second argument is a temporary path, and it is removed
        self.assertEqual(args[0], 'https://example.com')

        self.assertTrue(os.path.exists(cache_path))
        with open(cache_path) as f:
            stored_data = f.read()
        self.assertEqual(stored_data, 'test')
