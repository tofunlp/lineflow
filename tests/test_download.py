import os
import shutil
import tempfile
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

    @mock.patch('os.makedirs')
    def test_fails_to_make_directory(self, f):
        f.side_effect = OSError()
        with self.assertRaises(OSError):
            download.get_cache_directory('/lineflow_test_cache', True)


class TestCacheOrLoadFile(unittest.TestCase):

    def setUp(self):
        self.default_cache_root = download.get_cache_root()
        self.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(self.temp_dir)

    def tearDown(self):
        download.set_cache_root(self.default_cache_root)
        shutil.rmtree(self.temp_dir)

    def test_cache_exists(self):
        creator = mock.Mock()
        loader = mock.Mock()

        file_desc, file_name = tempfile.mkstemp()

        try:
            download.cache_or_load_file(file_name, creator, loader)
        finally:
            os.close(file_desc)
            os.remove(file_name)

        self.assertFalse(creator.called)
        loader.assert_called_once_with(file_name)

    def test_new_file(self):
        def create(path):
            with open(path, 'w') as f:
                f.write('test')

        creator = mock.Mock()
        creator.side_effect = create
        loader = mock.Mock()

        dir_path = tempfile.mkdtemp()
        # This file always does not exists as the directory is new.
        path = os.path.join(dir_path, 'cache')

        try:
            download.cache_or_load_file(path, creator, loader)

            self.assertEqual(creator.call_count, 1)
            self.assertFalse(loader.called)

            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                self.assertEqual(f.read(), 'test')

        finally:
            shutil.rmtree(dir_path)


class TestCacheOrLoadFileFileExists(unittest.TestCase):

    def setUp(self):
        self.default_cache_root = download.get_cache_root()
        self.temp_file_desc, self.temp_file_name = tempfile.mkstemp()
        download.set_cache_root(self.temp_file_name)
        self.dir_path = tempfile.mkdtemp()

    def tearDown(self):
        download.set_cache_root(self.default_cache_root)
        os.close(self.temp_file_desc)
        os.remove(self.temp_file_name)
        shutil.rmtree(self.dir_path)

    def test_file_exists(self):
        creator = mock.Mock()
        loader = mock.Mock()

        # This file always does not exists as the directory is new.
        path = os.path.join(self.dir_path, 'cache')

        with self.assertRaises(RuntimeError):
            download.cache_or_load_file(path, creator, loader)
