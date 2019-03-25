from unittest import TestCase
from unittest.mock import patch
import tempfile
from pathlib import Path


from lineflow.download import get_cache_root
from lineflow.datasets import Imdb
from lineflow.datasets.imdb import IMDB_URL


class ImdbTestCase(TestCase):

    def setUp(self):
        cache_fp = tempfile.NamedTemporaryFile()
        self.cache_fp = cache_fp

        cached_download_patcher = patch('lineflow.datasets.imdb.cached_download')
        cached_download_mock = cached_download_patcher.start()
        cached_download_mock.side_effect = lambda url: cache_fp.name
        self.cached_download_patcher = cached_download_patcher
        self.cached_download_mock = cached_download_mock

        tarfile_patcher = patch('lineflow.datasets.imdb.tarfile')
        tarfile_mock = tarfile_patcher.start()
        self.tarfile_patcher = tarfile_patcher
        self.tarfile_mock = tarfile_mock

        exists_patcher = patch('lineflow.datasets.imdb.Path.exists')
        exists_mock = exists_patcher.start()
        exists_mock.return_value = True
        self.exists_patcher = exists_patcher
        self.exists_mock = exists_mock

        self.cache_dir = Path(get_cache_root())

        data_fp = tempfile.NamedTemporaryFile()
        self.data_fp = data_fp
        glob_patcher = patch('lineflow.datasets.imdb.Path.glob')
        glob_mock = glob_patcher.start()
        glob_mock.return_value = [Path(data_fp.name)]
        self.glob_patcher = glob_patcher
        self.glob_mock = glob_mock

    def tearDown(self):
        self.cache_fp.close()
        self.cached_download_patcher.stop()
        self.tarfile_patcher.stop()
        self.exists_patcher.stop()
        self.data_fp.close()
        self.glob_patcher.stop()

    def test_returns_train_set(self):
        ds = Imdb(split='train')
        self.cached_download_mock.assert_called_once_with(IMDB_URL)
        self.glob_mock.assert_called_with('*.txt')
        self.assertEqual(self.glob_mock.call_count, 2)
        expected = ('', 1)
        for x in ds:
            self.assertTupleEqual(x, expected)

    def test_returns_test_set(self):
        ds = Imdb(split='test')
        self.cached_download_mock.assert_called_once_with(IMDB_URL)
        self.glob_mock.assert_called_with('*.txt')
        self.assertEqual(self.glob_mock.call_count, 2)
        expected = ('', 1)
        for x in ds:
            self.assertTupleEqual(x, expected)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            Imdb(split='invalid_split')

    def test_expands_tarfile(self):
        self.exists_mock.return_value = False
        Imdb(split='train')
        self.tarfile_mock.open.return_value.extractall.assert_called_once_with(self.cache_dir)
