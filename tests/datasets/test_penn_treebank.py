from unittest import TestCase
from unittest.mock import patch
import tempfile


from lineflow.datasets import PennTreebank
from lineflow.datasets.penn_treebank import TRAIN_URL, DEV_URL, TEST_URL


class PennTreebankTestCase(TestCase):

    def setUp(self):
        fp = tempfile.NamedTemporaryFile()
        self.fp = fp

        cached_download_patcher = patch('lineflow.datasets.penn_treebank.cached_download')
        cached_download_mock = cached_download_patcher.start()
        cached_download_mock.side_effect = lambda url: fp.name

        self.cached_download_patcher = cached_download_patcher
        self.cached_download_mock = cached_download_mock

    def tearDown(self):
        self.fp.close()
        self.cached_download_patcher.stop()

    def test_returns_train_set(self):
        PennTreebank(split='train')
        self.cached_download_mock.assert_called_once_with(TRAIN_URL)

    def test_returns_valid_set(self):
        PennTreebank(split='dev')
        self.cached_download_mock.assert_called_once_with(DEV_URL)

    def test_returns_test_set(self):
        PennTreebank(split='test')
        self.cached_download_mock.assert_called_once_with(TEST_URL)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            PennTreebank(split='invalid_split')
