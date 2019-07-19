from unittest import TestCase
from unittest.mock import patch
import tempfile


from lineflow.datasets import WikiText103
from lineflow.datasets.wikitext103 import WIKITEXT103_URL
from lineflow.datasets.wikitext103 import TRAIN_PATH, DEV_PATH, TEST_PATH
from lineflow.download import get_cache_root


class WikiText103TestCase(TestCase):

    def setUp(self):
        fp = tempfile.NamedTemporaryFile()
        self.fp = fp

        cached_download_patcher = patch('lineflow.datasets.wikitext103.cached_download')
        cached_download_mock = cached_download_patcher.start()
        cached_download_mock.side_effect = lambda url: fp.name
        self.cached_download_patcher = cached_download_patcher
        self.cached_download_mock = cached_download_mock

        zipfile_patcher = patch('lineflow.datasets.wikitext103.zipfile')
        zipfile_mock = zipfile_patcher.start()
        self.zipfile_patcher = zipfile_patcher
        self.zipfile_mock = zipfile_mock

        exists_patcher = patch('lineflow.datasets.wikitext103.os.path.exists')
        exists_mock = exists_patcher.start()
        exists_mock.return_value = False
        self.exists_patcher = exists_patcher
        self.exists_mock = exists_mock

        dunder_init_patcher = patch('lineflow.text.TextDataset.__init__')
        dunder_init_mock = dunder_init_patcher.start()
        self.dunder_init_patcher = dunder_init_patcher
        self.dunder_init_mock = dunder_init_mock

    def tearDown(self):
        self.fp.close()
        self.cached_download_patcher.stop()
        self.zipfile_patcher.stop()
        self.exists_patcher.stop()
        self.dunder_init_patcher.stop()

    def test_returns_train_set(self):
        WikiText103(split='train')
        self.cached_download_mock.assert_called_once_with(WIKITEXT103_URL)
        self.zipfile_mock.ZipFile.assert_called_once_with(self.fp.name, 'r')
        self.zipfile_mock.ZipFile.return_value.extractall.assert_called_once_with(get_cache_root())
        self.dunder_init_mock.assert_called_once_with(TRAIN_PATH)

    def test_returns_dev_set(self):
        WikiText103(split='dev')
        self.cached_download_mock.assert_called_once_with(WIKITEXT103_URL)
        self.zipfile_mock.ZipFile.assert_called_once_with(self.fp.name, 'r')
        self.zipfile_mock.ZipFile.return_value.extractall.assert_called_once_with(get_cache_root())
        self.dunder_init_mock.assert_called_once_with(DEV_PATH)

    def test_returns_test_set(self):
        WikiText103(split='test')
        self.cached_download_mock.assert_called_once_with(WIKITEXT103_URL)
        self.zipfile_mock.ZipFile.assert_called_once_with(self.fp.name, 'r')
        self.zipfile_mock.ZipFile.return_value.extractall.assert_called_once_with(get_cache_root())
        self.dunder_init_mock.assert_called_once_with(TEST_PATH)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            WikiText103(split='invalid_split')
