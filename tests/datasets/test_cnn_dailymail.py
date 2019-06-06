from unittest import TestCase
from unittest.mock import patch
import tempfile
from pathlib import Path


from lineflow.download import get_cache_directory
from lineflow.datasets import CnnDailymail
from lineflow.datasets.cnn_dailymail import CNN_DAILYMAIL_URL
from lineflow.datasets.cnn_dailymail import TRAIN_SOURCE_NAME, TRAIN_TARGET_NAME
from lineflow.datasets.cnn_dailymail import VAL_SOURCE_NAME, VAL_TARGET_NAME
from lineflow.datasets.cnn_dailymail import TEST_SOURCE_NAME, TEST_TARGET_NAME


class CnnDailymailTestCase(TestCase):

    def setUp(self):
        cache_fp = tempfile.NamedTemporaryFile()
        self.cache_fp = cache_fp

        cached_download_patcher = patch('lineflow.datasets.cnn_dailymail.cached_download')
        cached_download_mock = cached_download_patcher.start()
        cached_download_mock.side_effect = lambda url: cache_fp.name
        self.cached_download_patcher = cached_download_patcher
        self.cached_download_mock = cached_download_mock

        tarfile_patcher = patch('lineflow.datasets.cnn_dailymail.tarfile')
        tarfile_mock = tarfile_patcher.start()
        self.tarfile_patcher = tarfile_patcher
        self.tarfile_mock = tarfile_mock

        exists_patcher = patch('lineflow.datasets.cnn_dailymail.Path.exists')
        exists_mock = exists_patcher.start()
        exists_mock.return_value = True
        self.exists_patcher = exists_patcher
        self.exists_mock = exists_mock

        init_patcher = patch('lineflow.datasets.seq2seq.Seq2SeqDataset.__init__')
        init_mock = init_patcher.start()
        self.init_patcher = init_patcher
        self.init_mock = init_mock

        self.cache_dir = Path(get_cache_directory('cnndm'))

    def tearDown(self):
        self.cache_fp.close()
        self.cached_download_patcher.stop()
        self.tarfile_patcher.stop()
        self.exists_patcher.stop()
        self.init_patcher.stop()

    def test_returns_train_set(self):
        CnnDailymail(split='train')
        self.cached_download_mock.assert_called_once_with(CNN_DAILYMAIL_URL)
        self.init_mock.assert_called_once_with(
            source_file_path=str(self.cache_dir / TRAIN_SOURCE_NAME),
            target_file_path=str(self.cache_dir / TRAIN_TARGET_NAME))

    def test_returns_dev_set(self):
        CnnDailymail(split='dev')
        self.cached_download_mock.assert_called_once_with(CNN_DAILYMAIL_URL)
        self.init_mock.assert_called_once_with(
            source_file_path=str(self.cache_dir / VAL_SOURCE_NAME),
            target_file_path=str(self.cache_dir / VAL_TARGET_NAME))

    def test_returns_test_set(self):
        CnnDailymail(split='test')
        self.cached_download_mock.assert_called_once_with(CNN_DAILYMAIL_URL)
        self.init_mock.assert_called_once_with(
            source_file_path=str(self.cache_dir / TEST_SOURCE_NAME),
            target_file_path=str(self.cache_dir / TEST_TARGET_NAME))

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            CnnDailymail(split='invalid_split')

    def test_expands_tarfile(self):
        self.exists_mock.return_value = False
        CnnDailymail(split='train')
        self.tarfile_mock.open.return_value.extractall.assert_called_once_with(self.cache_dir)
