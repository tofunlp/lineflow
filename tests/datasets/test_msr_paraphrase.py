from unittest import TestCase
from unittest.mock import patch
import tempfile


from lineflow.datasets import MsrParaphrase
from lineflow.datasets.msr_paraphrase import TRAIN_URL, TEST_URL


class MsrParaphraseTestCase(TestCase):

    def setUp(self):
        fp = tempfile.NamedTemporaryFile()
        fp.write(b'header1\theader2\theader3\theader4\theader5')
        fp.seek(0)
        self.fp = fp

        cached_download_patcher = patch('lineflow.datasets.msr_paraphrase.cached_download')
        cached_download_mock = cached_download_patcher.start()
        cached_download_mock.side_effect = lambda url: fp.name

        self.cached_download_patcher = cached_download_patcher
        self.cached_download_mock = cached_download_mock

    def tearDown(self):
        self.fp.close()
        self.cached_download_patcher.stop()

    def test_returns_train_set(self):
        MsrParaphrase(split='train')
        self.cached_download_mock.assert_called_once_with(TRAIN_URL)

    def test_returns_test_set(self):
        MsrParaphrase(split='test')
        self.cached_download_mock.assert_called_once_with(TEST_URL)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            MsrParaphrase(split='invalid_split')
