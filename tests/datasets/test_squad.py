from unittest import TestCase
from unittest.mock import patch
import tempfile


from lineflow.datasets import Squad
from lineflow.datasets.squad import TRAIN_V1_URL, DEV_V1_URL, TRAIN_V2_URL, DEV_V2_URL


class SquadTestCase(TestCase):

    def setUp(self):
        fp = tempfile.NamedTemporaryFile()
        self.fp = fp

        cached_download_patcher = patch('lineflow.datasets.squad.cached_download')
        cached_download_mock = cached_download_patcher.start()
        cached_download_mock.side_effect = lambda url: fp.name

        self.cached_download_patcher = cached_download_patcher
        self.cached_download_mock = cached_download_mock

    def tearDown(self):
        self.fp.close()
        self.cached_download_patcher.stop()

    def test_returns_train_set_v1(self):
        Squad(split='train', version=1)
        self.cached_download_mock.assert_called_once_with(TRAIN_V1_URL)

    def test_returns_dev_set_v1(self):
        Squad(split='dev', version=1)
        self.cached_download_mock.assert_called_once_with(DEV_V1_URL)

    def test_returns_train_set_v2(self):
        Squad(split='train', version=2)
        self.cached_download_mock.assert_called_once_with(TRAIN_V2_URL)

    def test_returns_dev_set_v2(self):
        Squad(split='dev', version=2)
        self.cached_download_mock.assert_called_once_with(DEV_V2_URL)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            Squad(split='invalid_split')

    def test_raises_value_error_with_invalid_version(self):
        with self.assertRaises(ValueError):
            Squad(version=3)
