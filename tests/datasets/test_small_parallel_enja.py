from unittest import TestCase
from unittest.mock import patch
import tempfile


from lineflow.datasets import SmallParallelEnJa
from lineflow.datasets.small_parallel_enja import TRAIN_EN_URL, TRAIN_JA_URL
from lineflow.datasets.small_parallel_enja import DEV_EN_URL, DEV_JA_URL
from lineflow.datasets.small_parallel_enja import TEST_EN_URL, TEST_JA_URL


class SmallParallelEnJaTestCase(TestCase):

    def setUp(self):
        en_fp = tempfile.NamedTemporaryFile()
        en_fp.write(b'This is English .')
        en_fp.seek(0)
        ja_fp = tempfile.NamedTemporaryFile()
        ja_fp.write(b'This is Japanese .')
        ja_fp.seek(0)
        self.en_fp = en_fp
        self.ja_fp = ja_fp

        cached_download_patcher = patch('lineflow.datasets.small_parallel_enja.cached_download')
        cached_download_mock = cached_download_patcher.start()
        cached_download_mock.side_effect = lambda url: en_fp.name if '.en' in url else ja_fp.name

        self.cached_download_patcher = cached_download_patcher
        self.cached_download_mock = cached_download_mock

    def tearDown(self):
        self.en_fp.close()
        self.ja_fp.close()
        self.cached_download_patcher.stop()

    def test_returns_train_set(self):
        train = SmallParallelEnJa(split='train')
        expected = [((url,),) for url in (TRAIN_EN_URL, TRAIN_JA_URL)]
        self.assertListEqual(self.cached_download_mock.call_args_list, expected)
        self.assertEqual(len(train), 1)

    def test_returns_dev_set(self):
        dev = SmallParallelEnJa(split='dev')
        expected = [((url,),) for url in (DEV_EN_URL, DEV_JA_URL)]
        self.assertListEqual(self.cached_download_mock.call_args_list, expected)
        self.assertEqual(len(dev), 1)

    def test_returns_test_set(self):
        test = SmallParallelEnJa(split='test')
        expected = [((url,),) for url in (TEST_EN_URL, TEST_JA_URL)]
        self.assertListEqual(self.cached_download_mock.call_args_list, expected)
        self.assertEqual(len(test), 1)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            SmallParallelEnJa(split='invalid_split')
