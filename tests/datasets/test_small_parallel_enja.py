import tempfile
import shutil
from unittest import TestCase
from unittest import mock

from lineflow import download
from lineflow.datasets.small_parallel_enja import SmallParallelEnJa, get_small_parallel_enja


class SmallParallelEnJaTestCase(TestCase):

    def setUp(self):
        self.default_cache_root = download.get_cache_root()
        self.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(self.temp_dir)

    def tearDown(self):
        download.set_cache_root(self.default_cache_root)
        shutil.rmtree(self.temp_dir)

    def test_get_small_parallel_enja(self):
        raw = get_small_parallel_enja()
        params = [('train', 50_000), ('dev', 500), ('test', 500)]
        for key, size in params:
            with self.subTest(key=key, size=size):
                self.assertIn(key, raw)
                self.assertEqual(len(raw[key]), size)
                self.assertEqual(len(SmallParallelEnJa(split=key)), size)

    def test_get_small_parallel_enja_twice(self):
        get_small_parallel_enja()
        with mock.patch('lineflow.datasets.small_parallel_enja.pickle', autospec=True) as mock_pickle:
            get_small_parallel_enja()
        mock_pickle.dump.assert_not_called()
        mock_pickle.load.assert_called_once()

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            SmallParallelEnJa(split='invalid_split')
