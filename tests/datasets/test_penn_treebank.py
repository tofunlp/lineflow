import shutil
import tempfile
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets.penn_treebank import PennTreebank, get_penn_treebank


class PennTreebankTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)

    def test_get_penn_treebank(self):
        raw = get_penn_treebank()
        params = [('train', 42_068), ('dev', 3_370), ('test', 3_761)]
        for key, size in params:
            with self.subTest(key=key, size=size):
                self.assertIn(key, raw)
                self.assertEqual(len(raw[key]), size)
                self.assertEqual(len(PennTreebank(split=key)), size)

    def test_get_penn_treebank_twice(self):
        get_penn_treebank()
        with mock.patch('lineflow.datasets.penn_treebank.pickle', autospect=True) as mock_pickle:
            get_penn_treebank()
        mock_pickle.dump.assert_not_called()
        mock_pickle.load.assert_called_once()

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            PennTreebank(split='invalid_split')
