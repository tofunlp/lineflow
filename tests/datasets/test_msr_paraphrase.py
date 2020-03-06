import shutil
import tempfile
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets.msr_paraphrase import MsrParaphrase, get_msr_paraphrase


class MsrParaphraseTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)

    def test_get_msr_paraphrase(self):
        raw = get_msr_paraphrase()
        params = [('train', 3_962), ('test', 1_650)]
        for key, size in params:
            with self.subTest(key=key, size=size):
                self.assertIn(key, raw)
                self.assertEqual(len(raw[key]), size)
                self.assertEqual(len(MsrParaphrase(split=key)), size)

    def test_get_msr_paraphrase_twice(self):
        get_msr_paraphrase()
        with mock.patch('lineflow.datasets.msr_paraphrase.pickle', autospec=True) as mock_pickle:
            get_msr_paraphrase()
        mock_pickle.dump.assert_not_called()
        mock_pickle.load.assert_called_once()

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            MsrParaphrase(split='invalid_split')
