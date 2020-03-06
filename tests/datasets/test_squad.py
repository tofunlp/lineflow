import shutil
import tempfile
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets.squad import Squad, get_squad


class SquadTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)

    def test_get_squad_v1(self):
        raw = get_squad(version=1)
        self.assertIn('train', raw)
        self.assertEqual(len(raw['train']), 87_599)
        self.assertIn('dev', raw)
        self.assertEqual(len(raw['dev']), 10_570)

    def test_get_squad_v1_twice(self):
        get_squad(version=1)
        with mock.patch('lineflow.datasets.squad.pickle', autospec=True) as mock_pickle:
            get_squad(version=1)
        mock_pickle.dump.assert_not_called()
        self.assertEqual(mock_pickle.load.call_count, 1)

    def test_get_squad_v2_twice(self):
        get_squad(version=2)
        with mock.patch('lineflow.datasets.squad.pickle', autospec=True) as mock_pickle:
            get_squad(version=2)
        mock_pickle.dump.assert_not_called()
        self.assertEqual(mock_pickle.load.call_count, 1)

    def test_get_squad_v2(self):
        raw = get_squad(version=2)
        self.assertIn('train', raw)
        self.assertEqual(len(raw['train']), 130_319)
        self.assertIn('dev', raw)
        self.assertEqual(len(raw['dev']), 11_873)

    def test_loads_v1_each_split(self):
        train = Squad(split='train', version=1)
        self.assertEqual(len(train), 87_599)
        dev = Squad(split='dev', version=1)
        self.assertEqual(len(dev), 10_570)

    def test_loads_v2_each_split(self):
        train = Squad(split='train', version=2)
        self.assertEqual(len(train), 130_319)
        dev = Squad(split='dev', version=2)
        self.assertEqual(len(dev), 11_873)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            Squad(split='invalid_split')

    def test_raises_value_error_with_invalid_version(self):
        with self.assertRaises(ValueError):
            Squad(version=3)
