from unittest import TestCase
from unittest.mock import patch, Mock

import lineflow
from lineflow import Dataset
from lineflow.core import RandomAccessConcat, RandomAccessZip


class DatasetTestCase(TestCase):

    def setUp(self):
        self.base = range(100)
        self.data = Dataset(self.base)

    def test_getitem(self):
        for i, y in enumerate(self.base):
            self.assertEqual(self.data[i], y)
        self.assertEqual(self.data[10:20], list(self.base[10:20]))

    def test_len(self):
        self.assertEqual(len(self.data), len(self.base))

    def test_add(self):
        data = self.data + self.data + self.data
        expected = list(self.base) * 3
        for i, (x, y) in enumerate(zip(data, expected)):
            self.assertEqual(x, y)
            self.assertEqual(data[i], y)

    def test_map(self):
        def f(x):
            return x ** 2

        data = self.data.map(f)

        for x, y in zip(data, self.base):
            self.assertEqual(x, f(y))

        for i, y in enumerate(self.base):
            self.assertEqual(data[i], f(y))

        self.assertEqual(data._dataset, self.base)

    def test_method_chain(self):
        data = self.data.map(lambda x: x ** 2) \
            .map(lambda x: x / 2)

        expected = [x ** 2 / 2 for x in self.base]

        for x, y in zip(data, expected):
            self.assertEqual(x, y)

        self.assertEqual(data._dataset, self.base)

    def test_all(self):
        data = self.data
        expected = list(self.base)

        self.assertListEqual(data.all(), expected)

    def test_first(self):
        data = self.data.first()
        expected = next(iter(self.base))

        self.assertEqual(data, expected)

    def test_take(self):
        n = 50
        data = self.data.take(n)
        expected = list(self.base[:n])

        self.assertListEqual(data, expected)

    @patch('lineflow.core.Path.open')
    @patch('lineflow.core.Path')
    @patch('lineflow.core.pickle.dump')
    def test_saves_yourself(self, pickle_dump_mock, Path_mock, open_mock):
        path = Mock()
        Path_mock.return_value = path
        # Assume cache doesn't exist, but a directory exists
        path.exists.return_value = False
        path.parent.exists.return_value = True
        # Setup Path.open
        fp = Mock()
        open_mock.return_value.__enter__.return_value = fp
        path.open = open_mock

        filepath = '/path/to/cache'
        data = self.data.save(filepath)

        path.exists.assert_called_once()
        path.parent.exists.assert_called_once()
        path.open.assert_called_once_with('wb')
        pickle_dump_mock.assert_called_once_with(self.data.all(), fp)
        self.assertIsInstance(data, lineflow.core.CacheDataset)

    @patch('lineflow.core.Path.open')
    @patch('lineflow.core.Path')
    @patch('lineflow.core.pickle.dump')
    def test_makes_a_directory_and_saves_yourself(self,
                                                  pickle_dump_mock,
                                                  Path_mock,
                                                  open_mock):
        path = Mock()
        Path_mock.return_value = path
        # Assume cache doesn't exist, also a directory doesn't exist
        path.exists.return_value = False
        path.parent.exists.return_value = False
        # Setup Path.open
        fp = Mock()
        open_mock.return_value.__enter__.return_value = fp
        path.open = open_mock

        filepath = '/path/to/cache'
        data = self.data.save(filepath)

        path.exists.assert_called_once()
        path.parent.exists.assert_called_once()
        path.parent.mkdir.assert_called_once_with(parents=True)
        path.open.assert_called_once_with('wb')
        pickle_dump_mock.assert_called_once_with(self.data.all(), fp)
        self.assertIsInstance(data, lineflow.core.CacheDataset)

    @patch('lineflow.core.Path.open')
    @patch('lineflow.core.Path')
    @patch('lineflow.core.pickle.dump')
    def test_maps_func_and_saves_yourself(self,
                                          pickle_dump_mock,
                                          Path_mock,
                                          open_mock):
        path = Mock()
        Path_mock.return_value = path
        # Assume cache doesn't exist, but a directory exists
        path.exists.return_value = False
        path.parent.exists.return_value = True
        # Setup Path.open
        fp = Mock()
        open_mock.return_value.__enter__.return_value = fp
        path.open = open_mock

        filepath = '/path/to/cache'
        data = self.data.map(lambda x: x ** 2).save(filepath)

        path.exists.assert_called_once()
        path.parent.exists.assert_called_once()
        path.open.assert_called_once_with('wb')
        pickle_dump_mock.assert_called_once_with(data.all(), fp)
        self.assertIsInstance(data, lineflow.core.CacheDataset)

        for i, x in enumerate(data):
            y = self.data[i] ** 2
            self.assertEqual(x, y)
            self.assertEqual(data[i], y)

    @patch('lineflow.core.Path.open')
    @patch('lineflow.core.Path')
    @patch('lineflow.core.pickle.load')
    def test_loads_existed_cache_implicitly(self,
                                            pickle_load_mock,
                                            Path_mock,
                                            open_mock):
        path = Mock()
        Path_mock.return_value = path
        # Assume cache exists
        path.exists.return_value = True
        # Setup Path.open
        fp = Mock()
        open_mock.return_value.__enter__.return_value = fp
        path.open = open_mock
        # Setup pickle.load
        pickle_load_mock.return_value = list(self.base)

        filepath = '/path/to/cache'
        data = self.data.save(filepath)

        path.exists.assert_called_once()
        path.open.assert_called_once_with('rb')
        pickle_load_mock.assert_called_once_with(fp)
        self.assertIsInstance(data, lineflow.core.CacheDataset)

    @patch('lineflow.core.open')
    @patch('lineflow.core.pickle.load')
    def test_load(self, pickle_load_mock, open_mock):
        pickle_load_mock.return_value = list(self.base)
        enter_mock = Mock()
        open_mock.return_value.__enter__.return_value = enter_mock

        filepath = '/path/to/dataset'
        data = lineflow.load(filepath)
        open_mock.assert_called_once_with(filepath, 'rb')
        pickle_load_mock.assert_called_once_with(enter_mock)

        self.assertListEqual(data.all(), list(self.base))
        self.assertEqual(data._dataset, list(self.base))

        with self.assertWarns(DeprecationWarning):
            lineflow.Dataset.load(filepath)


class MiscTestCase(TestCase):

    def setUp(self):
        self.base = range(100)
        self.data = Dataset(self.base)

    def test_concat(self):
        data = lineflow.concat(self.data, self.data)
        expected = list(self.base) * 3
        for i, (x, y) in enumerate(zip(data, expected)):
            self.assertEqual(x, y)
            self.assertEqual(data[i], y)

        with self.assertRaises(IndexError):
            data[len(data) * 3]

        result = data.map(lambda x: x ** 2).map(lambda x: x)
        expected = [y ** 2 for y in expected]
        for i, (x, y) in enumerate(zip(result, expected)):
            self.assertEqual(x, y)
            self.assertEqual(result[i], y)
        self.assertIsInstance(result._dataset, RandomAccessConcat)

    def test_zip(self):
        data = lineflow.zip(self.data, self.data)
        expected = list(zip(self.base, self.base))
        for i, (x, y) in enumerate(zip(data, expected)):
            self.assertEqual(x, y)
            self.assertEqual(data[i], y)

        with self.assertRaises(IndexError):
            data[len(data)]

        result = data.map(lambda x: x).map(lambda x: x)
        for i, (x, y) in enumerate(zip(result, expected)):
            self.assertEqual(x, y)
            self.assertEqual(result[i], y)
        self.assertIsInstance(result._dataset, RandomAccessZip)

    def test_filter(self):
        result = lineflow.filter(lambda x: x % 2 == 0, self.data)
        expected = [x for x in self.data if x % 2 == 0]
        self.assertListEqual(result, expected)

        result = lineflow.filter(lambda x: x % 2 == 0, self.data, lazy=True)
        for x, y in zip(result, expected):
            self.assertEqual(x, y)

    def test_flat_map(self):
        result = lineflow.flat_map(lambda x: [x] * 3, self.data)
        expected = [[x] * 3 for x in self.data]
        expected = [x for xs in expected for x in xs]
        self.assertListEqual(result, expected)

        result = lineflow.flat_map(lambda x: [x] * 3, self.data, lazy=True)
        for x, y in zip(result, expected):
            self.assertEqual(x, y)
