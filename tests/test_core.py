from unittest import TestCase
from unittest.mock import patch, Mock
import tempfile

import lineflow
from lineflow import Dataset, TextDataset


class DatasetTestCase(TestCase):

    def setUp(self):
        self.base = range(100)
        self.data = Dataset(self.base)

    def test_getitem(self):
        for i, y in enumerate(self.base):
            self.assertEqual(self.data[i], y)

    def test_len(self):
        self.assertEqual(len(self.data), len(self.base))

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

    @patch('lineflow.core.open')
    @patch('lineflow.core.pickle.dump')
    def test_save(self, pickle_dump_mock, open_mock):
        enter_mock = Mock()
        # mock file object
        open_mock.return_value.__enter__.return_value = enter_mock

        filepath1 = '/path/to/dataset1'
        self.data.save(filepath1)
        open_mock.assert_called_with(filepath1, 'wb')
        pickle_dump_mock.assert_called_with(
            self.data.all(), enter_mock)
        self.assertEqual(open_mock.call_count, 1)
        self.assertEqual(pickle_dump_mock.call_count, 1)

        filepath2 = '/path/to/dataset2'
        data = self.data.map(lambda x: x ** 2).save(filepath2)
        open_mock.assert_called_with(filepath2, 'wb')
        pickle_dump_mock.assert_called_with(
            data.all(), enter_mock)
        self.assertEqual(open_mock.call_count, 2)
        self.assertEqual(pickle_dump_mock.call_count, 2)

        expected = [x ** 2 for x in self.base]
        self.assertListEqual(data.all(), expected)
        self.assertIsInstance(data, lineflow.core.CacheDataset)
        for i, y in enumerate(expected):
            self.assertEqual(data[i], y)

        data = data.map(lambda x: x ** 2)
        result = []
        for x in data._dataset:
            for f in data._map_func_list:
                x = f(x)
            result.append(x)
        expected = [x ** 2 for x in expected]
        self.assertListEqual(data.all(), expected)
        self.assertListEqual(result, expected)

    @patch('lineflow.core.open')
    @patch('lineflow.core.pickle.load')
    def test_load(self, pickle_load_mock, open_mock):
        pickle_load_mock.return_value = list(self.base)
        enter_mock = Mock()
        open_mock.return_value.__enter__.return_value = enter_mock

        filepath = '/path/to/dataset'
        data = Dataset.load(filepath)
        open_mock.assert_called_once_with(filepath, 'rb')
        pickle_load_mock.assert_called_once_with(enter_mock)

        self.assertListEqual(data.all(), list(self.base))
        self.assertEqual(data._dataset, list(self.base))


class TextDatasetTestCase(TestCase):

    @patch('lineflow.core.linecache.getline')
    def test_text(self, linecache_getline_mock):
        lines = ['This is a test .', 'That is also a test .']
        linecache_getline_mock.side_effect = lambda filename, i: lines[i - 1]
        fp = tempfile.NamedTemporaryFile()
        for x in lines:
            fp.write(f'{x}\n'.encode('utf-8'))
        fp.seek(0)

        data = TextDataset(fp.name)
        for x, y in zip(data, lines):
            self.assertEqual(x, y)

        for i, y in enumerate(lines):
            self.assertEqual(data[i], y)
            linecache_getline_mock.called_once_with(fp.name, i + 1)
        self.assertEqual(linecache_getline_mock.call_count, i + 1)

        self.assertEqual(len(data), len(lines))

        self.assertEqual(data._dataset, data)

        data = data.map(str.split)

        for x, y in zip(data, lines):
            self.assertEqual(x, y.split())

        self.assertIsInstance(data, lineflow.core.MapDataset)

        fp.close()
