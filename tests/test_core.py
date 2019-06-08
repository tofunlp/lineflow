from unittest import TestCase
from unittest.mock import patch, Mock

import lineflow
from lineflow import Dataset
from lineflow.core import ConcatDataset, ZipDataset
from lineflow.core import RandomAccessConcat, RandomAccessZip


class RandomAccessConcatTestCase(TestCase):

    def setUp(self):
        self.n = 5
        self.base = range(100)
        self.data = RandomAccessConcat(*[self.base for _ in range(5)])

    def test_init(self):
        self.assertEqual(len(self.data._datasets), self.n)
        self.assertIsNone(self.data._offsets)
        self.assertIsNone(self.data._length)

    def test_iter(self):
        for x, y in zip(self.data, list(self.base) * self.n):
            self.assertEqual(x, y)
        self.assertIsNone(self.data._offsets)
        self.assertIsNone(self.data._length)

    def test_supports_random_access_lazily(self):
        import itertools

        self.assertIsNone(self.data._offsets)
        self.assertSequenceEqual(self.data, list(self.base) * self.n)
        expected_lengths = list(itertools.accumulate(len(self.base) for _ in range(self.n)))
        self.assertListEqual(self.data._lengths, expected_lengths)
        self.assertListEqual(self.data._offsets, [0] + expected_lengths[:-1])

    def test_raises_index_error_with_invalid_index(self):
        with self.assertRaises(IndexError):
            self.data[len(self.data)]
            self.data[-1]

    def test_returns_length_lazily(self):
        self.assertIsNone(self.data._length)
        self.assertEqual(len(self.data), len(self.base) * self.n)
        self.assertEqual(self.data._length, len(self.data))


class RandomAccessZipTestCase(TestCase):

    def setUp(self):
        self.n = 5
        self.base = range(100)
        self.data = RandomAccessZip(*[self.base for _ in range(5)])

    def test_init(self):
        self.assertEqual(len(self.data._datasets), self.n)
        self.assertIsNone(self.data._length)

    def test_iter(self):
        for x, y in zip(self.data, self.base):
            self.assertEqual(x, tuple([y] * self.n))
        self.assertIsNone(self.data._length)

    def test_supports_random_access(self):
        self.assertSequenceEqual(self.data, list(zip(*[self.base for _ in range(self.n)])))

    def test_raises_index_error_with_invalid_index(self):
        with self.assertRaises(IndexError):
            self.data[len(self.data)]
            self.data[-1]

    def test_returns_lengths_lazily(self):
        self.assertIsNone(self.data._length)
        self.assertEqual(len(self.data), len(self.base))
        self.assertEqual(self.data._length, len(self.data))


class DatasetTestCase(TestCase):

    def setUp(self):
        self.base = range(100)
        self.data = Dataset(self.base)

    def test_getitem(self):
        self.assertSequenceEqual(self.data, self.base)

    def test_supports_slicing(self):
        slice1 = slice(10, 20)
        slice2 = slice(0, 99)
        self.assertListEqual(self.data[slice1], list(self.base[slice1]))
        self.assertListEqual(self.data[slice2], list(self.base[slice2]))

    def test_len(self):
        self.assertEqual(len(self.data), len(self.base))

    def test_add(self):
        data = self.data + self.data + self.data
        expected = list(self.base) * 3
        self.assertSequenceEqual(data, expected)

    def test_map(self):
        def f(x): return x ** 2

        self.assertSequenceEqual(
            self.data.map(f),
            list(map(f, self.base)))

    def test_keeps_original_dataset_after_multiple_maps(self):
        def f(x): return x

        data = self.data
        for i in range(100):
            data = data.map(f)
            self.assertEqual(data._dataset, self.base)
            self.assertEqual(len(data._funcs), i + 1)

    def test_supports_method_chain(self):
        data = self.data.map(lambda x: x ** 2).map(lambda x: x / 2)
        self.assertSequenceEqual(
            data, [x ** 2 / 2 for x in self.base])

    def test_all(self):
        self.assertListEqual(self.data.all(), list(self.base))

    def test_first(self):
        self.assertEqual(self.data.first(), self.base[0])

    def test_take(self):
        n = 50
        self.assertListEqual(self.data.take(n), list(self.base[:n]))

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
        self.assertListEqual(data._dataset, [x ** 2 for x in self.base])

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


class LineflowConcatTestCase(TestCase):

    def setUp(self):
        self.base = range(100)
        self.n = 5
        self.data = lineflow.concat(*[Dataset(self.base)] * self.n)

    def test_returns_concat_dataset(self):
        self.assertIsInstance(self.data, ConcatDataset)
        self.assertIsInstance(self.data._dataset, RandomAccessConcat)

    def test_keeps_original_dataset_after_multiple_maps(self):
        def f(x): return x

        data = self.data
        for i in range(100):
            data = data.map(f)
            self.assertIsInstance(data._dataset, RandomAccessConcat)
            self.assertEqual(len(data._funcs), i + 1)

    def test_supports_random_access(self):
        self.assertSequenceEqual(self.data, list(self.base) * self.n)


class LineflowZipTestCase(TestCase):

    def setUp(self):
        self.base = range(100)
        self.n = 5
        self.data = lineflow.zip(*[Dataset(self.base)] * self.n)

    def test_returns_zip_dataset(self):
        self.assertIsInstance(self.data, ZipDataset)
        self.assertIsInstance(self.data._dataset, RandomAccessZip)

    def test_keeps_original_dataset_after_multiple_maps(self):
        def f(x): return x

        data = self.data
        for i in range(100):
            data = data.map(f)
            self.assertIsInstance(data._dataset, RandomAccessZip)
            self.assertEqual(len(data._funcs), i + 1)

    def test_supports_random_access(self):
        self.assertSequenceEqual(self.data, list(zip(*[self.base for _ in range(self.n)])))


class LineflowFilterTestCase(TestCase):

    def setUp(self):
        self.data = Dataset(range(100))

    def test_returns_filtered_data_eagerly(self):
        result = lineflow.filter(lambda x: x % 2 == 0, self.data)
        expected = [x for x in self.data if x % 2 == 0]
        self.assertListEqual(result, expected)

    def test_returns_filtered_data_lazily(self):
        result = lineflow.filter(lambda x: x % 2 == 0, self.data, lazy=True)
        self.assertIsInstance(result, filter)
        expected = [x for x in self.data if x % 2 == 0]
        for x, y in zip(result, expected):
            self.assertEqual(x, y)


class LineflowFlatMapTestCase(TestCase):

    def setUp(self):
        self.data = Dataset(range(100))

    def test_returns_flat_mapped_data_eagerly(self):
        result = lineflow.flat_map(lambda x: [x] * 3, self.data)
        expected = [[x] * 3 for x in self.data]
        expected = [x for xs in expected for x in xs]
        self.assertListEqual(result, expected)

    def test_returns_flat_mapped_data_lazily(self):
        import itertools

        result = lineflow.flat_map(lambda x: [x] * 3, self.data, lazy=True)
        self.assertIsInstance(result, itertools.chain)
        expected = list(itertools.chain.from_iterable(
            [[x] * 3 for x in self.data]))
        for x, y in zip(result, expected):
            self.assertEqual(x, y)
