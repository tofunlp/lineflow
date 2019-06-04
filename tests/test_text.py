from unittest import TestCase
from unittest.mock import patch
import tempfile

import lineflow
from lineflow import TextDataset, CsvDataset
from lineflow.text import SingleTextDataset, ZipTextDataset, ConcatTextDataset


class TextDatasetTestCase(TestCase):

    def setUp(self):
        self.linecache_getline_patcher = patch('lineflow.text.linecache.getline')
        self.linecache_getline_mock = self.linecache_getline_patcher.start()
        lines = ['This is a test .', 'That is also a test .']
        self.linecache_getline_mock.side_effect = \
            lambda filename, i: lines[-1] if i - 1 > len(lines) else lines[i - 1]
        fp = tempfile.NamedTemporaryFile()
        for x in lines:
            fp.write(f'{x}\n'.encode('utf-8'))
        fp.seek(0)
        self.lines = lines
        self.fp = fp

    def tearDown(self):
        self.linecache_getline_patcher.stop()
        self.fp.close()

    def test_text(self):
        fp = self.fp
        lines = self.lines
        linecache_getline_mock = self.linecache_getline_mock

        data = TextDataset(fp.name)
        for x, y in zip(data, lines):
            self.assertEqual(x, y)

        for i, y in enumerate(lines):
            self.assertEqual(data[i], y)
            linecache_getline_mock.called_once_with(fp.name, i + 1)
        self.assertEqual(linecache_getline_mock.call_count, i + 1)

        self.assertEqual(data._length, None)
        self.assertEqual(len(data), len(lines))
        self.assertEqual(data._length, len(lines))
        # check if length is cached
        self.assertEqual(len(data), len(lines))

        self.assertIsInstance(data._dataset, SingleTextDataset)

        data = data.map(str.split)

        for x, y in zip(data, lines):
            self.assertEqual(x, y.split())

        self.assertIsInstance(data, lineflow.core.MapDataset)
        self.assertIsInstance(data._dataset, SingleTextDataset)

    def test_zips_multiple_files(self):
        fp = self.fp
        lines = self.lines
        linecache_getline_mock = self.linecache_getline_mock

        data = TextDataset([fp.name, fp.name], mode='zip')
        for x, y in zip(data, lines):
            self.assertTupleEqual(x, (y, y))
        for j, y in enumerate(lines):
            self.assertTupleEqual(data[j], (y, y))
            linecache_getline_mock.called_once_with(fp.name, j + 1)
        self.assertEqual(linecache_getline_mock.call_count,
                         (j + 1) * len(lines))
        self.assertEqual(len(data), len(lines))
        self.assertEqual(data._length, len(lines))
        self.assertIsInstance(data._dataset, ZipTextDataset)
        self.assertIsInstance(data.map(lambda x: x)._dataset, ZipTextDataset)

    def test_concats_multiple_files(self):
        fp = self.fp
        lines = self.lines
        linecache_getline_mock = self.linecache_getline_mock

        data = TextDataset([fp.name, fp.name], mode='concat')
        for x, y in zip(data, lines + lines):
            self.assertEqual(x, y)
        for j, y in enumerate(lines + lines):
            self.assertEqual(data[j], y)
            linecache_getline_mock.called_once_with(fp.name, j + 1)
        self.assertEqual(linecache_getline_mock.call_count,
                         len(lines) * 2)
        self.assertEqual(len(data), len(lines) * 2)
        self.assertEqual(data._length, len(lines) * 2)

        self.assertEqual(data[len(data) - 1], lines[-1])
        self.assertIsInstance(data._dataset, ConcatTextDataset)
        self.assertIsInstance(data.map(lambda x: x)._dataset, ConcatTextDataset)

    def test_raises_value_error_with_invalid_mode(self):
        with self.assertRaises(ValueError):
            TextDataset([self.fp.name, self.fp.name], mode='invalid_mode')


class CsvDatasetTestCase(TestCase):

    def test_loads_csv_with_header(self):
        lines = ['en,ja',
                 'this is English .,this is Japanese .',
                 'this is also English .,this is also Japanese .']
        fp = tempfile.NamedTemporaryFile()
        for x in lines:
            fp.write(f'{x}\n'.encode('utf-8'))
        fp.seek(0)

        ds = CsvDataset(fp.name, header=True)

        header = lines[0].split(',')

        for i, x in enumerate(ds, start=1):
            y = dict(zip(header, lines[i].split(',')))
            self.assertDictEqual(dict(x), y)
            self.assertDictEqual(dict(ds[i - 1]), y)

        self.assertEqual(len(ds), len(lines) - 1)

        fp.close()

    def test_loads_csv_without_header(self):
        lines = ['this is English .,this is Japanese .',
                 'this is also English .,this is also Japanese .']
        fp = tempfile.NamedTemporaryFile()
        for x in lines:
            fp.write(f'{x}\n'.encode('utf-8'))
        fp.seek(0)

        ds = CsvDataset(fp.name)

        for i, x in enumerate(ds):
            y = lines[i].split(',')
            self.assertListEqual(x, y)
            self.assertListEqual(ds[i], y)

        self.assertEqual(len(ds), len(lines))

        fp.close()
