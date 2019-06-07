from unittest import TestCase
import tempfile

import lineflow
from lineflow import TextDataset, CsvDataset
from lineflow.text import RandomAccessFile
from lineflow.text import ZipTextDataset, ConcatTextDataset


class RandomAccessFileTestCase(TestCase):

    def setUp(self):
        self.length = 100

        fp = tempfile.NamedTemporaryFile()
        for i in range(self.length):
            fp.write(f'line #{i}\n'.encode('utf-8'))
        fp.seek(0)
        self.fp = fp

    def tearDown(self):
        self.fp.close()

    def test_init(self):
        text = RandomAccessFile(self.fp.name)
        self.assertEqual(text._path, self.fp.name)
        self.assertEqual(text._offsets, None)
        self.assertEqual(text._length, None)

    def test_initialize_offsets(self):
        text = RandomAccessFile(self.fp.name)
        text._initialize_offsets()
        self.assertIsInstance(text._offsets, list)

    def test_getitem(self):
        text = RandomAccessFile(self.fp.name)
        for i in range(self.length):
            self.assertEqual(text[i], f'line #{i}')

    def test_raises_index_error_with_invalid_index(self):
        text = RandomAccessFile(self.fp.name)
        with self.assertRaises(IndexError):
            text[-1]
            text[self._length]

    def test_len(self):
        text = RandomAccessFile(self.fp.name)
        self.assertEqual(len(text), self.length)


class TextDatasetTestCase(TestCase):

    def setUp(self):
        lines = ['This is a test .', 'That is also a test .']
        fp = tempfile.NamedTemporaryFile()
        for x in lines:
            fp.write(f'{x}\n'.encode('utf-8'))
        fp.seek(0)
        self.lines = lines
        self.fp = fp

    def tearDown(self):
        self.fp.close()

    def test_text(self):
        fp = self.fp
        lines = self.lines

        data = TextDataset(fp.name)
        for x, y in zip(data, lines):
            self.assertEqual(x, y)

        for i, y in enumerate(lines):
            self.assertEqual(data[i], y)

        self.assertEqual(data._length, None)
        self.assertEqual(len(data), len(lines))
        self.assertEqual(data._length, len(lines))
        # check if length is cached
        self.assertEqual(len(data), len(lines))

        self.assertIsInstance(data._dataset, RandomAccessFile)

        data = data.map(str.split)

        for x, y in zip(data, lines):
            self.assertEqual(x, y.split())

        self.assertIsInstance(data, lineflow.core.MapDataset)
        self.assertIsInstance(data._dataset, RandomAccessFile)

    def test_zips_multiple_files(self):
        fp = self.fp
        lines = self.lines

        data = TextDataset([fp.name, fp.name], mode='zip')
        for x, y in zip(data, lines):
            self.assertTupleEqual(x, (y, y))
        for j, y in enumerate(lines):
            self.assertTupleEqual(data[j], (y, y))
        self.assertEqual(len(data), len(lines))
        self.assertEqual(data._length, len(lines))
        self.assertIsInstance(data._dataset, ZipTextDataset)
        self.assertIsInstance(data.map(lambda x: x)._dataset, ZipTextDataset)
        for d in data._dataset._datasets:
            self.assertIsInstance(d, RandomAccessFile)

    def test_concats_multiple_files(self):
        fp = self.fp
        lines = self.lines

        data = TextDataset([fp.name, fp.name], mode='concat')
        for x, y in zip(data, lines + lines):
            self.assertEqual(x, y)
        for j, y in enumerate(lines + lines):
            self.assertEqual(data[j], y)
        self.assertEqual(len(data), len(lines) * 2)
        self.assertEqual(data._length, len(lines) * 2)

        self.assertEqual(data[len(data) - 1], lines[-1])
        self.assertIsInstance(data._dataset, ConcatTextDataset)
        self.assertIsInstance(data.map(lambda x: x)._dataset, ConcatTextDataset)
        for d in data._dataset._datasets:
            self.assertIsInstance(d, RandomAccessFile)

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

        self.assertEqual(len(ds), len(lines) - 1)

        header = lines[0].split(',')

        for i, x in enumerate(ds, start=1):
            y = dict(zip(header, lines[i].split(',')))
            self.assertDictEqual(dict(x), y)
            self.assertDictEqual(dict(ds[i - 1]), y)

        self.assertIsInstance(ds._dataset, RandomAccessFile)

        fp.close()

    def test_loads_csv_without_header(self):
        lines = ['this is English .,this is Japanese .',
                 'this is also English .,this is also Japanese .']
        fp = tempfile.NamedTemporaryFile()
        for x in lines:
            fp.write(f'{x}\n'.encode('utf-8'))
        fp.seek(0)

        ds = CsvDataset(fp.name)

        self.assertEqual(len(ds), len(lines))

        for i, x in enumerate(ds):
            y = lines[i].split(',')
            self.assertListEqual(x, y)
            self.assertListEqual(ds[i], y)

        self.assertIsInstance(ds._dataset, RandomAccessFile)

        fp.close()
