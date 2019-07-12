from unittest import TestCase
import tempfile

import easyfile

import lineflow
from lineflow import TextDataset, CsvDataset


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
        self.assertEqual(data._length, None)

        for x, y in zip(data, lines):
            self.assertEqual(x, y)

        for i, y in enumerate(lines):
            self.assertEqual(data[i], y)

        self.assertEqual(len(data), len(lines))
        self.assertEqual(data._length, len(lines))
        # check if length is cached
        self.assertEqual(len(data), len(lines))

        self.assertIsInstance(data._dataset, easyfile.TextFile)

        data = data.map(str.split)

        for x, y in zip(data, lines):
            self.assertEqual(x, y.split())

        self.assertIsInstance(data, lineflow.core.MapDataset)
        self.assertIsInstance(data._dataset, TextDataset)

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
        self.assertIsInstance(data._dataset, lineflow.core.ZipDataset)
        self.assertIsInstance(data.map(lambda x: x)._dataset, TextDataset)

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
        self.assertIsInstance(data._dataset, lineflow.core.ConcatDataset)
        self.assertIsInstance(data.map(lambda x: x)._dataset, TextDataset)

    def test_raises_value_error_with_invalid_mode(self):
        with self.assertRaises(ValueError):
            TextDataset([self.fp.name, self.fp.name], mode='invalid_mode')


class CsvDatasetTestCase(TestCase):

    def setUp(self):
        lines = ['en,ja',
                 'this is English .,this is Japanese .',
                 'this is also English .,this is also Japanese .']
        self.lines = lines
        fp = tempfile.NamedTemporaryFile()
        for x in lines:
            fp.write(f'{x}\n'.encode('utf-8'))
        fp.seek(0)
        self.fp = fp

    def tearDown(self):
        self.fp.close()

    def test_loads_csv_with_header(self):
        data = CsvDataset(self.fp.name, header=True)
        self.assertIsInstance(data._dataset, easyfile.CsvFile)
        data = data.map(dict)
        header = self.lines[0].split(',')
        expected = [dict(zip(header, l.split(','))) for l in self.lines[1:]]
        self.assertSequenceEqual(data, expected)

    def test_loads_csv_without_header(self):
        data = CsvDataset(self.fp.name)
        self.assertIsInstance(data._dataset, easyfile.CsvFile)
        self.assertSequenceEqual(data, [l.split(',') for l in self.lines])
