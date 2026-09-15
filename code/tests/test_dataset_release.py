import csv
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {'Day': 2520, 'Adverse_A': 2580, 'Adverse_B': 900}


class RealDriveSimReleaseTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / 'data/realdrivesim/subset.csv').open(newline='') as handle:
            self.rows = list(csv.DictReader(handle))

    def test_complete_pool(self):
        self.assertEqual(len(self.rows), 6000)
        self.assertEqual(dict(Counter(r['component'] for r in self.rows)), EXPECTED)
        self.assertEqual(len({r['image'] for r in self.rows}), 6000)
        self.assertEqual(len({r['label'] for r in self.rows}), 6000)
        for row in self.rows:
            self.assertEqual(Path(row['image']).name, row['image'])
            self.assertEqual(Path(row['label']).name, row['label'])
            self.assertEqual(Path(row['image']).stem, Path(row['label']).stem)
            self.assertEqual(row['image'], row['sequence'] + '_' + row['frame'] + '.png')

    def test_existing_evaluation_is_covered(self):
        evaluation = json.loads((ROOT / 'data/splits/realdrivesim.json').read_text())
        self.assertEqual(len(evaluation), 5493)
        self.assertTrue({r['image'] for r in evaluation} <= {r['image'] for r in self.rows})

    def test_archive_inventory(self):
        inventory = json.loads((ROOT / 'data/realdrivesim/files.json').read_text())
        self.assertEqual(inventory['images'], 6000)
        self.assertEqual(inventory['components'], EXPECTED)
        self.assertEqual(sum(p['images'] for p in inventory['parts']), 6000)
        counts = Counter()
        for index, part in enumerate(inventory['parts'], 1):
            self.assertEqual(part['file'], f'realdrivesim-6000-part{index:02d}.zip')
            self.assertEqual(sum(part['components'].values()), part['images'])
            counts.update(part['components'])
        self.assertEqual(dict(counts), EXPECTED)


if __name__ == '__main__':
    unittest.main()
