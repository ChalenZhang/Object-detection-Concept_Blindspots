import sys
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from concept_blindspots.matching import error_counts
from concept_blindspots.model import RiskMLP, predict_risk
from evaluate_statistics import evaluate
from evaluate_predictions import main as evaluate_predictions


def predictions(boxes, scores, labels=None):
    return {'boxes': torch.tensor(boxes).float().reshape(-1, 4),
            'scores': torch.tensor(scores).float(),
            'labels': torch.tensor(labels if labels is not None else [3] * len(boxes))}


class FalsePositiveTests(unittest.TestCase):
    def test_fp_threshold_and_duplicate(self):
        gt = [{'class_id': 3, 'box': [0, 0, 10, 10]}]
        pred = predictions([[0, 0, 10, 10]] * 3, [0.9, 0.5, 0.49])
        self.assertEqual(error_counts(gt, pred, task='fp'), 1)
        self.assertEqual(error_counts(gt, pred, task='fn'), 0)

    def test_wrong_class_is_false_positive_and_false_negative(self):
        gt = [{'class_id': 5, 'box': [0, 0, 10, 10]}]
        pred = predictions([[0, 0, 10, 10]], [0.9])
        self.assertEqual(error_counts(gt, pred, task='fp'), 1)
        self.assertEqual(error_counts(gt, pred, task='fn'), 1)

    def test_no_predictions(self):
        gt = [{'class_id': 3, 'box': [0, 0, 10, 10]}]
        self.assertEqual(error_counts(gt, predictions([], []), task='fp'), 0)
        self.assertEqual(error_counts(gt, predictions([], []), task='fn'), 1)

    def test_top100_is_per_class(self):
        pred = predictions([[0, 0, 10, 10]] * 220, [0.9] * 220, [3] * 110 + [5] * 110)
        self.assertEqual(error_counts([], pred, task='fp'), 200)
        self.assertEqual(error_counts([], pred, (3,), task='fp'), 100)

    def test_kitti_neighbor_matched_only_once(self):
        pred = predictions([[0, 0, 10, 10]] * 2, [0.9, 0.8])
        original = [{'type': 'Van', 'bbox_xyxy': [0, 0, 10, 10]}]
        self.assertEqual(error_counts([], pred, task='fp', kitti=original), 1)

    def test_kitti_valid_gt_takes_priority(self):
        gt = [{'class_id': 3, 'box': [0, 0, 10, 10]}]
        original = [{'type': 'Van', 'bbox_xyxy': [0, 0, 10, 10]}]
        pred = predictions([[0, 0, 10, 10]] * 2, [0.9, 0.8])
        self.assertEqual(error_counts(gt, pred, task='fp', kitti=original), 0)

    def test_kitti_person_sitting_and_dontcare(self):
        original = [{'type': 'Person_sitting', 'bbox_xyxy': [0, 0, 10, 10]},
                    {'type': 'DontCare', 'bbox_xyxy': [20, 0, 25, 10]}]
        pred = predictions([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]],
                           [0.9, 0.8, 0.7], [5, 3, 3])
        self.assertEqual(error_counts([], pred, task='fp', kitti=original), 1)
        self.assertEqual(error_counts([], pred, (3,), task='fp', kitti=original), 1)

    def test_capture_uses_fp_counts(self):
        counts = torch.tensor([3., 0., 1., 0.])
        result = evaluate(counts, torch.tensor([4., 1., 3., 2.]))
        self.assertEqual(result['Capture@5%'], 0.75)
        self.assertEqual(result['AUROC'], 1.)
        self.assertEqual(result['AUPRC'], 1.)

    def test_evaluator_accepts_person_abbreviation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = {'images': [{'id': 1, 'file_name': 'a.png'}, {'id': 2, 'file_name': 'b.png'}],
                           'categories': [{'id': 91, 'name': ' Psn. '}, {'id': 92, 'name': 'Car'}],
                           'annotations': [{'image_id': 1, 'category_id': 91, 'bbox': [0, 0, 10, 10]}]}
            (root / 'annotations.json').write_text(json.dumps(annotations))
            torch.save({'task': 'fp', 'images': ['a.png', 'b.png'], 'risk': torch.tensor([0., 1.]),
                        'detections': [predictions([[0, 0, 10, 10]], [0.9], [5]),
                                       predictions([[0, 0, 10, 10]], [0.9], [3])]}, root / 'predictions.pt')
            argv = ['evaluate_predictions', '--predictions', str(root / 'predictions.pt'),
                    '--annotations', str(root / 'annotations.json'), '--output', str(root / 'metrics.json')]
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                evaluate_predictions()
            result = json.loads((root / 'metrics.json').read_text())
            self.assertEqual(result['error_count'], 1)
            self.assertEqual(result['AUROC'], 1.)

    def test_inference_selects_separate_checkpoints(self):
        config = {'clusters': [{'cluster_id': 'a'}], 'blindspot_ids': ['a']}
        with tempfile.TemporaryDirectory() as temporary:
            for task, prefix, log_count in [('fn', 'risk', 0.), ('fp', 'risk_fp', 1.)]:
                model = RiskMLP(torch.zeros(8), torch.ones(8), hidden_dim=4)
                for param in model.parameters():
                    torch.nn.init.zeros_(param)
                model.heads.missed_count.bias.data.fill_(log_count)
                for seed in (2027, 2028, 2029):
                    checkpoint = {'failure_target': task, 'groups': ['car', 'person', 'vehicle', 'people', 'all'],
                                  'normalization': {'means': {'concept': torch.zeros(8)}, 'stds': {'concept': torch.ones(8)}},
                                  'model_config': {'hidden_dim': 4, 'dropout': 0.1}, 'state_dict': model.state_dict()}
                    torch.save(checkpoint, Path(temporary) / f'{prefix}_seed_{seed}.pt')
            concept = torch.ones(2, 4)
            self.assertTrue((predict_risk(concept, temporary, config, task='fn') == 0).all())
            expected = torch.full((2,), 2 * torch.expm1(torch.tensor(1.)).item())
            torch.testing.assert_close(predict_risk(concept, temporary, config, task='fp'), expected)


if __name__ == '__main__':
    unittest.main()
