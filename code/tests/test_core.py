import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concept_blindspots.metrics import binary_auroc, average_precision, normalized_aurc, selected
from concept_blindspots.model import risk_inputs, RiskMLP, load_detector_state
from concept_blindspots.matching import match_class, match_group
from concept_blindspots.training_utils import SceneViewStream


class CoreTests(unittest.TestCase):
    def test_detector_weight_parts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = root / "detector_r101"
            parts.mkdir()
            torch.save({"backbone.weight": torch.arange(3)}, parts / "backbone.pt")
            torch.save({"head.weight": torch.ones(2)}, parts / "head.pt")
            state = load_detector_state(root)
            self.assertEqual(set(state), {"backbone.weight", "head.weight"})
            self.assertTrue(torch.equal(state["backbone.weight"], torch.arange(3)))
            self.assertTrue(torch.equal(state["head.weight"], torch.ones(2)))

    def test_detector_duplicate_parameters_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = root / "detector_r101"
            parts.mkdir()
            for name in ("a", "b"):
                torch.save({"weight": torch.ones(1)}, parts / f"{name}.pt")
            with self.assertRaisesRegex(ValueError, "Duplicate detector parameters"):
                load_detector_state(root)

    def test_empty_detector_directory_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "detector_r101").mkdir()
            with self.assertRaises(FileNotFoundError):
                load_detector_state(root)

    def test_single_detector_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save({"weight": torch.arange(2)}, root / "detector_r101.pt")
            self.assertTrue(torch.equal(load_detector_state(root)["weight"], torch.arange(2)))

    def test_input_identity_order(self):
        config=json.loads((Path(__file__).resolve().parents[1]/"configs/concepts.json").read_text())
        x=torch.arange(404).reshape(1,-1)
        y=risk_inputs(x,config)
        self.assertEqual(y.shape,(1,472))
        self.assertTrue(torch.equal(y[:,:404],x))
        self.assertEqual(int(y[0,404]),2)
        self.assertEqual(int(y[0,421]),103)
        self.assertEqual(len(config["blindspot_ids"]),17)

    def test_perfect_ranking(self):
        labels=torch.tensor([0,0,1,1]);score=torch.arange(4).float()
        self.assertEqual(binary_auroc(labels,score),1)
        self.assertEqual(average_precision(labels,score),1)
        self.assertEqual(binary_auroc(labels,torch.ones(4)),0.5)
        self.assertLess(normalized_aurc(labels,score)[1],normalized_aurc(labels,-score)[1])

    def test_budget_rounds_up(self):
        self.assertEqual(len(selected(torch.arange(21),0.05)),2)

    def test_matching_is_one_to_one(self):
        tp,fp,_,_=match_class([[0,0,10,10]],torch.tensor([[0,0,10,10],[0,0,10,10]]),torch.tensor([0.9,0.8]),0.5)
        self.assertEqual((tp,fp),(1,1))

    def test_unmatched_class_remains_missed(self):
        gt=[{"class_id":5,"box":[0,0,10,10]}]
        pred={"labels":torch.tensor([3]),"scores":torch.tensor([0.9]),"boxes":torch.tensor([[0,0,10,10]])}
        self.assertEqual(match_group(gt,pred,(3,5),0.05,0.5,100)["miss_count"],1)

    def test_scene_sampling(self):
        scenes=[torch.tensor([i,i+4]) for i in range(4)]
        stream=SceneViewStream(scenes,4,2027)
        self.assertEqual(len(set((stream.draw()%4).tolist())),4)

    def test_risk_model_shapes(self):
        model=RiskMLP(torch.zeros(472),torch.ones(472))
        for output in model(torch.zeros(2,472)).values():self.assertEqual(output.shape,(2,5))


if __name__=="__main__":unittest.main()
