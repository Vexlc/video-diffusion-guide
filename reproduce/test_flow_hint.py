"""逐项核对综合实验的两个加号、独立条件及冻结的训练路径。"""
import base64
import hashlib
import importlib.util
import json
import struct
import unittest
from pathlib import Path

from flow_hint_report import film

if importlib.util.find_spec("torch"):
    import torch
    import flow_hint as lab
else:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def floats(spec):
    return [entry[0] for entry in struct.iter_unpack("<f", base64.b64decode(spec["data"]))]


class RecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT/"flow-hint/numbers.json").read_text())

    def test_all_hidden_additions_and_euler_steps(self):
        for strength, rec in self.data["records"].items():
            gain = float(strength)
            hidden = iter(floats(rec["features"]))
            for before, residual, after in zip(hidden, hidden, hidden):
                self.assertAlmostEqual(before + gain*residual, after, delta=2e-6)
            xs, speeds = floats(rec["x"]), floats(rec["v"])
            self.assertEqual(rec["x"]["shape"], [12,25,18,4])
            for case in range(12):
                for step in range(24):
                    start = (case*25+step)*72
                    velocity = (case*24+step)*72
                    for j in range(72):
                        self.assertAlmostEqual(xs[start+j]+speeds[velocity+j]/24,
                                               xs[start+72+j], delta=5e-7)

    def test_counterfactuals_and_model_fingerprints(self):
        off, on = (self.data["records"][scale] for scale in ("0.0","1.0"))
        self.assertEqual(off["v"]["data"], off["free"]["data"])
        for seed in range(3):
            case = 4*seed+1
            self.assertEqual(film(off,case,24), film(off,case+2,24))
            self.assertEqual(film(on,case,0), film(on,case+2,0))
            self.assertNotEqual(film(on,case,24), film(on,case+2,24))
        for file, expected in [("flow-hint/branch.pt", self.data["branch_sha256"]),
                               ("dit-lab/weights.pt", self.data["base_hashes"]["dit"]),
                               ("latent-lab/weights.pt", self.data["base_hashes"]["vae"])]:
            self.assertEqual(hashlib.sha256((ROOT/file).read_bytes()).hexdigest(), expected)

    def test_first_unfiltered_miss_and_denominators(self):
        scores = self.data["scores"]
        self.assertEqual([scores[k]["n"] for k in ("0.0","0.5","1.0")], [224]*3)
        self.assertEqual([scores[k]["strict"] for k in ("0.0","0.5","1.0")], [1,199,215])
        miss = self.data["first_error"]
        self.assertEqual(miss["measured"][:5],miss["reference"][:5])
        self.assertEqual((miss["measured"][5],miss["reference"][5]),([12,2],[13,2]))

    @unittest.skipIf(torch is None, "检查训练梯度需要 PyTorch")
    def test_zero_exits_open_first_then_hint_and_backbone_stays_frozen(self):
        device = torch.device("cpu")
        lab.setup(device,437)
        _, base, _ = lab.models(ROOT,device)
        branch = lab.PointBranch(base)
        x = lab.initial_states(2,device)
        t = torch.full((2,1),.5)
        direction = torch.tensor([lab.LEFT,lab.RIGHT])
        points = torch.tensor([[7,3],[7,7]])
        hints = lab.hint_map(points)
        self.assertEqual(hints.flatten(1).argmax(1).tolist(), [3*16+7,7*16+7])
        pred = lab.velocity(base,branch,x,t,direction,hints,1)
        self.assertTrue(torch.equal(pred,base(x,t,direction)))
        pred.square().mean().backward()
        self.assertGreater(float(branch.exits[0].weight.grad.abs().sum()),0)
        self.assertEqual(float(branch.hint.weight.grad.abs().sum()),0)
        torch.optim.SGD(branch.parameters(),lr=.1).step()
        branch.zero_grad()
        lab.velocity(base,branch,x,t,direction,hints,1).square().mean().backward()
        self.assertGreater(float(branch.hint.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in base.parameters()))
