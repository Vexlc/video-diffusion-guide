"""用反传、有限差分、旧版录像和冻结基座的指纹交叉核对零出口课程。"""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from gate_lesson import deltas, render, return_sum, weight_sum
from hint_report import values

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "control-lab"


class GateTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((LAB / "gate-training.json").read_text())
        cls.probe = json.loads((LAB / "probe.json").read_text())

    def test_fixed_response_and_checkpoints(self):
        record = self.record
        stages = record["response"]
        self.assertEqual([s["updates"] for s in stages], [0, 1, 2, 3200])
        self.assertEqual(record["points"][0][2], [6, 5])
        self.assertEqual(record["points"][1][2], [10, 5])
        self.assertEqual([i for i,(a,b) in enumerate(zip(*record["points"])) if a != b], [2])
        for stage in stages:
            base, edit = deltas(stage)
            self.assertEqual(len(base), 1152)
            self.assertAlmostEqual(stage["base_max"], max(map(abs, base)), delta=3e-7)
            self.assertAlmostEqual(stage["edit_max"], max(map(abs, edit)), delta=3e-7)
            for f, avg in enumerate(stage["mean_by_frame"]):
                self.assertAlmostEqual(avg, sum(map(abs, edit[f*192:(f+1)*192]))/192, delta=2e-7)
        self.assertEqual(stages[0]["base_max"], 0)
        self.assertGreater(stages[1]["base_max"], 0.2)
        self.assertEqual(stages[1]["edit_max"], 0)
        self.assertEqual(stages[1]["hint_weight_max"], 0)
        self.assertGreater(stages[2]["hint_weight_max"], 0)
        self.assertAlmostEqual(stages[2]["edit_max"], 5.26905e-5, delta=1e-7)
        for key, answer in (("a","left"), ("b","moved")):
            expected = values(self.probe["answers"][answer]["v"])
            actual = values(stages[-1][key])
            self.assertLess(max(abs(a-b) for a,b in zip(actual,expected)), 3e-5)

    def test_parameter_gradients_three_ways(self):
        for batch in self.record["batches"]:
            for o in range(3):
                bias = math.fsum(values(batch["g"])[o::3])
                self.assertAlmostEqual(bias, batch["db"][o], delta=3e-7)
                for i in range(3):
                    with self.subTest(batch=batch["batch"], output=o, channel=i):
                        result = weight_sum(batch, o, i)
                        self.assertEqual(result["count"], 6912)
                        self.assertAlmostEqual(result["total"], result["autograd"], delta=2e-7)
                        self.assertAlmostEqual(result["central"], result["autograd"], delta=2e-7)
                        if batch["batch"] == 1:
                            grad = result["autograd"]
                            predicted = -self.record["lr"]*grad/(abs(grad)+self.record["adam_eps"])
                            self.assertAlmostEqual(predicted, result["after"], delta=4e-10)

    def test_returned_gradients_and_weight_freeze(self):
        self.assertEqual((self.record["base_delta"], self.record["frozen_grads"]), (0, 0))
        old = json.loads((LAB / "numbers.json").read_text())
        for batch, expected in zip(self.record["batches"], old["gradients"]):
            for name, n in expected.items():
                self.assertAlmostEqual(n, batch["norms"][name], delta=1e-5)
            for i in range(3):
                for f,y,x in ((2,2,3), (0,0,0), (5,5,7)):
                    result = return_sum(batch, i, f, y, x)
                    self.assertEqual(len(result["terms"]), 32)
                    self.assertAlmostEqual(result["total"], result["autograd"], delta=2e-11)
        for name, path in (("base",ROOT/"pixel-lab/weights.pt"),("probe",LAB/"probe.json"),("branch",LAB/"branch.pt")):
            self.assertEqual(self.record["hashes"][name], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_standalone_regeneration(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            render(self.record, directory)
            html = (directory / "gate-training.html").read_text()
            self.assertNotIn("__GATES__", html)
            self.assertNotIn('href="edit-points.html"', html)
            for name in ("gate-response.svg","gate-product.svg"):
                self.assertEqual((directory/name).read_bytes(), (LAB/name).read_bytes())
