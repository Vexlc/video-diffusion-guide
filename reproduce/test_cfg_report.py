"""用作者模型的实际输入与输出，检查文字和深度的配对对照。"""
import json
import struct
import unittest

from cfg_report import ROOT, guided, readout
from depth_report import unpack


class GuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "depth-lab/cfg.json").read_text(encoding="utf-8"))
        cls.runs = cls.data["runs"]

    def test_each_recorded_scheduler_input(self):
        for key, run in self.runs.items():
            self.assertEqual([s["n"] for s in run["samples"]], [0, 10, 19])
            for sample in run["samples"]:
                with self.subTest(key=key, step=sample["n"]):
                    def plane(field):
                        raw = unpack(sample[field])
                        self.assertEqual(len(raw), 2 * 5 * 24 * 32)
                        return [v for v, in struct.iter_unpack("<e", raw)]
                    pos, actual = plane("pos"), plane("actual")
                    if run["w"] > 1:
                        self.assertEqual(sample["batch"], 2)
                        self.assertEqual([guided(r, p, run["w"]) for r, p in zip(plane("ref"), pos)], actual)
                    else:
                        self.assertEqual(sample["batch"], 1)
                        self.assertNotIn("ref", sample)
                        self.assertEqual(pos, actual)
                    if run["scale"] == 0:
                        self.assertTrue(all(v == 0 for v in plane("residual")))

    def test_same_state_first_step(self):
        first = {key: run["samples"][0] for key, run in self.runs.items()}
        self.assertEqual(len({s["state_sha256"] for s in first.values()}), 1)
        for scale in (0, 1):
            a, b = [first[f"w{w}-d{scale}"] for w in (2, 7.5)]
            for field in ("ref", "pos", "residual"):
                self.assertEqual(unpack(a[field]), unpack(b[field]))
            self.assertNotEqual(unpack(a["actual"]), unpack(b["actual"]))
        a, b = first["w7.5-d1"], first["w7.5-d1-empty"]
        self.assertEqual(a["pos"], b["pos"])
        self.assertNotEqual(a["ref"], b["ref"])
        self.assertNotEqual(a["actual"], b["actual"])
        self.assertEqual(self.runs["w0-d1"]["film"], self.runs["w1-d1"]["film"])
        self.assertEqual(self.runs["w0-d1"]["samples"], self.runs["w1-d1"]["samples"])

    def test_printed_readout_and_shared_depth_condition(self):
        other = json.loads((ROOT / "depth-lab/record.json").read_text(encoding="utf-8"))
        self.assertEqual(self.data["maps"], other["maps"]["A"])
        for scale in (0, 1):
            self.assertEqual(other["runs"][f"A{scale}"]["film"], self.runs[f"w7.5-d{scale}"]["film"])
        for case, answer in (("w2-d1", -1.18359375), ("w7.5-d1", -1.263671875)):
            values = readout(self.data, case, 0, 2, 16, 12)
            self.assertEqual(values["actual"], answer)
            self.assertEqual(values["difference"], -0.0146484375)
            self.assertEqual(values["ref"], -1.154296875)
            self.assertEqual(values["pos"], -1.1689453125)
            self.assertEqual(values["residual"], -0.00873565673828125)
