"""核对窗口边界、同状态分叉、原评分和实际影片帧的配对。"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from window_report import judge

ROOT=Path(__file__).resolve().parents[1]


class ControlWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/"flow-hint/windows.json").read_text())

    def test_paired_routes(self):
        data=self.data
        old=json.loads((ROOT/"flow-hint/numbers.json").read_text())
        self.assertEqual(data["hashes"],{**old["base_hashes"],"branch":old["branch_sha256"]})
        self.assertEqual(len(set(data["seeds"])),8)
        self.assertEqual(len({str(t) for t in data["targets"]}),28)
        self.assertEqual(len(set(zip(data["seeds"],map(str,data["targets"])))),224)
        grades=judge(data)
        for mode,scale in (("off","0.0"),("full","1.0")):
            self.assertEqual(sum(q["counts"][mode]==6 for q in grades),old["scores"][scale]["strict"])
        self.assertEqual([sum(q["category"]==k for q in grades) for k in ("both","early","late","neither")],
                         [52,12,106,54])
        self.assertEqual(grades[6]["counts"]["early"],6)
        self.assertEqual(data["runs"]["late"]["centers"][6][0],[4,7])

    def test_switch_step_from_same_state(self):
        r=self.data["runs"]
        for before,after in (("early","full"),("late","off")):
            self.assertEqual(r[before]["path"][:13],r[after]["path"][:13])
            self.assertEqual(r[before]["speeds"][:12],r[after]["speeds"][:12])
            vgap=r[before]["speeds"][12]-r[after]["speeds"][12]
            self.assertNotEqual(vgap,0)
            self.assertAlmostEqual(r[before]["path"][13]-r[after]["path"][13],vgap/24,places=6)
        for row in r.values():
            self.assertEqual(len(row["path"]),25)
            for k,u in enumerate(row["speeds"]):
                self.assertAlmostEqual(row["path"][k]+u/24,row["path"][k+1],places=6)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"模型复跑需要 PyTorch")
    def test_zero_and_full_width_on_another_seed(self):
        from control_window import run
        with tempfile.TemporaryDirectory() as tmp:
            for width,matching in ((0,"off"),(24,"full")):
                out=Path(tmp)/str(width)
                run(ROOT,out,width=width,sample=True,col=9,row=7,direction="left",seed=9318,strength=.5)
                record=json.loads((out/"windows.json").read_text())
                for mode in ("early","late"):
                    for field in ("path","speeds","film","centers"):
                        self.assertEqual(record["runs"][mode][field],record["runs"][matching][field])
                self.assertTrue((out/"window-scores.html").is_file())
