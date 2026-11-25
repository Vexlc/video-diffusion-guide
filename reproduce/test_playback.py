"""按源帧、显示像素和模型记录，验证倒放与改条件重新生成的对照。"""
import json
import unittest
from pathlib import Path

from motion_readout import picture, probe_record
from playback_report import PANEL, compare, display, measure


class PlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]/"flow-hint"
        cls.records = json.loads((root/"playback.json").read_text())
        cls.original = json.loads((root/"numbers.json").read_text())

    def test_shared_conditions_and_integrator(self):
        self.assertEqual(self.records["hashes"],{**self.original["base_hashes"],"branch":self.original["branch_sha256"]})
        earlier = probe_record(self.original)
        for seed,group in self.records["seeds"].items():
            self.assertEqual(len({run["x"][0] for run in group.values()}),1)
            for key,run in group.items():
                self.assertEqual(len(run["x"]),25)
                self.assertEqual(len(run["u"]),24)
                for k,u in enumerate(run["u"]):
                    self.assertAlmostEqual(run["x"][k]+u/24,run["x"][k+1],delta=3e-7)
                if seed=="9317" and key in earlier:
                    films=display(run,False)["frames"]
                    for f,frame in enumerate(films):
                        self.assertEqual(frame,bytes(v for row in picture(earlier[key],24,f) for v in row))
                    for k,u in enumerate(run["u"]):
                        self.assertAlmostEqual(u,earlier[key]["u"][k][1],delta=1e-5)

    def test_reversing_only_changes_display_axis(self):
        for group in self.records["seeds"].values():
            a,b,c,d = [display(group[key],reverse) for key,reverse,_ in PANEL]
            self.assertEqual(b["frames"],a["frames"][::-1])
            self.assertEqual(b["u"],a["u"])
            for f in range(5):
                for fps in (3,6,12):
                    p=measure(a,4-f,12,fps)
                    q=measure(b,f,12,fps)
                    self.assertEqual(q["pixels_per_second"],[-v for v in p["pixels_per_second"]])
                    self.assertEqual(p["u"],q["u"])
            self.assertEqual(c["centers"][0],[7,3])
            self.assertEqual(d["centers"][0],[12,3])
            self.assertIsNone(measure(d,5,12,6)["pixels_per_second"])

    def test_shown_picture_differences(self):
        totals={ "9317":36, "9318":40, "9319":38 }
        for seed,group in self.records["seeds"].items():
            a=display(group["right"],True)
            b=display(group["left_end"],False)
            report=compare(a,b)
            self.assertEqual(report["center_hits"],6)
            self.assertEqual(report["different"],totals[seed])
            if seed=="9317":
                self.assertEqual(report["changes"],[8,8,8,4,6,2])
                self.assertEqual(report["sum_abs"],[312,320,102,11,24,4])
                self.assertEqual(report["peaks"][1],{"col":10,"row":4,"a":21,"b":255,"diff":234})
