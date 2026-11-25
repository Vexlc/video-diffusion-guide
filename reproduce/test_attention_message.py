"""用投影层实际读到的值与整网速度核对跨时间组的计算。"""
import json
import unittest

from attention_report import ROOT, readout


class AttentionMessageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT/"dit-lab/message.json").read_text(encoding="utf-8"))

    def test_all_edges_and_interventions(self):
        for mode, run in self.data["runs"].items():
            for q in range(18):
                self.assertAlmostEqual(sum(run["weights"][q]), 1., places=6)
                for j in range(18):
                    entry = readout(self.data, q, j, mode)
                    self.assertAlmostEqual(entry["weight"], entry["weight_from_qk"], places=6)
                    self.assertAlmostEqual(entry["sum"], entry["received"], places=6)
                    self.assertAlmostEqual(entry["received_plus_one"]-entry["received"], entry["weight"], places=6)
                    if mode=="local" and q//6!=j//6:
                        self.assertEqual(entry["weight"], 0.)
                        self.assertEqual(entry["velocity"], entry["velocity_plus_one"])

    def test_paper_example_and_frame_order(self):
        r = readout(self.data)
        self.assertEqual([round(x,4) for x in r["group_sums"]], [.0408,-.0692,-.1878])
        self.assertEqual([round(r[k],4) for k in ("weight","value","dot","mass","total","received_plus_one")],
                         [.1809,-1.2406,-2.1634,.7351,4.0640,-.0353])
        self.assertGreater(abs(r["velocity_plus_one"]-r["velocity"]), .005)
        for f,frame in enumerate(self.data["frames"]):
            lit=[x for row in frame for x,v in enumerate(row) if v==255]
            self.assertEqual(len(lit),9)
            self.assertEqual(sum(lit)/len(lit),13-f)
