"""核对讲义中的来源、一次调用的边界和浏览器复算结果。"""
import hashlib
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from clock_report import Recording
from temporal_report import address, differences, possible_sources, values


class TemporalProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=Path(__file__).resolve().parents[1]/"pixel-lab"
        cls.data=json.loads((cls.folder/"temporal.json").read_text())

    def test_same_weights_and_state_as_sampling_movie(self):
        for key,name in (("base_sha256","weights.pt"),("recording_sha256","clock-data.js")):
            self.assertEqual(self.data[key],hashlib.sha256((self.folder/name).read_bytes()).hexdigest())
        recording=Recording(self.folder/"clock-data.js")
        start=recording.plane_offset(0,12,0)
        self.assertEqual(values(self.data["original"]),list(recording.states[start:start+1152]))
        start2=recording.plane_offset(1,12,0)
        self.assertEqual(values(self.data["donor"]),list(recording.states[start2:start2+1152]))
        baseline=values(self.data["runs"]["111"])[:1152]
        self.assertLess(max(abs(a-b) for a,b in zip(baseline,recording.velocities[start:start+1152])),1e-4)

    def test_received_grid_and_boundaries(self):
        for c,source,answer in ((2,1,-.2737),(6,3,-.2595),(12,2,.0250)):
            actual=address(self.data,"111",0,2,c,5,7)
            self.assertEqual(actual["source_frame"],source)
            self.assertAlmostEqual(actual["received"],answer,places=4)
        for mask in self.data["runs"]:
            for layer in range(3):
                limit=self.data["taps"][mask[:layer]]["shape"][1]
                for frame,channel in ((0,0),(5,limit//4),(3,limit-1)):
                    self.assertEqual(address(self.data,mask,layer,frame,channel,0,0)["source_frame"],frame)
        self.assertEqual(address(self.data,"111",2,2,2,11,15)["stored_col"],7)
        with self.assertRaises(ValueError):
            address(self.data,"111",1,2,2,6,7)

    def test_full_matrix_for_every_switch_pattern(self):
        for mask,run in self.data["runs"].items():
            possible=possible_sources(mask)
            for j,row in enumerate(differences(run)):
                for f,m in enumerate(row):
                    self.assertEqual(j in possible[f],abs(f-j)<=mask.count("1"))
                    self.assertEqual(m>0,j in possible[f])
        self.assertEqual([round(x,4) for x in differences(self.data["runs"]["111"])[0]],
                         [.8890,.1880,.0652,.0228,0,0])

    def test_edit_happens_after_first_shift(self):
        test=self.data["edits"][0]
        self.assertEqual(test["edit"],[0,2,2,5,7,1])
        edited=values(test["v"])
        baseline=values(self.data["runs"]["111"])
        means=[sum(abs(edited[p]-baseline[p]) for p in range(f*192,(f+1)*192))/192 for f in range(6)]
        self.assertLess(means[-1],1e-6)
        self.assertEqual([round(m,4) for m in means[:5]],[.0018,.0056,.0116,.0052,.0021])

    @unittest.skipUnless(shutil.which("node"),"独立网页算子校验需 Node 18+")
    def test_javascript_matches_all_recorded_settings(self):
        major=int(subprocess.check_output(["node","-p","process.versions.node.split('.')[0]"],text=True))
        if major<18:
            self.skipTest("需要 Node 18 或更高版本")
        source=Path(__file__).with_name("verify_temporal.js")
        result=subprocess.check_output(["node",str(source),str(self.folder/"temporal.json")],text=True)
        self.assertIn("18432",result)
