"""检查同一状态上两次提示的对照，以及网页实际计算的网络。"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from hint_report import values, render


class HintProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parent
        cls.datafile = cls.source.parent/"control-lab/probe.json"
        cls.data = json.loads(cls.datafile.read_text())

    def test_change_is_confined_to_one_prompt_frame(self):
        data = self.data["answers"]
        points = zip(data["left"]["points"],data["moved"]["points"])
        self.assertEqual([(f,a,b) for f,(a,b) in enumerate(points) if a!=b],[(2,[6,5],[10,5])])
        ref, moved = (values(data[key]["clean"]) for key in ("left","moved"))
        mae = [sum(abs(moved[i]-ref[i]) for i in range(f*192,(f+1)*192))/192 for f in range(6)]
        self.assertEqual([round(v,4) for v in mae],[.0072,.0214,.0695,.0215,.0038,.0044])

    def test_feature_units_and_pixel_units(self):
        idx = (2*32)*48+2*8+3
        low = values(self.data["fixed"]["low"])[idx]
        res = values(self.data["answers"]["moved"]["coarse"])[idx]
        self.assertAlmostEqual(low+.5*res,.3859,places=4)
        idx = 2*192+5*16+7
        x = values(self.data["fixed"]["x"])[idx]
        for key in ("left","moved","off","empty"):
            ans = self.data["answers"][key]
            expected = self.data["a"]*x-self.data["sigma"]*values(ans["v"])[idx]
            self.assertAlmostEqual(expected,values(ans["clean"])[idx],places=6)
        self.assertNotAlmostEqual(values(self.data["answers"]["empty"]["clean"])[idx],
                                  values(self.data["answers"]["off"]["clean"])[idx],places=2)

    def test_standalone_export(self):
        with tempfile.TemporaryDirectory() as working:
            out = Path(working)
            shutil.copyfile(self.datafile,out/"probe.json")
            render(out)
            self.assertTrue((out/"hint-features.svg").is_file())
            page = (out/"edit-points.html").read_text()
            self.assertNotIn('<script src=',page)
            self.assertNotIn('__DATA__',page)
            self.assertNotIn('__NETWORK__',page)

    @unittest.skipUnless(shutil.which("node"),"独立浏览器算子数值校验需 Node.js 18+")
    def test_browser_prediction_matches_pytorch(self):
        major=int(subprocess.check_output(["node","-p","process.versions.node.split('.')[0]"],text=True))
        if major<18:
            self.skipTest("需要 Node.js 18 或更高版本")
        result=subprocess.run(["node",str(self.source/"verify_hint.js"),str(self.datafile)],check=True,
                              capture_output=True,text=True)
        self.assertIn("149760",result.stdout)


if __name__ == "__main__":
    unittest.main()
