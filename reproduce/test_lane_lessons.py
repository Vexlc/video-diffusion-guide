import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from lane_lessons import lane_path, lane_speed, noise_problem, train_exit, video_lanes


class FirstNotebook(unittest.TestCase):
    def test_single_file_script_writes_the_notebook_figures_and_accepts_edits(self):
        starter = Path(__file__).resolve().parents[1] / "starter-lab"
        notebook = json.loads((starter / "first-lab.ipynb").read_text(encoding="utf-8"))
        expected = [cell["outputs"][0]["data"]["image/svg+xml"]
                    for cell in notebook["cells"] if cell.get("outputs")]
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            script = folder / "first_lab.py"
            shutil.copyfile(starter / script.name, script)
            def run(*args):
                return subprocess.run([sys.executable, "-I", "-S", str(script), *args],
                                      cwd=folder, text=True, capture_output=True)
            default = run()
            self.assertEqual(default.returncode, 0, default.stderr)
            page = (folder / "my-first-lab.html").read_text(encoding="utf-8")
            self.assertEqual(re.findall(r"<svg\b.*?</svg>", page, re.S), expected)
            output = folder / "实验 2" / "我的结果.html"
            changed = run("--steps", "2", "--out", str(output))
            self.assertEqual(changed.returncode, 0, changed.stderr)
            page = output.read_text(encoding="utf-8")
            figures = re.findall(r"<svg\b.*?</svg>", page, re.S)
            self.assertEqual([a != b for a,b in zip(expected,figures)], [False,True,False,False])
            self.assertIn("终点为 0.3799", page)
            self.assertIn("<td>0.2000</td><td>0.3599</td><td>0.1799</td><td>0.3799</td>", page)
            self.assertNotEqual(run("--steps", "0").returncode, 0)
            self.assertNotEqual(run("--strength", "nan").returncode, 0)

    def test_noise_error_shift_of_the_entire_frame(self):
        original,mixed,estimate = noise_problem(.15)
        self.assertEqual(len(mixed)*len(mixed[0]),192)
        for row,out in zip(original,estimate):
            for y,guess in zip(row,out):
                self.assertAlmostEqual(guess-y,-.1125)

    def test_closed_form_agrees_with_gaussian_weighting(self):
        for t in (0,.1,.5,.83):
            for x in (-1.3,-.2,0,.4,1.8):
                weights=[math.exp(-.5*((x-t*y)/(1-t))**2) for y in (-1,1)]
                expected=sum(w*(y-x)/(1-t) for w,y in zip(weights,(-1,1)))/sum(weights)
                self.assertAlmostEqual(lane_speed(x,t),expected,places=12)
        self.assertAlmostEqual(lane_speed(.4,.5),.52807354,places=7)
        self.assertEqual(lane_path(.4,1),[.4,0.])
        self.assertEqual(lane_path(0.,12),[0.]*13)
        self.assertAlmostEqual(lane_path(.4,12)[-1],1.,places=5)

    def test_shared_lane_and_frozen_branch(self):
        independent,shared,legal,possible=video_lanes(17)
        self.assertEqual(shared,[independent[0]]*6)
        self.assertEqual((legal,possible),(2,64))
        for updates in (0,1,8):
            history,rows=train_exit(updates,1.)
            self.assertAlmostEqual(history[-1],1-.75**updates)
            self.assertEqual(train_exit(updates,0),(history,[5.,5.]))
            self.assertAlmostEqual(sum(rows),10.)

    def test_standalone_notebook_cells_reproduce_the_saved_diagrams(self):
        root=Path(__file__).resolve().parents[1]
        doc=json.loads((root/"starter-lab/first-lab.ipynb").read_text(encoding="utf-8"))
        # Exercise the exact distributed cells, including defaults captured during definition.
        # Separate globals keep the notebook independent of the imported teaching modules.
        namespace={}
        import ast
        seen=[]
        for cell in doc["cells"]:
            if cell["cell_type"]!="code":
                continue
            script=ast.parse(cell["source"])
            if not cell["outputs"]:
                exec(compile(script,"plot-tools","exec"),namespace)
                continue
            last=script.body.pop()
            exec(compile(script,"lesson","exec"),namespace)
            figure=eval(compile(ast.Expression(last.value),"figure","eval"),namespace)
            self.assertEqual(figure._repr_svg_(),cell["outputs"][0]["data"]["image/svg+xml"])
            seen.append(cell["id"])
        self.assertEqual(seen,["noise-code","flow-code","video-code","control-code"])
