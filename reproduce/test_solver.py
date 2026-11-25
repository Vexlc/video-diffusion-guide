"""核验同预算比较使用的真实查询、成对统计、图中步长与两种运行环境。"""
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from solver_report import rows_for

ROOT=Path(__file__).resolve().parents[1]
LAB=ROOT/"flow-hint"


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((LAB/"solver.json").read_text())
        cls.old=json.loads((LAB/"numbers.json").read_text())

    def test_all_accepted_steps_and_each_query_position(self):
        for row in self.data["rows"]:
            with self.subTest(mode=row["mode"],budget=row["budget"]):
                path,queries=row["path"],row["queries"]
                self.assertEqual(len(queries),row["budget"])
                self.assertEqual(len(path),row["updates"]+1)
                self.assertEqual(row["at"],row["updates"]/row["steps"])
                mid=row["mode"]=="midpoint"
                for i,x in enumerate(path[:-1]):
                    t,q,u=queries[2*i if mid else i]
                    self.assertAlmostEqual(t,i/row["steps"],places=6)
                    self.assertEqual(q,x)
                    if mid:
                        ht,hx,hu=queries[2*i+1]
                        self.assertAlmostEqual(ht,(i+.5)/row["steps"],places=6)
                        self.assertAlmostEqual(hx,x+u/(2*row["steps"]),places=6)
                        u=hu
                    self.assertAlmostEqual(path[i+1],x+u/row["steps"],places=6)

    def test_fair_pairing_and_fine_grid(self):
        self.assertEqual(self.data["hashes"],{**self.old["base_hashes"],"branch":self.old["branch_sha256"]})
        for b in self.data["budgets"]:
            stop,e,m=rows_for(self.data,b)
            p=self.data["paired"][str(b)]
            self.assertEqual(sum(p.values()),224)
            self.assertEqual((p["both"]+p["euler_only"],p["both"]+p["mid_only"]),(e["strict"],m["strict"]))
            if b<24:
                self.assertIsNone(stop["latent_rmse"])
            else:
                for key in ("path","queries","strict","frame_hits","film","centers","latent_rmse"):
                    self.assertEqual(stop[key],e[key])
                self.assertEqual(e["strict"],self.old["scores"]["1.0"]["strict"])
        ref=self.data["reference"]
        self.assertEqual((ref["strict"],ref["check_strict"],ref["changed_centers"]),(215,215,0))
        self.assertLess(ref["rmse"],.00006)

    def browser(self,*args):
        node=shutil.which("node")
        if not node or int(subprocess.check_output([node,"-p","process.versions.node.split('.')[0]"],text=True))<18:
            self.skipTest("浏览器算子核对需要 Node.js 18+")
        run=subprocess.run([node,str(ROOT/"reproduce/verify_solver.js"),str(LAB),*map(str,args)],
                           check=True,capture_output=True,text=True)
        return json.loads(run.stdout)

    def test_every_budget_and_mode_against_torch_record(self):
        self.assertEqual(self.browser()["runs"],15)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"自选样片需要 PyTorch")
    def test_new_point_and_changed_direction(self):
        import torch
        from solver_lab import custom
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            custom(ROOT,out,torch.device("cpu"),9,8,"left",.5,9318,8)
            self.assertEqual(self.browser(out/"sample.json")["runs"],3)
            self.assertTrue((out/"paths.svg").is_file())
