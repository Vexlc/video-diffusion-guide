"""用标准库逐格重算库的真实返回值，并核对两组时间表。"""
import json
import math
from pathlib import Path
import unittest

from hint_report import values
from scheduler_report import check_files, read_cell, recompute

ROOT = Path(__file__).resolve().parents[1]


class SchedulerReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT/"scheduler-lab/record.json").read_text())

    def test_steps_and_clocks(self):
        check_files(self.data, ROOT)
        for kind in ("ddim", "flow"):
            steps=self.data[kind]["steps"]
            self.assertEqual(len(steps),24)
            for i,row in enumerate(steps):
                self.assertEqual(row["i"],i)
                if kind=="ddim":
                    self.assertEqual(row["t"],23-i)
                    a,b,c,d=row["coef"]
                    self.assertAlmostEqual(a, math.sin(i*math.pi/48), places=6)
                    self.assertAlmostEqual(a*a+b*b,1,places=6)
                    self.assertAlmostEqual(c*c+d*d,1,places=6)
                else:
                    self.assertAlmostEqual(row["sigma"],1-i/24,places=6)
                    self.assertAlmostEqual(row["following"],1-(i+1)/24,places=6)
                    self.assertAlmostEqual(row["t"],row["sigma"]*1000,delta=.0001)
                if i:
                    self.assertEqual(steps[i-1]["next"]["data"],row["x"]["data"])

    def test_every_recorded_update(self):
        for kind in ("ddim","flow"):
            for row in self.data[kind]["steps"]:
                vectors=[values(row[name]) for name in ("x","v","next","alternate")]
                self.assertEqual({len(v) for v in vectors},{math.prod(self.data[kind]["shape"])})
                outside=0
                for x,v,y,alt in zip(*vectors):
                    first,second=recompute(kind,row,x,v)
                    self.assertAlmostEqual(first,y,delta=1e-6)
                    self.assertAlmostEqual(second,alt,delta=1e-6)
                    if kind=="ddim":
                        a,b,_,_=row["coef"]
                        outside+=int(abs(a*x-b*v)>1)
                if kind=="ddim":
                    self.assertEqual(row["outside"],outside)

    def test_final_films_and_printed_examples(self):
        for kind in ("ddim","flow"):
            self.assertLess(self.data[kind]["reference"]["rmse"],1e-6)
            self.assertGreater(self.data[kind]["changed"]["rmse"],.01)
        self.assertEqual(self.data["flow"]["centers"][:2],[[[7+i,3] for i in range(6)]]*2)
        p=read_cell(self.data,"ddim",13,522)
        self.assertEqual(p["坐标（从零数）"],[2,8,10])
        self.assertAlmostEqual(p["库返回"],.30836,places=5)
        f=read_cell(self.data,"flow",13,30)
        self.assertEqual(f["坐标（从零数）"],[1,0,1,2])
        self.assertAlmostEqual(f["改动后库返回"],1.13586,places=5)
