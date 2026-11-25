"""独立核对混合坐标、条件平均、每个记录节点以及四次采样的终点。"""
import hashlib
import json
import math
from pathlib import Path
import unittest

from clock_report import Recording
from hint_report import values
from path_coordinates import coefficients, known_pair, line_coordinates, next_step

LAB = Path(__file__).resolve().parents[1]/"pixel-lab"


class CoordinateTests(unittest.TestCase):
    def test_training_pair_and_endpoints(self):
        for tau in (0., .03, .25, .5, 4/7, .98, 1.):
            a, b = coefficients(tau)
            for clean, noise in ((.8,-1.),(.8,1.),(-1.2,-.25)):
                p = known_pair(clean,noise,a)
                self.assertAlmostEqual(p["converted"], clean-noise)
                self.assertAlmostEqual(p["line"], tau*clean+(1-tau)*noise)
                self.assertAlmostEqual(p["tau"], tau)
                self.assertAlmostEqual(a*a+b*b,1.)
        for args in ((.8,float("nan"),.5),(.8,1,-1)):
            with self.assertRaises(ValueError):
                known_pair(*args)
        with self.assertRaises(ValueError):
            coefficients(1.1)

    def test_two_source_conditional_expectation(self):
        # 干净值取两个可能的亮度，噪声为标准正态。分别计算两种坐标下的后验权重。
        clean = [-.7,.95]
        prior = [.3,.7]
        for tau in (.1,.4,.75):
            a,b = coefficients(tau)
            for line in (-.45,.1,.75):
                x = line*(a+b)
                z_arc = [(x-a*y)/b for y in clean]
                z_line = [(line-tau*y)/(1-tau) for y in clean]
                def average(labels,noise,sigma):
                    weights = [p*math.exp(-z*z/2)/sigma for p,z in zip(prior,noise)]
                    return sum(v*w for v,w in zip(labels,weights))/sum(weights)
                v = average([a*z-b*y for y,z in zip(clean,z_arc)],z_arc,b)
                u = average([y-z for y,z in zip(clean,z_line)],z_line,1-tau)
                self.assertAlmostEqual(line_coordinates(x,v,a,b)[2],u)

    def test_whole_archive_next_step(self):
        r = Recording(LAB/"clock-data.js")
        count = r.cases*r.frames*r.plane
        error = 0.
        for n,(a,b,c,d) in enumerate(r.coefficients):
            for j in range(n*count,(n+1)*count):
                t,x,u = line_coordinates(r.states[j],r.velocities[j],a,b)
                new = x+(c/(c+d)-t)*u
                error = max(error,abs(new-r.states[j+count]/(c+d)))
        self.assertLess(error,1e-6)
        p=next_step(r,0,12,2,5,7)
        self.assertAlmostEqual(p["euler"],.396763603,places=8)
        self.assertGreater(p["raw_v"],p["line"])
        with self.assertRaises(ValueError):
            next_step(r,0,r.steps,0,0,0)

    def test_endpoints_and_shared_fingerprints(self):
        data=json.loads((LAB/"path-runs.json").read_text())
        self.assertEqual(data["shape"],[8,6,12,16])
        for fname,key in (("weights.pt","weights"),("clock-data.js","record")):
            self.assertEqual(hashlib.sha256((LAB/fname).read_bytes()).hexdigest(),data["hashes"][key])
        reference=values(data["rows"]["ddim"]["end"])
        for mode,row in data["rows"].items():
            output=values(row["end"])
            self.assertEqual(len(output),9216)
            diff=[a-b for a,b in zip(output,reference)]
            self.assertAlmostEqual(math.sqrt(sum(d*d for d in diff)/len(diff)),row["rmse"],places=11)
            self.assertEqual(max(abs(d) for d in diff),row["max_error"])
            for case in range(8):
                part=diff[case*1152:(case+1)*1152]
                self.assertAlmostEqual(math.sqrt(sum(v*v for v in part)/1152),row["per_case"][case],places=11)
            self.assertEqual(len(row["nodes"]),25)
            self.assertTrue(all(a<b for a,b in zip(row["nodes"],row["nodes"][1:])))
            self.assertAlmostEqual(row["nodes"][-1],1.)
            if mode=="uniform":
                self.assertIsNone(row["same_node_rmse"])
                self.assertEqual(row["nodes"][6],.25)
            else:
                self.assertAlmostEqual(row["nodes"][6],1-1/math.sqrt(2),places=6)
                self.assertAlmostEqual(row["same_node_rmse"][-1],row["rmse"],places=9)
        self.assertLess(max(data["rows"]["matched"]["same_node_rmse"]),1e-5)
        self.assertGreater(data["rows"]["raw"]["rmse"],2)
        self.assertGreater(data["rows"]["uniform"]["rmse"],.01)
        self.assertLess(data["archived_rmse"],1e-5)
