"""用独立密度积分、固定训练样本和网页计算，核对第 8、12 页的推断。"""
import json
import math
import shutil
import subprocess
import unittest
from pathlib import Path

from posterior_lab import (BACKGROUND, BODY, MODES, RADIUS, SIGNAL, SIGMA, estimate, noise_for,
                           phi, sample_window, slope_from_density, window_expected)


class PosteriorTests(unittest.TestCase):
    def test_same_input_and_unequal_likelihoods(self):
        self.assertAlmostEqual(noise_for(.04,BODY),-1.)
        self.assertAlmostEqual(noise_for(.04,BACKGROUND),.8)
        for mode,n,prob in (("hidden",.64,.45512111),("shared",.64,.80845465),
                            ("shared",-.44,.14185106),("independent",.64,.45512111)):
            result=estimate(.04,mode,n)
            self.assertAlmostEqual(result["body"],prob,places=7)
            self.assertAlmostEqual(SIGNAL*result["clean"]+SIGMA*result["noise"],.04)
            for g in [-1.,0.,.8]:
                loss=prob*(g+1)**2+(1-prob)*(g-.8)**2
                self.assertAlmostEqual(loss,result["loss"]+(g-result["noise"])**2,places=7)

    def test_score_is_a_partial_derivative(self):
        for x in (-1.5,.04,.6,1.4):
            for n in (-1.4,-.44,.64,1.5):
                for mode in MODES:
                    self.assertAlmostEqual(estimate(x,mode,n)["score"],slope_from_density(x,mode,n),places=6)

    def test_numerical_joint_window_integration(self):
        # 直接把两种隐状态展开为四个（本帧、邻帧）组合，在观测平面做中点求积。
        x,n=.04,-.44
        for mode in MODES:
            mass=moment=bright=0.
            for y in [BODY,BACKGROUND]:
                for other in [BODY,BACKGROUND]:
                    if mode=="shared" and y!=other:
                        continue
                    for i in range(48):
                        z=noise_for(x+RADIUS*((2*i+1)/48-1),y)
                        for j in range(48):
                            z1=noise_for(n+RADIUS*((2*j+1)/48-1),other)
                            w=phi(z)*(phi(z1) if mode!="hidden" else 1)
                            mass+=w;moment+=w*z;bright+=w*(y==BODY)
            wanted=window_expected(x,mode,n)
            self.assertAlmostEqual(bright/mass,wanted["body"],places=5)
            self.assertAlmostEqual(moment/mass,wanted["noise"],places=5)

    def test_default_sample_and_empty_window(self):
        sample=sample_window(.04,"shared",.64)
        self.assertEqual((sample["kept"],sample["body_count"]),(411,336))
        self.assertAlmostEqual(sample["noise"],-.66337317)
        self.assertEqual(sample_window(1.5,"shared",-1.5,1)["noise"],None)

    @unittest.skipUnless(shutil.which("node"),"网页数值核对需要 Node")
    def test_browser_computation_matches_python(self):
        inputs=[(x,mode,n,count) for x in (-1.5,.04,1.5) for n in (-1.5,-.44,.64,1.5)
                for mode in MODES for count in (64,4096)]
        program=Path(__file__).with_name("posterior_math.js").read_text()+"""
const params=JSON.parse(require("fs").readFileSync(0,"utf8")), set=trainingSet(4096);
console.log(JSON.stringify(params.map(([x,mode,n,count])=>
  [estimate(x,mode,n),windowExpected(x,mode,n),sampleWindow(set,count,x,mode,n)])));
"""
        output=json.loads(subprocess.check_output(["node","-e",program],input=json.dumps(inputs),text=True))
        for (x,mode,n,count),results in zip(inputs,output):
            expected=(estimate(x,mode,n),window_expected(x,mode,n),sample_window(x,mode,n,count))
            for actual,want in zip(results,expected):
                self.assertEqual(actual.keys(),want.keys())
                for key,v in want.items():
                    if v is None:
                        self.assertIsNone(actual[key])
                    else:
                        self.assertAlmostEqual(actual[key],v,places=7)
