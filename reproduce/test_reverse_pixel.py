"""用正向联合密度、区间积分及独立抽样，核对一步反向的均值、方差和分组概率。"""
import json
import math
from pathlib import Path
import tempfile
import unittest

from posterior_lab import BACKGROUND, BODY, SIGNAL, SIGMA, estimate, phi
from reverse_pixel import (EDGES, WINDOW, distributions, draws, finite_window, forward_check,
                           forward_pairs, step_parameters)
from reverse_pixel_figures import render


class ReversePixelTests(unittest.TestCase):
    def test_reverse_mixture_equals_bayes_forward_density(self):
        for x in (-1.2,.04,1.4):
            for target in (.82,.90,.96,.99):
                r=step_parameters(x,target)
                for v in (-1.3,-.5,.04,.65,1.2):
                    early_sd=math.sqrt(1-target*target)
                    prior=sum(phi((v-target*y)/early_sd) for y in (BODY,BACKGROUND))/(2*early_sd)
                    extra=math.sqrt(r["beta"])
                    direct=prior*phi((x-r["ratio"]*v)/extra)/extra/estimate(x)["density"]
                    p,s=r["p"],r["std"]
                    reverse=(p*phi((v-r["body"])/s)+(1-p)*phi((v-r["back"])/s))/s
                    self.assertAlmostEqual(direct,reverse,places=10)
                epsilon_center=(x-r["beta"]*r["noise"]/SIGMA)/r["ratio"]
                self.assertAlmostEqual(r["mean"],epsilon_center,places=12)
                second=r["within"]+r["p"]*r["body"]**2+(1-r["p"])*r["back"]**2
                self.assertAlmostEqual(r["variance"],second-r["mean"]**2,places=12)

    def test_last_step_is_a_mixture_of_atoms(self):
        for x in (-2,.04,2):
            s=step_parameters(x,1.)
            b=distributions(s)
            self.assertEqual(sum(v>0 for v in b["exact"]),2)
            self.assertEqual(sum(v>0 for v in b["ddpm"]),1)
            self.assertEqual(b["ddpm"],b["ddim"])
            self.assertEqual(s["within"],0)
            self.assertAlmostEqual(draws(s,0,3)["exact"],BODY)
            self.assertAlmostEqual(draws(s,1-1e-9,-3)["exact"],BACKGROUND)
        with self.assertRaises(ValueError):
            step_parameters(.04,SIGNAL)

    def test_printed_numbers_and_tails(self):
        s=step_parameters(.04,.96)
        b=distributions(s)
        for row in b.values():
            self.assertEqual(len(row),len(EDGES)+1)
            self.assertAlmostEqual(sum(row),1)
            self.assertGreaterEqual(min(row),0)
        self.assertAlmostEqual(b["exact"][9],.02643266,places=7)
        self.assertAlmostEqual(b["ddpm"][9],.22836362,places=7)
        self.assertAlmostEqual(draws(s,.25,1)["ddpm"],.317704,places=5)
        self.assertAlmostEqual(s["ddim"],.0564559172)
        result=forward_check(forward_pairs(.96,65536,17),.04,.96)
        self.assertEqual((result["kept"],result["counts"][9]),(2299,75))
        self.assertEqual(sum(result["counts"]),result["kept"])

    def test_forward_integration_matches_reverse_window(self):
        for x,target in ((-1.85,.84),(.04,.96),(1.5,1),(-.12,.9999999)):
            wanted=finite_window(x,target)
            count=180
            sums=[0.]*len(wanted)
            mass=0.
            for i in range(count+1):
                v=x-WINDOW+i*2*WINDOW/count
                weight=estimate(v)["density"]*(1 if i in (0,count) else 4 if i%2 else 2)
                row=distributions(step_parameters(v,target))["exact"]
                mass+=weight
                for j,p in enumerate(row):
                    sums[j]+=weight*p
            self.assertLess(max(abs(v-t/mass) for v,t in zip(wanted,sums)),1e-7)
        almost=finite_window(.04,math.nextafter(.8,1.) if hasattr(math,"nextafter") else .8000000000000002)
        self.assertAlmostEqual(almost[9],1,places=6)

    def test_standalone_custom_settings(self):
        with tempfile.TemporaryDirectory() as work:
            out=Path(work)
            render(out,-.1,.95,15,73,.91,-.33)
            html=(out/"reverse-step.html").read_text(encoding="utf-8")
            data=json.loads(html.split('id="data">')[1].split('</script>')[0])
            self.assertIn(-.1,data["xs"])
            self.assertIn(.95,data["targets"])
            self.assertEqual(data["initialDraw"],[.91,-.33])
            self.assertNotIn('href="posterior.html"',html)
            self.assertEqual(len(data["tickets"]),256)
