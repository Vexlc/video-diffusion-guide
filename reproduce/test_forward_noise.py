"""核对同一格、整幅图、成对分布和开源库的正向加噪。"""
import importlib.util
import math
import unittest
from importlib.metadata import PackageNotFoundError, version

from forward_noise import COUNT, MODES, PIXEL, bins, image_inputs, pictures, scatter, theory, two_steps


class ForwardNoiseTests(unittest.TestCase):
    def test_same_sample_and_independent_increments(self):
        ref=two_steps(.8,-1,1)
        self.assertAlmostEqual(ref["first"],.04)
        self.assertAlmostEqual(ref["chain"],.6914378277661477)
        self.assertAlmostEqual(ref["reuse"],-.32)
        for r in (0.,.23,.75,1.):
            norm=two_steps(0,0,0,r)
            self.assertAlmostEqual((.6*r)**2+norm["new"]**2,norm["sigma"]**2)
            plane=pictures(31,r,-.2,-.7)
            self.assertEqual(plane["clean"][PIXEL],.8)
            self.assertLess(max(abs(a-b) for a,b in zip(plane["direct"],plane["chain"])),1e-12)

    def test_theory_and_measured_windows(self):
        theory13=theory(.75,.04)
        for key,rho,center in zip(MODES,(.5625,1,0),(.03,-.32,.48)):
            self.assertAlmostEqual(theory13[key][0],rho)
            self.assertAlmostEqual(theory13[key][1],center)
        points,rows=scatter(17,.75,.04)
        for key in MODES:
            self.assertEqual(len(points[key]),COUNT)
            self.assertEqual(rows[key]["n"],185)
            self.assertAlmostEqual(sum(bins(points[key],at=.04)),1.)
            self.assertLess(abs(rows[key]["corr"]-theory13[key][0]),.035)
            self.assertLess(abs(rows[key]["window_mean"]-theory13[key][1]),.03)
        _,empty=scatter(17,.75,100.)
        self.assertTrue(all(row["window_mean"] is None for row in empty.values()))

    def test_exact_limits(self):
        for x,y in ((.2,-1.4),(-.1,.47)):
            keep_all=two_steps(.8,x,y,1.)
            self.assertAlmostEqual(keep_all["chain"],keep_all["first"])
            self.assertAlmostEqual(keep_all["chain"],keep_all["reuse"])
            keep_none=two_steps(.8,x,y,0.)
            self.assertEqual(keep_none["chain"],keep_none["fresh"])

    def test_pinned_diffusers_whole_frame(self):
        try:
            installed=version("diffusers")
        except PackageNotFoundError:
            installed=None
        if installed!="0.35.1" or not importlib.util.find_spec("torch"):
            self.skipTest("可选的 Diffusers 0.35.1 与 PyTorch 对照")
        import torch
        from diffusers import DDPMScheduler
        vals=image_inputs(17)
        clean,e1,e2=[torch.tensor(seq).reshape(1,1,30,48) for seq in zip(*vals)]
        e1.view(-1)[PIXEL],e2.view(-1)[PIXEL]=-1.,1.
        lib=DDPMScheduler(num_train_timesteps=2,trained_betas=[.36,.4375])
        first=lib.add_noise(clean,e1,torch.tensor([0]))
        chain=.75*first+math.sqrt(.4375)*e2
        merged=(.45*e1+math.sqrt(.4375)*e2)/.8
        direct=lib.add_noise(clean,merged,torch.tensor([1]))
        self.assertLess((chain-direct).abs().max().item(),1e-6)
        expected=torch.tensor(pictures(17,.75,-1,1)["chain"]).reshape_as(chain)
        torch.testing.assert_close(direct,expected,atol=1e-6,rtol=0)
