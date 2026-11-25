import math
import unittest

from flow_coupling import euler
from hunyuan_trace import address, guidance_features, knots, layout, solve


class SourceTraceTests(unittest.TestCase):
    def test_alignment_and_release_shape(self):
        self.assertEqual(layout(129,720,1280)["count"],118800)
        self.assertEqual(layout(65,361,639)["canvas"],(65,368,640))
        self.assertEqual(layout(1,1,1)["latent"],(1,16,1,2,2))
        with self.assertRaises(ValueError):
            layout(18,64,96)

    def test_flat_addresses_are_one_to_one(self):
        info = layout(17,64,96)
        grid = info["grid"]
        self.assertEqual(address(grid,57,22),((2,1,3),(0,5,2,3,6)))
        seen = {address(grid,i,j)[1] for i in range(info["count"]) for j in range(64)}
        _,c,t,h,w = info["latent"]
        expected = {(0,ch,f,r,col) for f in range(t) for r in range(h) for col in range(w) for ch in range(c)}
        self.assertEqual(seen,expected)

    def test_grid_moves_but_endpoints_hold(self):
        self.assertEqual(knots(4,3),[1.,.9,.75,.5,0.])
        for strength in (1,3,9):
            points = knots(30,strength)
            self.assertEqual((len(points),points[0],points[-1]),(31,1.,0.))
            self.assertTrue(all(a>b for a,b in zip(points,points[1:])))

    def test_clock_change_preserves_uniform_euler(self):
        for n in (1,4,8,30):
            for initial in (-1,.2,2):
                expected = euler(initial,n)
                trace = solve(knots(n,1),initial)
                for x,y in zip(trace,expected):
                    self.assertAlmostEqual(x,y,places=12)

    def test_guidance_order_and_frequency(self):
        for g in (0,.25,6,9):
            values = guidance_features(g)
            self.assertEqual(len(values),256)
            self.assertAlmostEqual(values[96],math.cos(g))
            self.assertAlmostEqual(values[224],math.sin(g))
            self.assertTrue(all(abs(c*c+s*s-1)<1e-12 for c,s in zip(values[:128],values[128:])))
