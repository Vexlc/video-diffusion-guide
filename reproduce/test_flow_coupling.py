import random
import unittest

import flow_coupling as flow


class CouplingTests(unittest.TestCase):
    def test_conditional_mean_and_endpoints(self):
        for t in (0.,.25,.5,.75,1.):
            for x in (-1.2,0.,.25,.8):
                self.assertAlmostEqual(flow.arrows(x,t)["v"], flow.velocity(x,t))
        a = flow.arrows(.25,.5)
        self.assertAlmostEqual(a["p"],.8320183851,places=9)
        self.assertAlmostEqual(a["v"],1.8249176649,places=9)
        self.assertAlmostEqual(flow.velocity(0,.5),0.)
        self.assertAlmostEqual(flow.euler(.75,1)[-1],0.)

    def test_quantile_checks_field_independently(self):
        for q in (.03,.25,.75,.97):
            for t in (.1,.3,.5,.9):
                x = flow.quantile(q,t)
                self.assertAlmostEqual(flow.cdf(x,t),q,places=12)
                h = 1e-5
                derivative = (flow.quantile(q,t+h)-flow.quantile(q,t-h))/(2*h)
                self.assertAlmostEqual(derivative,flow.velocity(x,t),places=6)
            z = flow.quantile(q,0)
            end = flow.quantile(q,1)
            self.assertLess(abs(flow.euler(z,64)[-1]-end),abs(flow.euler(z,4)[-1]-end))

    def test_identical_ends_different_pairs(self):
        starts = [i/500 for i in range(-1000,1001)]
        targets = [1.5*z+.25 for z in starts]
        table = flow.fit_displacements(starts,targets)
        shuffled = targets.copy()
        random.Random(41).shuffle(shuffled)
        self.assertEqual(sorted(shuffled),targets)
        unpaired = flow.fit_displacements(starts,shuffled)
        probes = [-1.15,-.74,.51,.91,1.74]
        answers = [1.5*z+.25 for z in probes]
        self.assertLess(flow.mae([flow.student(z,table) for z in probes],answers),1e-12)
        self.assertGreater(flow.mae([flow.student(z,unpaired) for z in probes],answers),.6)
