import math
import unittest

from prediction_lab import HEADS, car_row, exercise, read_prediction, scene, trace
from author_endpoints import check, load_record


class PredictionTests(unittest.TestCase):
    def test_author_endpoint_order_against_the_independent_path_formula(self):
        record = load_record()
        check(record)
        forward, backward = record["orders"]
        for k,t in enumerate(record["times"]):
            for order in (forward,backward):
                _,_,x,v = trace(order["end"],order["start"],t)["line"]
                self.assertAlmostEqual(order["x_t"][k],x,places=12)
                self.assertAlmostEqual(order["dx_t"][k],v,places=12)
        self.assertEqual(forward["x_t"][50],backward["x_t"][50])
        self.assertAlmostEqual(forward["x_t"][25],-.55)
        self.assertAlmostEqual(backward["x_t"][25],.35)
        self.assertEqual(forward["dx_t"][50],-backward["dx_t"][50])

    def test_same_observation_different_truth(self):
        x, labels = exercise(.8, -1., .8)
        other, other_labels = exercise(-.55, .8, .8)
        self.assertAlmostEqual(x, .04)
        self.assertAlmostEqual(other, x)
        self.assertNotEqual(labels["epsilon"], other_labels["epsilon"])
        self.assertAlmostEqual(labels["v_prediction"], -1.28)
        self.assertAlmostEqual(read_prediction(x, -1.28, "epsilon", .8)[0], 1.01)

    def test_inverse_and_response_to_bias(self):
        for y, eps in [(.8, -1.), (-.55, .8), (0., 0.), (-1.5, -2.)]:
            for a in (.01, .2, .8, .99999):
                sigma = math.sqrt(1-a*a)
                x, labels = exercise(y, eps, a)
                self.assertAlmostEqual(x*x+labels["v_prediction"]**2, y*y+eps*eps)
                for kind, gain in zip(HEADS, (-sigma/a, 1, -sigma)):
                    yhat, ehat = read_prediction(x, labels[kind], kind, a)
                    self.assertAlmostEqual(yhat, y)
                    self.assertAlmostEqual(ehat, eps)
                    biased, _ = read_prediction(x, labels[kind]+.15, kind, a)
                    self.assertAlmostEqual(biased-y, gain*.15)

    def test_endpoints_are_explicit(self):
        for a in (0, 1):
            x, labels = exercise(.8, -1., a)
            self.assertEqual(read_prediction(x, labels["v_prediction"], "v_prediction", a), (.8, -1.))
        self.assertEqual(read_prediction(-1., -1., "epsilon", 0), (None, -1.))
        self.assertEqual(read_prediction(.8, .8, "sample", 1), (.8, None))

    def test_velocity_uses_generation_clock(self):
        dt = 1e-5
        for eps in (-1., 1.):
            for key in ("arc", "line"):
                self.assertEqual(trace(.8, eps, 0)[key][2], eps)
                self.assertEqual(trace(.8, eps, 1)[key][2], .8)
                for t in (.1, .5, .9):
                    speed = trace(.8, eps, t)[key][3]
                    delta = trace(.8, eps, t+dt)[key][2]-trace(.8, eps, t-dt)[key][2]
                    self.assertAlmostEqual(delta/(2*dt), speed, places=8)

    def test_car_uses_unclipped_errors(self):
        clean, noise = scene(17)
        self.assertEqual(len(noise), 48*30)
        for a, expected in ((.8, .2413524), (.2, 3.7239738)):
            _, _, matched, mistaken = car_row(clean, noise, a)
            self.assertLess(max(abs(y-v) for y,v in zip(clean,matched)), 1e-12)
            self.assertAlmostEqual(sum(abs(y-v) for y,v in zip(clean,mistaken))/len(clean),
                                   expected, places=6)
