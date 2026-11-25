"""核对同一格潜码的教学坐标、生成记录与两次单变量对照。"""
import base64
import unittest

from reading_route import ROOT, STEPS, comparison_values, lesson, readings


class ReadingRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = lesson(ROOT)

    def test_shared_noise_and_different_condition(self):
        first, moved = self.cases["3"], self.cases["7"]
        self.assertEqual(first["x"][0], moved["x"][0])
        self.assertNotEqual(first["y"], moved["y"])
        for row,case in self.cases.items():
            self.assertEqual(len(base64.b64decode(case["film"])), (STEPS+1)*6*12*16)
            self.assertEqual(case["centers"], [[7+i,int(row)] for i in range(6)])

    def test_targets_and_measured_steps(self):
        for case in self.cases.values():
            start = readings(case,0)
            self.assertAlmostEqual(start["diffusion"],start["linear"])
            self.assertAlmostEqual(start["teacher_prediction"],start["speed"],places=4)
            for k in range(STEPS+1):
                p = readings(case,k)
                self.assertAlmostEqual(p["component_squared_error"],(case["guesses"][k]-(case["y"]-case["x"][0]))**2)
                self.assertAlmostEqual(p["a"]**2+p["sigma"]**2,1)
                self.assertAlmostEqual(p["linear"],p["z"]+p["t"]*p["velocity_label"])
                if k<STEPS:
                    self.assertAlmostEqual(p["generated"]+p["speed"]/STEPS,p["next"],places=6)
            self.assertAlmostEqual(p["diffusion"],p["y"])
            self.assertAlmostEqual(p["linear"],p["y"])
            self.assertIsNone(p["speed"])
            self.assertIsNone(p["next"])
            self.assertEqual(comparison_values(p)["sample-use"],"解码六帧")

    def test_printed_halfway_example(self):
        result = readings(self.cases["3"],12)
        for key,value in (("y",1.979),("z",-.390),("diffusion",1.124),("linear",.795),
                          ("velocity_label",2.369),("teacher_prediction",2.151),
                          ("generated",.680),("speed",2.239),("next",.773)):
            self.assertEqual(round(result[key],3),value)
        self.assertEqual(comparison_values(result)["exercise-use"],"平方误差 0.0476")
