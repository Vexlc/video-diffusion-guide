import json
import math
import tempfile
import unittest
from pathlib import Path

import reproduce


class ReplayTests(unittest.TestCase):
    def test_book_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = reproduce.run(17, 32, 80, output)
            self.assertEqual(json.loads((output / "metrics.json").read_text())["seed"], 17)
            self.assertEqual(len(list(output.glob("*.svg"))), 4)
            self.assertTrue((output / "index.html").is_file())
        self.assertAlmostEqual(metrics["A_pixel_error_k90"], 0.895, places=3)
        for key, expected in [
                ("B_mean_distance_to_reference", [2.178, 0.037]),
                ("C_mean_absolute_second_difference", [1.910, 0]),
                ("D_position_MAE", [0.650, 0.285, 0.463])]:
            for actual, target in zip(metrics[key], expected):
                self.assertAlmostEqual(actual, target, places=3)

    def test_one_step_collapses(self):
        self.assertTrue(all(math.hypot(*point) < 1e-12 for point in reproduce.integrate(23, 1, 1)))

    def test_joint_constant_speed(self):
        for seed in range(10):
            path = reproduce.sample_video(seed, True)
            self.assertTrue(all(abs(after - 2*now + before) < 1e-8
                                for before, now, after in zip(path, path[1:], path[2:])))

    def test_zero_branch_preserves_base(self):
        weights = reproduce.train_control(0)
        self.assertEqual(weights, (0, 0))
        truth, guide, base, aligned, delayed = reproduce.use_control(17, weights)
        self.assertEqual(base, aligned)
        self.assertEqual(base, delayed)


if __name__ == "__main__":
    unittest.main()
