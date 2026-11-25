import json
import tempfile
import unittest
from pathlib import Path

import reverse_paths
import workshop
import workshop_models as model


class WorkshopTests(unittest.TestCase):
    def test_zero_velocity_is_identity(self):
        starts = [-0.33, 1.27]
        self.assertEqual(model.advance(model.train_tables(17, 0), starts), starts)
        for value in (-5, -0.15, 0, 3.2, 7):
            self.assertAlmostEqual(sum(weight for cell, weight in model.neighbors(value, 0.55)), 1)

    def test_equal_keys_share_mass(self):
        distinct, duplicates = model.attention(False), model.attention(True)
        self.assertGreater(distinct[0][1], .83)
        self.assertEqual(duplicates[0][0], duplicates[0][1])
        for row in distinct+duplicates:
            self.assertAlmostEqual(sum(row), 1)

    def test_zero_is_frozen(self):
        frames = model.spatial_workshop(18, 0)["frames"]
        for frame in frames:
            self.assertEqual(frame["base"], frame["aligned"])
            self.assertEqual(frame["aligned"], frame["late"])

    def test_sampler_sources(self):
        fixed = [-0.2, 0.1, 0.7, -0.3, 0.4, 0.8, -1.1, 0.9]
        for kind in ("DDIM", "DDPM"):
            self.assertEqual(reverse_paths.sample(fixed, kind, 27), reverse_paths.sample(fixed, kind, 27))
        self.assertEqual(reverse_paths.sample(fixed, "DDIM", 27), reverse_paths.sample(fixed, "DDIM", 85))

    def test_workbook_readouts_and_output(self):
        with tempfile.TemporaryDirectory() as temp:
            result = workshop.run(17, 32768, 80, Path(temp))
            self.assertEqual(result, json.loads((Path(temp)/"metrics.json").read_text()))
            for name in ("index.html", "motion.svg", "spatial.svg", "reverse-counts.svg"):
                self.assertTrue((Path(temp)/name).is_file())
        self.assertEqual([stage["gap"] for stage in result["fm"]], [96, 27, 7])
        self.assertAlmostEqual(result["fm"][-1]["loss"], .814, places=3)
        self.assertEqual(result["spatial"]["n_changed"], 448)
        self.assertAlmostEqual(result["spatial"]["changed"]["late"], 1.394, places=3)
        self.assertEqual(result["reverse"]["DDIM_fixed"]["counts"], [0, 0, 0, 96])
        for key, record in result["reverse"].items():
            self.assertEqual(sum(record["counts"]), 96)
            self.assertLess(record["max_rms"], 1e-10)
            if key != "DDIM_fixed":
                self.assertTrue(all(record["counts"]))


if __name__ == "__main__":
    unittest.main()
