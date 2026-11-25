"""逐像素核对随讲义交付的采样记录，并保护采样器已有行为。"""
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

from clock_report import Recording


class ClockReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(__file__).resolve().parents[1]/"pixel-lab"
        cls.recording = Recording(cls.folder/"clock-data.js")

    def test_canonical_capture_matches_sampler_and_weights(self):
        rec = self.recording
        self.assertEqual((rec.cases, rec.frames, rec.height, rec.width, rec.steps), (8, 6, 12, 16, 24))
        self.assertEqual(rec.weight_sha256, hashlib.sha256((self.folder/"weights.pt").read_bytes()).hexdigest())
        panels = json.loads((self.folder/"numbers.json").read_text())["examples"]["joint"]
        for case in range(rec.cases):
            for frame in range(rec.frames):
                self.assertEqual(rec.image(case, 0, frame), panels["start"][case][frame])
                self.assertEqual(rec.image(case, 24, frame), panels["23"][case][frame])

    def test_every_recorded_next_state_follows_ddim(self):
        rec = self.recording
        stride = rec.cases*rec.frames*rec.plane
        largest_error = 0
        for n, (a, sigma, b, t) in enumerate(rec.coefficients):
            for i in range(n*stride, (n+1)*stride):
                x, v = rec.states[i], rec.velocities[i]
                expected = b*(a*x-sigma*v) + t*(sigma*x+a*v)
                largest_error = max(largest_error, abs(expected-rec.states[i+stride]))
        self.assertLess(largest_error, 1e-6)

    def test_book_pixel_and_terminal_step(self):
        rec = self.recording
        p = rec.pixel(0, 12, 2, 5, 7)
        self.assertAlmostEqual(p["clean"], -.173202, places=6)
        self.assertAlmostEqual(p["recorded_next"], .559907, places=6)
        self.assertEqual(set(rec.pixel(7, 24, 5, 11, 15)), {"state"})
        with self.assertRaises(ValueError):
            rec.pixel(8, 12, 0, 0, 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "重新运行采样器需要 PyTorch")
    def test_observation_preserves_sampling(self):
        import torch
        from clock_report import ClockRecorder
        from pixel_lab import sample, STEPS

        def model(state, tick):
            return state*.125+tick[:, None, None, None]

        initial = torch.randn((3, 6, 12, 16), generator=torch.Generator().manual_seed(17))
        baseline, panels = sample(model, initial)
        observer = ClockRecorder(2)
        traced, traced_panels = sample(model, initial, observe=observer.observe)
        self.assertTrue(torch.equal(baseline, traced))
        self.assertTrue(all(torch.equal(panels[k], traced_panels[k]) for k in panels))
        self.assertEqual(len(observer.velocities), STEPS)
        self.assertEqual(len(observer.states), STEPS+1)
        self.assertEqual(observer.shape, (2, 6, 12, 16))


if __name__ == "__main__":
    unittest.main()
