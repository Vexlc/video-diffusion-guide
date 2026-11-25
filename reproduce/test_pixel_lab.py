import importlib.util
import unittest

if importlib.util.find_spec("torch"):
    import torch
    import pixel_lab as pixel
else:
    torch = None


@unittest.skipIf(torch is None, "可选实验 I 需要 PyTorch")
class PixelLabTests(unittest.TestCase):
    def test_inverse(self):
        clean = pixel.make_clips(4, pixel.stream(8))
        eps = torch.randn(clean.shape, generator=pixel.stream(9))
        tick = torch.tensor([0, .25, .65, 1.0])
        state, target = pixel.exercise(clean, eps, tick)
        signal, sigma = pixel.amplitudes(tick)
        self.assertTrue(torch.allclose(signal*state-sigma*target, clean, atol=1e-6))
        self.assertTrue(torch.allclose(sigma*state+signal*target, eps, atol=1e-6))

    def test_scoring_distinguishes_shuffle(self):
        clean = pixel.make_clips(12, pixel.stream(10))
        mixed = clean[:, [0, 3, 2, 1, 4, 5]]
        self.assertEqual(pixel.template_scores(clean)["route_hits"], 12)
        scores = pixel.template_scores(mixed)
        self.assertEqual(scores["route_hits"], 0)
        self.assertGreater(scores["clip_mse"], 0)

    def test_edge_does_not_wrap(self):
        feature = torch.arange(6).float()[:, None, None, None].expand(6, 4, 1, 1)
        result = pixel.mix_neighbors(feature, 6, True)
        self.assertEqual(result[0, :, 0, 0].tolist(), [0, 1, 0, 0])
        self.assertEqual(result[-1, :, 0, 0].tolist(), [4, 5, 5, 5])

    def test_only_joint_uses_neighbors(self):
        torch.manual_seed(19)
        clean = pixel.make_clips(2, pixel.stream(19))
        changed = clean.roll(1, dims=0)
        changed[:, 2] = clean[:, 2]
        tick = torch.full((2,), .65)
        with torch.no_grad():
            model = pixel.TinyVideo(False)
            self.assertEqual(sum(param.numel() for param in model.parameters()), 21745)
            self.assertTrue(torch.equal(model(clean, tick)[:, 2], model(changed, tick)[:, 2]))
            model.temporal = True
            self.assertFalse(torch.equal(model(clean, tick)[:, 2], model(changed, tick)[:, 2]))


if __name__ == "__main__":
    unittest.main()
