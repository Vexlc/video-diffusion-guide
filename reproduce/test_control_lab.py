import importlib.util
import unittest

if importlib.util.find_spec("torch"):
    import control_lab as lab
    import torch
else:
    torch = None


@unittest.skipIf(torch is None, "可选实验 J 需要 PyTorch")
class ControlTests(unittest.TestCase):
    def test_pairs_keep_noise_and_start_only(self):
        truth, noise = lab.paired_cases(31, 5)
        dots = lab.center_dots(truth)
        self.assertTrue(torch.equal(noise[::2], noise[1::2]))
        self.assertTrue(torch.equal(dots[:, 0][::2], dots[:, 0][1::2]))
        self.assertTrue(torch.all(dots.sum((-1, -2))==1))
        for mode, count in enumerate([1, 2, 6]):
            hints = lab.visible(dots, torch.full((10,), mode))
            self.assertTrue(torch.all(hints.sum((1, 2, 3))==count))
            self.assertEqual(torch.equal(hints[::2], hints[1::2]), mode==0)

    def test_wrong_route_with_perfect_control(self):
        truth, _ = lab.paired_cases(7, 4)
        other = truth.reshape(4, 2, 6, 12, 16).flip(1).reshape_as(truth)
        score = lab.measurements(other, truth, other)
        self.assertEqual((score["exact"], score["opposite"], score["middle_l1"]), (0, 8, 5))

    def test_zero_gates_then_backprop_into_copy(self):
        torch.manual_seed(8)
        base = lab.TinyVideo(True)
        branch = lab.Branch(base)
        base.requires_grad_(False)
        truth, noise = lab.paired_cases(8, 2)
        guide = lab.center_dots(truth)
        tick = torch.full((4,), .5)
        before = base(noise, tick)
        self.assertTrue(torch.equal(before, lab.controlled(base, branch, noise, tick, guide)))
        grad, _ = lab.teach(base, branch, 8, 2, torch.device("cpu"))
        self.assertEqual(grad[0]["hint"], 0)
        self.assertEqual(grad[0]["enter"], 0)
        self.assertGreater(min(grad[0]["coarse"], grad[0]["fine"]), 0)
        self.assertGreater(min(grad[1]["hint"], grad[1]["enter"]), 0)
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertTrue(torch.equal(before, lab.controlled(base, branch, noise, tick, guide, (0, 0))))
