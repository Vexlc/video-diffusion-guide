import importlib.util
import unittest

if importlib.util.find_spec("torch"):
    import torch
    import dit_lab as lab
else:
    torch = None


@unittest.skipIf(torch is None, "实验 M 的网络测试需要 PyTorch")
class DiTTests(unittest.TestCase):
    def test_patch_addresses_and_inverse(self):
        volume = torch.arange(144).reshape(2, *lab.LATENT_SHAPE)
        tokens = lab.patchify(volume)
        self.assertEqual(tuple(tokens.shape), (2, 18, 4))
        for group in range(3):
            for row in range(3):
                for side in range(2):
                    expected = volume[:, group, :, row, 2*side:2*side+2].reshape(2, 4)
                    self.assertTrue(torch.equal(tokens[:, 6*group+2*row+side], expected))
        self.assertTrue(torch.equal(lab.unpatchify(tokens), volume))

    def test_zero_gates_and_group_mask(self):
        block = lab.Block()
        h = torch.randn(2, lab.TOKENS, lab.DIM)
        condition = torch.randn(2, lab.DIM)
        for local in (False, True):
            output, weights = block(h, condition, local)
            self.assertTrue(torch.equal(output, h))
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
            outside = weights[..., block.other]
            self.assertTrue((outside == 0).all() if local else (outside > 0).all())

    def test_guidance_and_cross_group_intervention(self):
        torch.manual_seed(718)
        net = lab.TinyDiT()
        for block in net.blocks:
            torch.nn.init.normal_(block.modulate[-1].weight, std=.15)
        torch.nn.init.normal_(net.output[-1].weight, std=.15)
        state = torch.randn(2, *lab.LATENT_SHAPE)
        labels = torch.tensor([lab.LEFT, lab.RIGHT])
        t = torch.full((2, 1), .5)
        plain = net(state, t, torch.zeros_like(labels))
        conditional = net(state, t, labels)
        self.assertGreater((conditional-plain).abs().max().item(), .01)
        for w in (0, .5, 1, 2):
            torch.testing.assert_close(lab.guided(net,state,t,labels,w,False),
                                       plain+w*(conditional-plain), atol=2e-6, rtol=2e-6)
        result = lab.probe(net, state[:1], state[1:])
        self.assertGreater(result["effects"]["full"], .001)
        self.assertEqual(result["by_token"]["local"], [0.]*6)

    def test_legal_direction_and_middle_frame(self):
        clips, _ = lab.catalog()
        pair = clips[:2].clone()
        labels = torch.tensor([lab.LEFT, lab.RIGHT])
        self.assertTrue(torch.equal(lab.direction_ids(pair), labels))
        self.assertEqual(lab.compare(pair, labels)["both_strict"], 1)
        self.assertEqual(lab.compare(pair, labels.flip(0))["direction"], 0)
        pair[0, 2] = pair[0, 0]
        damaged = lab.compare(pair, labels)
        self.assertEqual((damaged["direction"], damaged["strict"], damaged["both_strict"]), (2, 1, 0))
