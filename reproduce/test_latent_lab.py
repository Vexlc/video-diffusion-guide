import importlib.util
import unittest

if importlib.util.find_spec("torch"):
    import latent_lab as lab
    import torch
else:
    torch = None


@unittest.skipIf(torch is None,"可选实验 L 需要 PyTorch")
class LatentTests(unittest.TestCase):
    def test_catalog_and_shape(self):
        clean,hold = lab.catalog()
        self.assertEqual(clean.unique(dim=0).shape[0],112)
        self.assertEqual(int(hold.sum()),28)
        scores = lab.template_scores(clean[hold])
        self.assertEqual(scores["route_hits"],28)
        net = lab.PairVAE()
        mu,logvar = net.encode(clean[:2])
        self.assertEqual(tuple(mu.shape),(2,3,2,3,4))
        self.assertEqual(mu.shape,logvar.shape)
        self.assertEqual(net.decode(mu).shape,clean[:2].shape)

    def test_kl_and_reparameterization(self):
        mu = torch.tensor([1.],requires_grad=True)
        logvar = torch.tensor([0.],requires_grad=True)
        self.assertAlmostEqual(lab.kl(mu,logvar).item(),.5)
        z = lab.posterior(mu,logvar,torch.tensor([-2.]))
        z.backward()
        self.assertEqual((z.item(),mu.grad.item(),logvar.grad.item()),(-1.,1.,-1.))

    def test_pair_intervention_is_local(self):
        torch.manual_seed(13)
        net = lab.PairVAE()
        clean,_ = lab.catalog()
        mu,_ = net.encode(clean[[0,111]])
        code = mu.clone()
        code[0,1] = mu[1,1]
        base,altered = net.decode(mu),net.decode(code)
        self.assertTrue(torch.equal(base[0,[0,1,4,5]],altered[0,[0,1,4,5]]))
        self.assertTrue(torch.equal(altered[0,2:4],base[1,2:4]))
        self.assertFalse(torch.equal(base[0,2:4],altered[0,2:4]))

    def test_channel_units_and_ode(self):
        codes = torch.randn(9,*lab.LATENT_SHAPE,generator=lab.stream(29))
        center,spread = lab.normalization(codes)
        normalized = (codes-center)/spread
        self.assertEqual(tuple(center.shape),(1,1,2,1,1))
        self.assertTrue(torch.allclose(normalized.std((0,1,3,4),unbiased=False),torch.ones(2)))
        for steps in (1,4,24):
            end = lab.transport(lambda x,t:torch.ones_like(x),normalized,steps)
            self.assertTrue(torch.allclose(end,normalized+1,atol=5e-6))
