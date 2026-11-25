import math
import unittest

import route_pair as pair

try:
    import torch
except ImportError:
    torch=None


class PairedRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models=pair.load_models(pair.ROOT/"pair-lab/weights.json")

    def test_oracle_advances_to_next_training_point(self):
        y,z=-1.,pair.start(1700)[1]
        for k in (0,12,23):
            t=k/pair.STEPS
            dt=1/pair.STEPS
            a,s=math.sin(t*math.pi/2),math.cos(t*math.pi/2)
            x=a*y+s*z
            v=a*z-s*y
            n=(t+dt)*math.pi/2
            self.assertAlmostEqual(pair.update("diffusion",[x],[v],t,dt)[0],math.sin(n)*y+math.cos(n)*z)
            x=(1-t)*z+t*y
            self.assertAlmostEqual(pair.update("flow",[x],[y-z],t,dt)[0],(1-t-dt)*z+(t+dt)*y)

    def test_only_enabled_branches_respond_to_requested_row(self):
        upper,lower=(pair.four(self.models,1700,row) for row in (3,7))
        for task in pair.TASKS:
            self.assertEqual(upper[task][0],lower[task][0])
            self.assertEqual(upper[task][1][0],lower[task][1][0])
            for case,row in ((upper,3),(lower,7)):
                track=pair.cells(case[task][1][-1])
                self.assertEqual([y for _,y in track],[row]*6)

    @unittest.skipIf(torch is None,"对照 PyTorch 训练模型需要可选依赖")
    def test_stdlib_inference_matches_training_module(self):
        from route_pair_train import Backbone, PositionBranch
        noise=pair.start(1701)
        for task,params in self.models.items():
            base=Backbone()
            branch=PositionBranch(base)
            for module,prefix in ((base,"base."),(branch,"branch.")):
                module.load_state_dict({k[len(prefix):]:torch.tensor(v) for k,v in params.items() if k.startswith(prefix)})
            for t,row,gain in ((0.,3,1),(.5,7,1),(23/24,7,0)):
                features=torch.tensor([noise+[t,math.sin(t*math.pi/2),math.cos(t*math.pi/2)]])
                with torch.no_grad():
                    hidden=base(features)+gain*branch(features,torch.tensor([[(row-5)/2]]))
                    expected=base.head(hidden).flatten().tolist()
                actual=pair.predict(params,noise,t,row,gain)
                self.assertLess(max(abs(a-b) for a,b in zip(actual,expected)),3e-6)
