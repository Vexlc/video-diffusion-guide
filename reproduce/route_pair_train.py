"""在同一套六帧轨迹上训练两个相同结构的基座，再分别冻结基座、训练位置支路。CPU / PyTorch。"""
import argparse
import base64
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn

INPUTS,HIDDEN=12,48
TASKS=("diffusion","flow")


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=nn.Linear(INPUTS+3,HIDDEN)
        self.second=nn.Linear(HIDDEN,HIDDEN)
        self.head=nn.Linear(HIDDEN,INPUTS)

    def forward(self,features):
        return self.second(self.first(features).tanh()).tanh()


class PositionBranch(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.first=copy.deepcopy(base.first)
        self.second=copy.deepcopy(base.second)
        self.hint=nn.Linear(1,HIDDEN,bias=False)
        self.exit=nn.Linear(HIDDEN,HIDDEN)
        nn.init.zeros_(self.exit.weight)
        nn.init.zeros_(self.exit.bias)

    def forward(self,features,condition):
        hidden=(self.first(features)+self.hint(condition)).tanh()
        return self.exit(self.second(hidden).tanh())


def batch(generator,size,task):
    direction=torch.randint(2,(size,1),generator=generator)*2.-1.
    lane=torch.randint(2,(size,1),generator=generator)*2.-1.
    y=torch.stack((direction*torch.arange(6)/5,lane.expand(-1,6)),-1).reshape(size,12)
    noise=torch.randn(y.shape,generator=generator)
    time=torch.rand((size,1),generator=generator)
    angle=time*math.pi/2
    a,s=angle.sin(),angle.cos()
    if task=="diffusion":
        x=a*y+s*noise
        target=a*noise-s*y
    else:
        x=(1-time)*noise+time*y
        target=y-noise
    return torch.cat((x,time,a,s),1),lane,target


def train(task,steps,seed):
    torch.manual_seed(seed)
    base=Backbone()
    branch=None
    summary={}
    for stage in ("base","branch"):
        generator=torch.Generator().manual_seed(seed+(1 if stage=="base" else 2))
        if stage=="branch":
            branch=PositionBranch(base)
            base.requires_grad_(False)
        active=base if stage=="base" else branch
        optimizer=torch.optim.Adam(active.parameters(),lr=.002)
        for i in range(steps):
            if i==3*steps//4:
                for group in optimizer.param_groups:group["lr"]=.00065
            features,hint,target=batch(generator,96,task)
            hidden=base(features)
            if stage=="branch":
                update=branch(features,hint)
                if i==0:summary["initial_residual_max"]=update.abs().max().item()
                hidden=hidden+update
            loss=(base.head(hidden)-target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        summary[stage+"_last_loss"]=loss.item()
    parameters={**{"base."+k:v for k,v in base.state_dict().items()},
                **{"branch."+k:v for k,v in branch.state_dict().items()}}
    state={k:{"shape":list(v.shape),"data":base64.b64encode(v.numpy().astype("<f4").tobytes()).decode("ascii")}
           for k,v in parameters.items()}
    return state,summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=3000,help="每个基座和每条支路各自的训练次数")
    parser.add_argument("--seed",type=int,default=617)
    parser.add_argument("--out",type=Path,default=Path("replay/route-pair"))
    args=parser.parse_args()
    if args.steps<1:parser.error("训练次数须大于 0")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model={"seed":args.seed,"steps":args.steps,"torch":str(torch.__version__),"models":{},"training":{}}
    for task in TASKS:
        model["models"][task],model["training"][task]=train(task,args.steps,args.seed)
        print(task,model["training"][task],flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"weights.json").write_text(json.dumps(model,separators=(",",":")),encoding="utf-8")
