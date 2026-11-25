"""实验 M：把视频潜码切成 token，用同一台条件 Transformer 实测 CFG。"""

import argparse
import hashlib
import json
from pathlib import Path

from latent_lab import LATENT_SHAPE, PairVAE, catalog, device_noise, normalization
from pixel_lab import byte_clips, stream, template_scores
import torch
from torch import nn


DIM, HEADS, TOKENS, COUNT = 48, 3, 18, 128
SCALES = (0, .5, 1, 2, 4, 8)
NULL, LEFT, RIGHT = range(3)


def patchify(z):
    # [组, 通道, 行, 列] → 每行两个宽 2 的块；每块先列出通道 0 的左右格。
    return z.reshape(-1,3,2,3,2,2).permute(0,1,3,4,2,5).reshape(-1,TOKENS,4)


def unpatchify(tokens):
    return tokens.reshape(-1,3,3,2,2,2).permute(0,1,4,2,3,5).reshape(-1,*LATENT_SHAPE)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DIM,elementwise_affine=False)
        self.qkv = nn.Linear(DIM,3*DIM)
        self.projection = nn.Linear(DIM,DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM,2*DIM),nn.GELU(),nn.Linear(2*DIM,DIM))
        self.modulate = nn.Sequential(nn.SiLU(),nn.Linear(DIM,6*DIM))
        nn.init.zeros_(self.modulate[-1].weight)
        nn.init.zeros_(self.modulate[-1].bias)
        group = torch.arange(TOKENS)//6
        self.register_buffer("other",group[:,None]!=group[None,:],persistent=False)

    def forward(self,h,condition,local):
        shift,scale,gate,shift_m,scale_m,gate_m = self.modulate(condition).unsqueeze(1).chunk(6,-1)
        modulated = self.norm(h)*(1+scale)+shift
        q,k,v = self.qkv(modulated).reshape(-1,TOKENS,3,HEADS,DIM//HEADS).permute(2,0,3,1,4)
        logits = (q @ k.transpose(-1,-2)) / (DIM//HEADS)**.5
        if local:
            logits = logits.masked_fill(self.other,float("-inf"))
        weights = logits.softmax(-1)
        message = (weights @ v).transpose(1,2).reshape_as(h)
        h = h + gate*self.projection(message)
        h = h + gate_m*self.mlp(self.norm(h)*(1+scale_m)+shift_m)
        return h,weights


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4,DIM)
        self.position = nn.Parameter(torch.randn(1,TOKENS,DIM)*.05)
        self.label = nn.Embedding(3,DIM)
        self.time = nn.Sequential(nn.Linear(3,DIM),nn.SiLU(),nn.Linear(DIM,DIM))
        self.blocks = nn.ModuleList([Block(),Block()])
        self.output = nn.Sequential(nn.LayerNorm(DIM),nn.Linear(DIM,4))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def embedding(self,state,t,labels):
        h = self.input(patchify(state))+self.position
        condition = self.time(torch.cat([t,t.square(),(torch.pi*t).sin()],dim=1))+self.label(labels)
        return h,condition

    def forward(self,state,t,labels,local=False,trace=False,residuals=None):
        h,condition = self.embedding(state,t,labels)
        maps = []
        for i,block in enumerate(self.blocks):
            h,weights = block(h,condition,local)
            if residuals is not None:
                h = h+residuals[i]
            if trace:
                maps.append(weights)
        velocity = unpatchify(self.output(h))
        return (velocity,maps) if trace else velocity


def direction_ids(clips):
    centers = template_scores(clips.cpu())["centers"]
    return torch.tensor([RIGHT if row[-1][0]>row[0][0] else LEFT for row in centers])


def train(net,codes,directions,seed,updates):
    rng = stream(seed+1)
    opt = torch.optim.AdamW(net.parameters(),lr=.00085,weight_decay=.01)
    device = codes.device
    for step in range(updates):
        if step in (3000,4400):
            for group in opt.param_groups:
                group["lr"] *= .5
        index = torch.randint(len(codes),(64,),generator=rng)
        target,labels = codes[index],directions[index].clone()
        labels[torch.rand(64,generator=rng).to(device)<.2] = NULL
        noise = device_noise(target.shape,rng,device)
        time = torch.rand(64,1,generator=rng).to(device)
        t = time.reshape(-1,1,1,1,1)
        pred = net((1-t)*noise+t*target,time,labels)
        loss = (pred-(target-noise)).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step+1 in (1,1000,3000,updates):
            print("训练 DiT",step+1,round(loss.item(),5),flush=True)


def guided(net,x,t,labels,w,local):
    if w==0:
        return net(x,t,torch.zeros_like(labels),local)
    if w==1:
        return net(x,t,labels,local)
    together = net(torch.cat([x,x]),torch.cat([t,t]),torch.cat([torch.zeros_like(labels),labels]),local)
    empty,conditional = together.chunk(2)
    return empty+w*(conditional-empty)


def integrate(net,initial,labels,w,steps,local=False):
    x = initial.clone()
    for k in range(steps):
        t = torch.full((len(x),1),k/steps,device=x.device)
        x = x+guided(net,x,t,labels,w,local)/steps
    return x


def compare(out,labels):
    stats = template_scores(out.cpu())
    paths = torch.tensor(stats["centers"])
    dx = paths[:,-1,0]-paths[:,0,0]
    ok = (dx*torch.where(labels.cpu()==LEFT,-1,1)>0)
    strict = ok & torch.tensor(stats["hits"])
    paired = ok.reshape(-1,2)
    both_strict = strict.reshape(-1,2).all(1)
    return {"direction":int(ok.sum()),"stationary":int((dx==0).sum()),"strict":int(strict.sum()),
            "both":int(paired.all(1).sum()),"both_strict":int(both_strict.sum()),"mse":stats["clip_mse"],
            "unique":len({tuple(map(tuple,p)) for p,hit in zip(stats["centers"],strict) if hit}),
            "both_bits":paired.all(1).tolist(),"first_dx":dx[:16].tolist(),"first_centers":paths[:16].tolist()}


@torch.no_grad()
def probe(net,state,foreign):
    t = torch.full((1,1),.5,device=state.device)
    rows = {}
    for name,label in [("empty",NULL),("left",LEFT),("right",RIGHT)]:
        ids = torch.tensor([label],device=state.device)
        velocity,maps = net(state,t,ids,trace=True)
        rows[name] = {"velocity":patchify(velocity)[0].tolist(),"maps":[layer[0].tolist() for layer in maps]}
    ids = torch.tensor([LEFT],device=state.device)
    replacement = foreign.clone()
    replacement[:,1] = state[:,1]
    effects,by_token = {},{}
    for key,local in [("full",False),("local",True)]:
        before = net(state,t,ids,local)
        after = net(replacement,t,ids,local)
        change = patchify(after-before)[0,6:12].abs()
        effects[key] = change.mean().item()
        by_token[key] = change.mean(dim=1).tolist()
    return {"tokens":patchify(state)[0].tolist(),"answers":rows,"effects":effects,"by_token":by_token}


@torch.no_grad()
def evaluate(net,vae,clips,hold,codes,mean,spread,seed,steps):
    device = mean.device
    initial = device_noise((COUNT,*LATENT_SHAPE),stream(seed+101),device).repeat_interleave(2,0)
    labels = torch.tensor([LEFT,RIGHT],device=device).repeat(COUNT)
    rows,examples = {},{}
    for key,scale,local in [(str(s),s,False) for s in SCALES]+[("local",1,True)]:
        z = integrate(net,initial,labels,scale,steps,local)
        out = vae.decode(mean+spread*z)
        rows[key]=compare(out,labels)
        rows[key]["pair_delta"]=(out[::2]-out[1::2]).abs().max().item()
        examples[key] = byte_clips(out[:16])
    test = (vae.encode(clips[hold])[0]-mean)/spread
    chosen = .5*(initial[0:1]+test[0:1])
    other = .5*(initial[2:3]+test[-1:])
    return {"stats":rows,"examples":examples,"probe":probe(net,chosen,other),"pairs":COUNT,
            "scales":[str(s) for s in SCALES],"steps":steps,"train_count":len(codes)}


def run(base,replay,output,seed,updates,eval_seed,steps,device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    fingerprint = hashlib.sha256(base.read_bytes()).hexdigest()
    if replay:
        saved = torch.load(replay,map_location="cpu",weights_only=True)
        if fingerprint!=saved["base_sha256"]:
            raise ValueError("请使用与支路一起训练的 VAE 权重文件")
        seed,updates = saved["seed"],saved["updates"]
    torch.manual_seed(seed)
    vae = PairVAE().to(device).eval().requires_grad_(False)
    vae.load_state_dict(torch.load(base,map_location="cpu",weights_only=True)["light"])
    clips,hold = catalog()
    directions = direction_ids(clips[~hold]).to(device)
    clips = clips.to(device)
    with torch.no_grad():
        codes = vae.encode(clips[~hold])[0]
        mean,spread = normalization(codes)
        standardized = (codes-mean)/spread
    net = TinyDiT().to(device)
    output.mkdir(parents=True,exist_ok=True)
    if replay:
        net.load_state_dict(saved["model"])
    else:
        train(net,standardized,directions,seed,updates)
        torch.save({"model":net.state_dict(),"base_sha256":fingerprint,"seed":seed,"updates":updates},output/"weights.pt")
    eval_seed = seed if eval_seed is None else eval_seed
    data = evaluate(net.eval(),vae,clips,hold,codes,mean,spread,eval_seed,steps)
    data.update(seed=seed,updates=updates,eval_seed=eval_seed,device=str(device),base_sha256=fingerprint,
                params=sum(p.numel() for p in net.parameters()))
    (output/"numbers.json").write_text(json.dumps(data,separators=(",",":")),encoding="utf-8")
    print({k:(v["direction"],v["strict"],v["both"],v["unique"]) for k,v in data["stats"].items()},flush=True)
    from dit_report import render
    render(data,output)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=Path(__file__).resolve().parents[1]/"latent-lab"/"weights.pt")
    parser.add_argument("--replay",type=Path)
    parser.add_argument("--seed",type=int,default=67)
    parser.add_argument("--updates",type=int,default=5200)
    parser.add_argument("--eval-seed",type=int)
    parser.add_argument("--sample-steps",type=int,default=24)
    parser.add_argument("--out",type=Path,default=Path("replay/dit-lab"))
    parser.add_argument("--device",choices=("cpu","cuda","auto"),default="auto")
    args=parser.parse_args()
    if args.sample_steps<1 or args.updates<0:
        parser.error("采样至少一步，训练次数不可为负")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    run(args.base,args.replay,args.out,args.seed,args.updates,args.eval_seed,args.sample_steps,torch.device(device))
