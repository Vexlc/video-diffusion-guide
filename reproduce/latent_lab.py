"""实验 L：学习视频潜表示，冻结解码器，在潜空间训练生成速度。需要 PyTorch。"""

import argparse
import base64
import copy
import json
import struct
from pathlib import Path

from pixel_lab import FRAMES, HEIGHT, WIDTH, byte_clips, paint, stream, template_scores
import torch
from torch import nn
from torch.nn import functional as F


CHANNELS, ROWS, COLS = 2, 3, 4
LATENT_SHAPE = (FRAMES//2, CHANNELS, ROWS, COLS)
WIDTH_Z = (FRAMES//2)*CHANNELS*ROWS*COLS
BETAS = {"light": .01, "strong": .2}


def catalog():
    triples = [(left if sign==1 else WIDTH-1-left, row, sign)
               for left in range(2,9) for row in range(2,10) for sign in (-1,1)]
    hold = torch.tensor([(left+row+2*(sign==1))%4==0
                         for left in range(2,9) for row in range(2,10) for sign in (-1,1)])
    return paint(*torch.tensor(triples).T), hold


class PairVAE(nn.Module):
    """每相邻两帧视作两通道，共用空间编解码器；中间无跳连。"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(2,24,3,padding=1), nn.SiLU(),
            nn.Conv2d(24,24,4,stride=2,padding=1), nn.SiLU(),
            nn.Conv2d(24,32,4,stride=2,padding=1), nn.SiLU(),
            nn.Conv2d(32,2*CHANNELS,1))
        self.decoder = nn.Sequential(nn.Conv2d(CHANNELS,32,3,padding=1), nn.SiLU(),
            nn.ConvTranspose2d(32,24,4,stride=2,padding=1), nn.SiLU(),
            nn.ConvTranspose2d(24,24,4,stride=2,padding=1), nn.SiLU(),
            nn.Conv2d(24,2,3,padding=1))

    def encode(self, clip):
        params = self.encoder(clip.reshape(-1,2,HEIGHT,WIDTH))
        mu, logvar = params.chunk(2,dim=1)
        return mu.reshape(-1,*LATENT_SHAPE), logvar.clamp(-12,8).reshape(-1,*LATENT_SHAPE)

    def logits(self,z):
        return self.decoder(z.reshape(-1,CHANNELS,ROWS,COLS)).reshape(-1,FRAMES,HEIGHT,WIDTH)

    def decode(self,z):
        return 2*self.logits(z).sigmoid()-1


def posterior(mu,logvar,eps):
    return mu+(logvar*.5).exp()*eps


def kl(mu,logvar):
    return .5*(mu.square()+logvar.exp()-1-logvar).mean()


class LatentFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(WIDTH_Z+3,192), nn.SiLU(),
                                 nn.Linear(192,192), nn.SiLU(), nn.Linear(192,WIDTH_Z))

    def forward(self,state,t):
        time = torch.cat([t,t.square(),(t*torch.pi).sin()],dim=1)
        return self.net(torch.cat([state.flatten(1),time],dim=1)).reshape_as(state)


def device_noise(shape,rng,device):
    return torch.randn(shape,generator=rng).to(device)


def train_vaes(models, clips, seed, steps):
    rng = stream(seed+1)
    optim = {name:torch.optim.Adam(net.parameters(),lr=.002) for name,net in models.items()}
    for step in range(steps):
        batch = clips[torch.randint(len(clips),(32,),generator=rng)]
        eps = device_noise((len(batch),*LATENT_SHAPE),rng,batch.device)
        for name,net in models.items():
            mu,logvar = net.encode(batch)
            logits = net.logits(posterior(mu,logvar,eps))
            bce = F.binary_cross_entropy_with_logits(logits,(batch+1)/2)
            loss = bce + BETAS[name]*kl(mu,logvar)
            optim[name].zero_grad(set_to_none=True)
            loss.backward()
            optim[name].step()
        if step+1 in (1,500,steps):
            print("训练 VAE",step+1,flush=True)


def normalization(codes):
    axes=(0,1,3,4)
    return codes.mean(axes,keepdim=True), codes.std(axes,unbiased=False,keepdim=True).clamp_min(1e-5)


def train_flow(net,codes,seed,steps):
    rng = stream(seed+2)
    opt = torch.optim.Adam(net.parameters(),lr=.001)
    for step in range(steps):
        target = codes[torch.randint(len(codes),(96,),generator=rng)]
        noise = device_noise(target.shape,rng,target.device)
        time = torch.rand(len(target),1,generator=rng).to(target.device)
        t = time.reshape(-1,1,1,1,1)
        velocity = net((1-t)*noise+t*target,time)
        loss = (velocity-(target-noise)).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step+1 in (1,500,steps):
            print("训练潜空间速度",step+1,flush=True)


def transport(net,noise,steps):
    state = noise.clone()
    for k in range(steps):
        time = torch.full((len(noise),1),k/steps,device=noise.device)
        state = state + net(state,time)/steps
    return state


def reconstruction(output,truth):
    scores = template_scores(output.cpu())
    reference = template_scores(truth.cpu())
    exact = [a==b for a,b in zip(scores["centers"],reference["centers"])]
    return {"mse":(output-truth).square().mean().item(),"exact":sum(exact)}


def export_decoder(model):
    layers = []
    for layer in model.decoder:
        if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):
            weights = layer.weight.detach().cpu().flatten().tolist()
            blob = struct.pack(f"<{len(weights)}f",*weights)
            layers.append({"kind":"up" if isinstance(layer,nn.ConvTranspose2d) else "conv",
                           "in":layer.in_channels,"out":layer.out_channels,"kernel":layer.kernel_size[0],
                           "weights":base64.b64encode(blob).decode("ascii"),"bias":layer.bias.detach().cpu().tolist()})
    return layers


@torch.no_grad()
def evaluate(models,flow,clips,hold,seed):
    truth = clips[hold]
    device = clips.device
    picks = torch.linspace(0,len(truth)-1,8).round().long()
    eps = device_noise((len(truth),*LATENT_SHAPE),stream(seed+11),device)
    extra = device_noise(eps.shape,stream(seed+12),device)
    examples,stats = {"truth":byte_clips(truth[picks])},{}
    for name,model in models.items():
        mu,logvar = model.encode(truth)
        mean_video = model.decode(mu)
        first = model.decode(posterior(mu,logvar,eps))
        second = model.decode(posterior(mu,logvar,extra))
        stats[name] = {"mean":reconstruction(mean_video,truth),"draw":reconstruction(first,truth),
                       "kl":kl(mu,logvar).item(),"avg_sigma":(logvar*.5).exp().mean().item(),
                       "draw_change":(first-second).abs().mean().item()}
        for key,video in [("mean",mean_video),("draw",first),("again",second)]:
            examples[name+"_"+key]=byte_clips(video[picks])
    chosen = models["light"]
    mu,logvar = chosen.encode(truth)
    donor = mu[picks.flip(0)]
    combined = mu[picks].clone()
    combined[:,1] = donor[:,1]
    hybrid = chosen.decode(combined)
    examples["donor"] = byte_clips(chosen.decode(donor))
    examples["hybrid"] = byte_clips(hybrid)
    original = chosen.decode(mu[picks])
    stats["splice"] = {"outer_delta":(hybrid[:,[0,1,4,5]]-original[:,[0,1,4,5]]).abs().max().item(),
                       "middle_delta":(hybrid[:,2:4]-original[:,2:4]).abs().mean().item()}
    modified = mu[picks].clone()
    modified[:,1,0,1,1] += 1.5
    changed = chosen.decode(modified)
    check = {"base":original[0,2:4].cpu().flatten().tolist(),"edit":changed[0,2:4].cpu().flatten().tolist()}
    mean,std = normalization(chosen.encode(clips[~hold])[0])
    initial = device_noise((256,*LATENT_SHAPE),stream(seed+13),device)
    stats["generation"] = {}
    legal = {tuple(map(tuple,track)):index for index,track in enumerate(template_scores(clips.cpu())["centers"])}
    keys = [("prior",initial),("start",mean+std*initial)]
    for steps in (1,4,24):
        keys.append((f"flow{steps}",mean+std*transport(flow,initial,steps)))
    for key,raw in keys:
        out = chosen.decode(raw)
        score = template_scores(out.cpu())
        stats["generation"][key] = {k:score[k] for k in ("route_hits","clip_mse","nearest_distinct")}
        matched = [legal[tuple(map(tuple,track))] for track,hit in zip(score["centers"],score["hits"]) if hit]
        stats["generation"][key]["distinct_exact"] = len(set(matched))
        stats["generation"][key]["held_exact"] = sum(bool(hold[index]) for index in matched)
        examples[key] = byte_clips(out[:8])
    return {"seed":seed,"betas":BETAS,"train_count":int((~hold).sum()),"hold_count":int(hold.sum()),
            "picks":picks.tolist(),"latent":mu[picks].cpu().tolist(),"first_sigma":(logvar[picks]*.5).exp().cpu().tolist(),
            "first_eps":eps[picks].cpu().tolist(),"channel_mean":mean.flatten().tolist(),
            "channel_std":std.flatten().tolist(),"stats":stats,"examples":examples,
            "decoder":export_decoder(chosen),"browser_check":check}


def run(output,seed,vae_steps,flow_steps,device,replay=None):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(replay,map_location="cpu",weights_only=True) if replay else None
    if replay:
        seed,vae_steps,flow_steps = checkpoint["settings"]
    torch.manual_seed(seed)
    base = PairVAE()
    models = {name:copy.deepcopy(base).to(device) for name in BETAS}
    net = LatentFlow().to(device)
    clips,hold = catalog()
    clips = clips.to(device)
    output.mkdir(parents=True,exist_ok=True)
    if replay:
        for name,model in models.items():
            model.load_state_dict(checkpoint[name])
        net.load_state_dict(checkpoint["flow"])
    else:
        train_vaes(models,clips[~hold],seed,vae_steps)
        for model in models.values():
            model.eval().requires_grad_(False)
        with torch.no_grad():
            codes = models["light"].encode(clips[~hold])[0]
            center,spread = normalization(codes)
            standardized = (codes-center)/spread
        train_flow(net,standardized,seed,flow_steps)
        checkpoint = {name:model.state_dict() for name,model in models.items()}
        checkpoint.update(flow=net.state_dict(),settings=[seed,vae_steps,flow_steps])
        torch.save(checkpoint,output/"weights.pt")
    payload = evaluate(models,net,clips,hold,seed)
    payload.update(vae_steps=vae_steps,flow_steps=flow_steps,device=str(device),
                   vae_params=sum(p.numel() for p in base.parameters()),flow_params=sum(p.numel() for p in net.parameters()))
    (output/"numbers.json").write_text(json.dumps(payload,separators=(",",":")),encoding="utf-8")
    print(payload["stats"],flush=True)
    from latent_report import render
    render(payload,output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("replay/latent-lab"))
    parser.add_argument("--seed",type=int,default=53)
    parser.add_argument("--vae-steps",type=int,default=2200)
    parser.add_argument("--flow-steps",type=int,default=3200)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--replay",type=Path)
    args = parser.parse_args()
    if min(args.vae_steps,args.flow_steps)<0:
        parser.error("训练步数不能为负")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    run(args.out,args.seed,args.vae_steps,args.flow_steps,torch.device(device),args.replay)
