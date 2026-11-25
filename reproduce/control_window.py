"""对同一个视频流匹配网络，只改变首帧位置支路在哪些生成步启用。需要 PyTorch。"""
import argparse
import hashlib
from pathlib import Path

import torch
from dit_lab import patchify
from flow_hint import (EVAL_SEED, LEFT, RIGHT, ROOT, STEPS, brightest_centers, catalog,
                       initial_states, load_trained, setup, solve)
from latent_lab import LATENT_SHAPE, device_noise
from pixel_lab import byte_clips, stream


def windows(width):
    if not 0<=width<=STEPS:
        raise ValueError("窗口宽度须在 0 到 24 之间")
    return {"off":(0,0), "early":(0,width), "late":(STEPS-width,STEPS), "full":(0,STEPS)}


@torch.no_grad()
def compare(weights, start, targets, labels, width, strength, focus):
    """四排各自做完 24 步。每次独立问速度，保持同题的起点、方向与权重一致。"""
    vae,base,branch,saved=weights
    runs={}
    for mode,window in windows(width).items():
        path=[patchify(start[focus:focus+1])[0,8,0].item()]
        speeds=[]
        def observe(k,state,v,following,t,hint):
            speeds.append(patchify(v[focus:focus+1])[0,8,0].item())
            path.append(patchify(following[focus:focus+1])[0,8,0].item())
        end=solve(base,branch,start,labels,targets[:,0],strength,observe,control_window=window)
        decoded=vae.decode(saved["mean"]+saved["spread"]*end)
        runs[mode]={"window":list(window), "centers":brightest_centers(decoded).tolist(),
                    "path":path,"speeds":speeds,"film":byte_clips(decoded[focus:focus+1])[0]}
    return runs


def run(root, out, *, width=12, strength=1., sample=False, col=7, row=3, direction="right", seed=EVAL_SEED):
    cpu=torch.device("cpu")
    setup(cpu,437)
    checkpoint=root/"flow-hint/branch.pt"
    weights=load_trained(root,checkpoint,cpu)
    if sample:
        start=device_noise((1,*LATENT_SHAPE),stream(seed),cpu)
        target=torch.tensor([[[col+i*(1 if direction=="right" else -1),row] for i in range(6)]])
        seeds=[seed]
        focus=0
    else:
        clips,hold=catalog()
        routes=brightest_centers(clips[hold])
        target=routes.repeat(8,1,1)
        start=initial_states(8,cpu).repeat_interleave(len(routes),0)
        seeds=[EVAL_SEED+i for i in range(8) for _ in routes]
        example=torch.tensor([[7+i,3] for i in range(6)])
        focus=(routes==example).all(-1).all(-1).nonzero().item()
    labels=torch.where(target[:,-1,0]>target[:,0,0],RIGHT,LEFT)
    runs=compare(weights,start,target,labels,width,strength,focus)
    record={"width":width,"steps":STEPS,"strength":strength,"seeds":seeds,"targets":target.tolist(),
            "focus":focus,"hashes":{**weights[3]["base_hashes"],"branch":hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
            "runs":runs}
    from window_report import render
    render(record,out)
    for mode,r in runs.items():
        correct=(torch.tensor(r["centers"])==target).all(-1)
        print(mode,"首帧",int(correct[:,0].sum()),"六帧全中",int(correct.all(-1).sum()),"/",len(target),
              "样片中心",r["centers"][focus])


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root",type=Path,default=ROOT)
    cli.add_argument("--out",type=Path,default=Path("replay/control-window"))
    cli.add_argument("--width",type=int,choices=range(25),default=12,metavar="0..24")
    cli.add_argument("--strength",type=float,choices=(0.,.5,1.),default=1.)
    cli.add_argument("--sample",action="store_true",help="只生成指定的一题；默认跑八份噪声与28条留出路线")
    cli.add_argument("--point",type=int,nargs=2,default=(7,3),metavar=("COL","ROW"))
    cli.add_argument("--direction",choices=("left","right"),default="right")
    cli.add_argument("--noise-seed",type=int,default=EVAL_SEED)
    args=cli.parse_args()
    col,row=args.point
    if not 0<=col<16 or not 0<=row<12:
        cli.error("点须位于宽16、高12的画框内")
    run(args.root,args.out,width=args.width,strength=args.strength,sample=args.sample,
        col=col,row=row,direction=args.direction,seed=args.noise_seed)
