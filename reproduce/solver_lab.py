"""在同一训练过的视频速度场内，比较中途停、欧拉粗步和显式中点。CPU 可运行。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from dit_lab import patchify
from flow_hint import ROOT, EVAL_SEED, LEFT, RIGHT, STEPS, brightest_centers, catalog, initial_states, load_trained, setup, solve
from pixel_lab import byte_clips, stream
from latent_lab import LATENT_SHAPE, device_noise

BUDGETS=(2,4,8,12,24)


def schedule(mode,budget):
    if budget not in BUDGETS:
        raise ValueError("查询预算须为 2、4、8、12 或 24")
    if mode=="stop":
        return STEPS,"euler",budget
    if mode=="euler":
        return budget,"euler",budget
    if mode=="midpoint":
        return budget//2,"midpoint",budget//2
    raise ValueError("走法应为 stop、euler 或 midpoint")


def strict_hits(decoded,goal):
    centers=brightest_centers(decoded.cpu())
    hits=(centers==goal.cpu()).all(-1)
    return centers,hits,hits.all(1)


def rmse(a,b):
    return float((a.double()-b.double()).square().mean().sqrt())


def sample_path(base,branch,start,labels,points,strength,decode,focus,budget,mode):
    """记录同一展示样片的真实查询，再解码本次实际抵达的位置。"""
    steps,method,count=schedule(mode,budget)
    def coordinate(x):
        return float(patchify(x[focus:focus+1])[0,8,0])
    path=[coordinate(start)]
    queries=[]
    speeds=[]
    def observe_query(t,state,v):
        queries.append([float(t[0,0]),coordinate(state),coordinate(v)])
    def record(k,x,v,following,t,hint):
        path.append(coordinate(following))
        speeds.append(coordinate(v))
    end=solve(base,branch,start,labels,points,strength,record,steps=steps,method=method,
              stop_after=count,on_query=observe_query)
    video=decode(end)
    selected=video[focus:focus+1]
    row={"mode":mode,"budget":budget,"steps":steps,"updates":count,"at":count/steps,
         "calls":len(queries),"path":path,"speeds":speeds,"queries":queries,
         "centers":brightest_centers(selected.cpu())[0].tolist(),"film":byte_clips(selected)[0]}
    return row,end,video


@torch.no_grad()
def custom(root,out,device,col,row,direction,strength,noise_seed,budget):
    from solver_report import film_chart
    setup(device,437)
    vae,base,branch,saved=load_trained(root,root/"flow-hint/branch.pt",device)
    initial=device_noise((1,*LATENT_SHAPE),stream(noise_seed),device)
    points=torch.tensor([[col,row]],device=device)
    labels=torch.tensor([RIGHT if direction=="right" else LEFT],device=device)
    mean,spread=(saved[key].to(device) for key in ("mean","spread"))
    def decode(x):
        return vae.decode(mean+spread*x)
    rows=[]
    for mode in ("stop","euler","midpoint"):
        item,_,_=sample_path(base,branch,initial,labels,points,strength,decode,0,budget,mode)
        rows.append(item)
    out.mkdir(parents=True,exist_ok=True)
    (out/"paths.svg").write_text(film_chart(rows),encoding="utf-8")
    settings=dict(point=[col,row],direction=direction,strength=strength,noise_seed=noise_seed,budget=budget)
    (out/"sample.json").write_text(json.dumps(dict(settings=settings,rows=rows),separators=(",",":")),encoding="utf-8")
    for item in rows:
        print(item["mode"],"查询",item["calls"],"进度",item["at"],"中心",item["centers"])


@torch.no_grad()
def run(root,out,device):
    setup(device,437)
    ckpt=root/"flow-hint/branch.pt"
    vae,base,branch,saved=load_trained(root,ckpt,device)
    mean,spread=saved["mean"].to(device),saved["spread"].to(device)
    clips,mask=catalog()
    truth=brightest_centers(clips[mask])
    target=truth.repeat(8,1,1)
    start=initial_states(8,device).repeat_interleave(len(truth),0)
    points=target[:,0].to(device)
    labels=torch.where(target[:,-1,0]>target[:,0,0],RIGHT,LEFT).to(device)
    focus=(truth==torch.tensor([[7+i,3] for i in range(6)])).all(-1).all(-1).nonzero().item()
    ref96=solve(base,branch,start,labels,points,1.,steps=96,method="midpoint")
    reference=solve(base,branch,start,labels,points,1.,steps=192,method="midpoint")
    def decode(state):
        return vae.decode(mean+spread*state)
    ref_centers,_,ref_good=strict_hits(decode(reference),target)
    coarse_centers,_,coarse_good=strict_hits(decode(ref96),target)
    data={"budgets":list(BUDGETS),"count":len(target),"routes":len(truth),"seeds":8,"seed":EVAL_SEED,
          "reference":{"steps":192,"check_steps":96,"rmse":rmse(reference,ref96),"strict":int(ref_good.sum()),
                       "check_strict":int(coarse_good.sum()),"changed_centers":int((coarse_centers!=ref_centers).any(-1).sum())},
          "hashes":{**saved["base_hashes"],"branch":hashlib.sha256(ckpt.read_bytes()).hexdigest()},"rows":[]}
    for budget in BUDGETS:
        strict_by_mode={}
        for mode in ("stop","euler","midpoint"):
            row,end,video=sample_path(base,branch,start,labels,points,1.,decode,focus,budget,mode)
            _,hits,strict=strict_hits(video,target)
            strict_by_mode[mode]=strict
            row.update(strict=int(strict.sum()),frame_hits=int(hits.sum()),
                       latent_rmse=rmse(end,reference) if row["at"]==1 else None)
            data["rows"].append(row)
        e,m=(strict_by_mode[key] for key in ("euler","midpoint"))
        data.setdefault("paired",{})[str(budget)]={"both":int((e&m).sum()),"euler_only":int((e&~m).sum()),
           "mid_only":int((~e&m).sum()),"neither":int((~(e|m)).sum())}
        print(budget,[(r["mode"],r["strict"],r["latent_rmse"]) for r in data["rows"] if r["budget"]==budget],flush=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/"solver.json").write_text(json.dumps(data,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    from solver_report import render
    render(data,out)
    print("参照加密复查",data["reference"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--out",type=Path,default=Path("replay/solver"))
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--sample",action="store_true",help="只复算自选起点的三条路径")
    parser.add_argument("--point",type=int,nargs=2,default=[7,3],metavar=("COL","ROW"))
    parser.add_argument("--direction",choices=("right","left"),default="right")
    parser.add_argument("--noise-seed",type=int,default=EVAL_SEED)
    parser.add_argument("--strength",type=float,default=1.)
    parser.add_argument("--budget",type=int,choices=BUDGETS,default=4)
    a=parser.parse_args()
    if not (0<=a.point[0]<16 and 0<=a.point[1]<12 and math.isfinite(a.strength)):
        parser.error("点应在 16×12 画布内，强度应为有限数")
    if a.sample:
        custom(a.root,a.out,torch.device(a.device),*a.point,a.direction,a.strength,a.noise_seed,a.budget)
    else:
        run(a.root,a.out,torch.device(a.device))
