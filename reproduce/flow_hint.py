"""实验 P：冻结实验 M 的方向 DiT，训练零出口支路，根据首帧一个点控制潜空间流。"""
import argparse
import base64
import math
import copy
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

from dit_lab import DIM, LEFT, RIGHT, TinyDiT, patchify
from hint_probe import tensor_record
from latent_lab import LATENT_SHAPE, PairVAE, catalog, device_noise, normalization
from pixel_lab import FRAMES, HEIGHT, WIDTH, brightest_centers, byte_clips, stream


ROOT=Path(__file__).resolve().parents[1]
STEPS=24
SCALES=(0.,.5,1.)
SEED=437
EVAL_SEED=9317


def hint_map(points):
    """首帧点 (列, 行) 变为一个 12×16 的高斯亮点；后续帧的位置不作为输入。"""
    cols=torch.arange(WIDTH,device=points.device)
    rows=torch.arange(HEIGHT,device=points.device)
    dx=cols[None,None,:]-points[:,0,None,None]
    dy=rows[None,:,None]-points[:,1,None,None]
    return (-(dx.square()+dy.square())/8.).exp().unsqueeze(1)


class PointBranch(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.hint=nn.Linear(32,DIM)
        self.blocks=copy.deepcopy(base.blocks).requires_grad_(True)
        self.exits=nn.ModuleList([nn.Linear(DIM,DIM) for _ in self.blocks])
        for layer in self.exits:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self,h,condition,image):
        # 每块覆盖 4 行×8 列，按行序得到 6 块；同一张首帧提示送给三个时间组。
        patches=F.unfold(image,kernel_size=(4,8),stride=(4,8)).transpose(1,2)
        h=h+self.hint(patches).repeat(1,3,1)
        residuals=[]
        for block,exit_layer in zip(self.blocks,self.exits):
            h,_=block(h,condition,False)
            residuals.append(exit_layer(h))
        return residuals


def velocity(base,branch,x,time,direction,hint,strength):
    """当前状态、进度和方向保持不变，两个零出口的修正分别加到基座两个块之后。"""
    if strength==0:
        return base(x,time,direction)
    h,condition=base.embedding(x,time,direction)
    residuals=branch(h,condition,hint)
    return base(x,time,direction,residuals=[strength*r for r in residuals])


def train(base,branch,codes,labels,points,seed,updates):
    rng=stream(seed+1)
    opt=torch.optim.AdamW(branch.parameters(),lr=.00085,weight_decay=.002)
    device=codes.device
    hints=hint_map(points)
    history=[]
    for k in range(updates):
        if k in (2000,3000):
            for group in opt.param_groups:
                group["lr"]*=.5
        ids=torch.randint(len(codes),(48,),generator=rng)
        y=codes[ids]
        z=device_noise(y.shape,rng,device)
        t=torch.rand(len(y),1,generator=rng).to(device)
        progress=t.reshape(-1,1,1,1,1)
        pred=velocity(base,branch,(1-progress)*z+progress*y,t,labels[ids],hints[ids],1)
        loss=(pred-(y-z)).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if k+1 in (1,500,1000,2000,3000,updates):
            history.append([k+1,float(loss)])
            print("支路训练",k+1,round(float(loss),5),flush=True)
    return history


def models(root,device):
    vae_file=root/"latent-lab/weights.pt"
    dit_file=root/"dit-lab/weights.pt"
    hashes={k:hashlib.sha256(f.read_bytes()).hexdigest() for k,f in (("vae",vae_file),("dit",dit_file))}
    saved=torch.load(dit_file,map_location="cpu",weights_only=True)
    if saved["base_sha256"]!=hashes["vae"]:
        raise ValueError("方向 DiT 与 VAE 权重不配套")
    vae=PairVAE().to(device)
    vae.load_state_dict(torch.load(vae_file,map_location="cpu",weights_only=True)["light"])
    base=TinyDiT().to(device)
    base.load_state_dict(saved["model"])
    return vae.eval().requires_grad_(False),base.eval().requires_grad_(False),hashes


def dataset(vae,device):
    clips,hold=catalog()
    centers=brightest_centers(clips)
    labels=torch.where(centers[:,-1,0]>centers[:,0,0],RIGHT,LEFT)
    with torch.no_grad():
        encoded=vae.encode(clips[~hold].to(device))[0]
        mean,spread=normalization(encoded)
    return (encoded-mean)/spread,mean,spread,centers.to(device),labels.to(device),hold.to(device)


@torch.no_grad()
def solve(base,branch,start,labels,points,strength,observe=None,*,steps=STEPS,method="euler",stop_after=None,on_query=None,control_window=None):
    """在等距生成进度上求解。stop_after 限制实际更新次数，保留原网格步长。"""
    count=steps if stop_after is None else stop_after
    if steps<1 or not 0<=count<=steps or method not in ("euler","midpoint"):
        raise ValueError("步数、停止位置或积分方法无效")
    begin,end=(0,steps) if control_window is None else control_window
    if not 0<=begin<=end<=steps or not isinstance(begin,int) or not isinstance(end,int):
        raise ValueError("控制窗口使用从 0 数的整数步，须满足 0 ≤ 起点 ≤ 终点 ≤ 总步数")
    state=start.clone()
    hint=hint_map(points)
    for k in range(count):
        t=torch.full((len(start),1),k/steps,device=start.device)
        scale=strength if begin<=k<end else 0.
        v=velocity(base,branch,state,t,labels,hint,scale)
        if on_query is not None:
            on_query(t,state,v)
        if method=="midpoint":
            trial=state+v/(2*steps)
            at_half=torch.full_like(t,(k+.5)/steps)
            v=velocity(base,branch,trial,at_half,labels,hint,scale)
            if on_query is not None:
                on_query(at_half,trial,v)
        following=state+v/steps
        if observe is not None:
            observe(k,state,v,following,t,hint)
        state=following
    return state


def score(decoded,truth):
    measured=brightest_centers(decoded.cpu())
    truth=truth.cpu()
    equal=(measured==truth).all(-1)
    direction=(measured[:,-1,0]-measured[:,0,0])*(truth[:,-1,0]-truth[:,0,0])>0
    return {"first":int(equal[:,0].sum()),"direction":int(direction.sum()),
            "strict":int(equal.all(1).sum()),"frame_hits":int(equal.sum()),
            "n":len(truth),"centers":measured.tolist()}


def setup(device,seed):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(seed)


def save_training(root,out,device,updates,seed):
    setup(device,seed)
    vae,base,hashes=models(root,device)
    codes,mean,spread,tracks,labels,hold=dataset(vae,device)
    branch=PointBranch(base).to(device)
    history=train(base,branch,codes,labels[~hold],tracks[~hold,0],seed,updates)
    out.mkdir(parents=True,exist_ok=True)
    torch.save({"model":branch.state_dict(),"base_hashes":hashes,"updates":updates,
                "seed":seed,"history":history,"mean":mean.cpu(),"spread":spread.cpu()},out/"branch.pt")


def load_trained(root,checkpoint,device):
    vae,base,hashes=models(root,device)
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if saved["base_hashes"]!=hashes:
        raise ValueError("请使用训练时配套的方向 DiT 与 VAE")
    branch=PointBranch(base).to(device)
    branch.load_state_dict(saved["model"])
    return vae,base,branch.eval().requires_grad_(False),saved


def initial_states(count,device):
    # 每份起点单独设种子，单题运行与整批比较可取到完全相同的噪声。
    return torch.stack([device_noise(LATENT_SHAPE,stream(EVAL_SEED+i),device) for i in range(count)])


@torch.no_grad()
def evaluation(vae,base,branch,mean,spread,device):
    clips,hold=catalog()
    truth=brightest_centers(clips[hold])
    labels=torch.where(truth[:,-1,0]>truth[:,0,0],RIGHT,LEFT).to(device)
    starts=initial_states(8,device)
    states=starts.repeat_interleave(len(truth),0)
    expected=truth.repeat(len(starts),1,1)
    points=expected[:,0].to(device)
    scores={}
    first_error=None
    for scale in SCALES:
        last=solve(base,branch,states,labels.repeat(len(starts)),points,scale)
        decoded=vae.decode(mean+spread*last)
        result=score(decoded,expected)
        centers=result.pop("centers")
        if scale==1.:
            for i,(measured,reference) in enumerate(zip(centers,expected.tolist())):
                if measured!=reference:
                    first_error={"seed":EVAL_SEED+i//len(truth),"order":i%len(truth)+1,
                                 "reference":reference,"measured":measured,"film":byte_clips(decoded[i:i+1])[0]}
                    break
        scores[str(scale)]=result
    return scores,first_error


@torch.no_grad()
def record(vae,base,branch,mean,spread,device):
    points=torch.tensor([[7,3],[7,3],[7,7],[7,7]],device=device).repeat(3,1)
    labels=torch.tensor([LEFT,RIGHT,LEFT,RIGHT],device=device).repeat(3)
    initial=initial_states(3,device).repeat_interleave(4,0)
    records={}
    for strength in SCALES:
        states=[patchify(initial)]
        speeds=[];free=[];features=[]
        def observe(k,state,v,following,t,hint):
            states.append(patchify(following))
            speeds.append(patchify(v))
            free.append(patchify(base(state,t,labels)))
            h,ctx=base.embedding(state,t,labels)
            residuals=branch(h,ctx,hint)
            layer_values=[]
            for block,r in zip(base.blocks,residuals):
                before,_=block(h,ctx,False)
                h=before+strength*r
                layer_values.append(torch.stack((before[:,:,0],r[:,:,0],h[:,:,0]),dim=-1))
            features.append(torch.stack(layer_values,1))
            torch.testing.assert_close(base.output(h),patchify(v),rtol=2e-5,atol=3e-6)
        final=solve(base,branch,initial,labels,points,strength,observe)
        # 帧图均由截点的真实状态解码。只在保存显示亮度时量化为 uint8。
        from dit_lab import unpatchify
        trace=torch.stack(states,1)
        all_frames=vae.decode(mean+spread*unpatchify(trace.flatten(0,1)))
        pixels=((all_frames.clamp(-1,1)+1)*127.5).round().byte().cpu().numpy().tobytes()
        centers=brightest_centers(vae.decode(mean+spread*final).cpu()).tolist()
        records[str(strength)]={"x":tensor_record(trace),"v":tensor_record(torch.stack(speeds,1)),
            "free":tensor_record(torch.stack(free,1)),"features":tensor_record(torch.stack(features,1)),
            "pixels":base64.b64encode(pixels).decode("ascii"),"centers":centers}
    return records


def evaluate_run(root,checkpoint,out,device):
    setup(device,SEED)
    vae,base,branch,saved=load_trained(root,checkpoint,device)
    mean,spread=saved["mean"].to(device),saved["spread"].to(device)
    scores,first_error=evaluation(vae,base,branch,mean,spread,device)
    clips,hold=catalog()
    catalog_tracks=brightest_centers(clips)
    examples=[]
    for row in (3,7):
        for label,dx in (("向左",-1),("向右",1)):
            target=[[7+f*dx,row] for f in range(FRAMES)]
            indices=(catalog_tracks==torch.tensor(target)).all(2).all(1).nonzero().flatten()
            assert len(indices)==1
            examples.append({"point":[7,row],"label":label,"target":target,"held":bool(hold[indices[0]])})
    data={"seed":saved["seed"],"updates":saved["updates"],"history":saved["history"],"scores":scores,"first_error":first_error,
          "steps":STEPS,"eval_seed":EVAL_SEED,"train_count":int((~hold).sum()),"hold_count":int(hold.sum()),
          "scales":list(SCALES),"examples":examples,"base_hashes":saved["base_hashes"],
          "branch_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          "branch_params":sum(p.numel() for p in branch.parameters()),"records":record(vae,base,branch,mean,spread,device)}
    out.mkdir(parents=True,exist_ok=True)
    (out/"numbers.json").write_text(json.dumps(data,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    print(json.dumps(scores,ensure_ascii=False),flush=True)
    from flow_hint_report import render
    render(data,out)
    from first_film_export import export
    export(vae,base,branch,saved,checkpoint,out)


@torch.no_grad()
def custom_sample(root,checkpoint,out,device,point,direction,strength,noise_seed):
    from pixel_report import contact
    setup(device,SEED)
    vae,base,branch,saved=load_trained(root,checkpoint,device)
    initial=device_noise((1,*LATENT_SHAPE),stream(noise_seed),device)
    ids=torch.tensor([LEFT if direction=="left" else RIGHT],device=device)
    last=solve(base,branch,initial,ids,torch.tensor([point],device=device),strength)
    video=vae.decode(saved["mean"].to(device)+saved["spread"].to(device)*last)
    centers=brightest_centers(video.cpu())[0].tolist()
    title=f"首帧点 ({point[0]}, {point[1]})，{'向左' if direction=='left' else '向右'}，控制强度 {strength:g}"
    out.mkdir(parents=True,exist_ok=True)
    (out/"sample.svg").write_text(contact(title,[("24 步成片",byte_clips(video)[0])]),encoding="utf-8")
    print(json.dumps({"point":point,"direction":direction,"strength":strength,"noise_seed":noise_seed,
                      "measured_centers":centers},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--out",type=Path,default=Path("replay/flow-hint"))
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--updates",type=int,default=3600)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--replay",type=Path,help="对随包支路重跑推理与评测")
    p.add_argument("--point",nargs=2,type=int,metavar=("COL","ROW"),help="只生成自己指定首帧点的一段片")
    p.add_argument("--direction",choices=("left","right"),default="right")
    p.add_argument("--strength",type=float,default=1.)
    p.add_argument("--noise-seed",type=int,default=EVAL_SEED)
    args=p.parse_args()
    if args.updates<0:
        p.error("训练次数须为非负数")
    if args.point and not (0<=args.point[0]<WIDTH and 0<=args.point[1]<HEIGHT):
        p.error("点的列号为 0–15，行号为 0–11")
    if not math.isfinite(args.strength):
        p.error("控制强度须为有限数")
    device=torch.device(args.device)
    if args.replay is None:
        save_training(args.root,args.out,device,args.updates,args.seed)
    checkpoint=args.replay or args.out/"branch.pt"
    if args.point:
        custom_sample(args.root,checkpoint,args.out,device,args.point,args.direction,args.strength,args.noise_seed)
    else:
        evaluate_run(args.root,checkpoint,args.out,device)


if __name__=="__main__":
    main()
