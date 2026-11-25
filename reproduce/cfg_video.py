"""用作者版五帧 ControlVideo 实测文字 CFG 与深度控制。需要作者环境、GPU 和预训练权重。"""
import argparse
import functools
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from depth_video import F,H,W,NEGATIVE,PROMPT,load_pipeline,maps,packed
import numpy as np
import torch


def channel_plane(tensor,branch):
    plane=tensor[branch,0].detach().contiguous().cpu().numpy().astype("<f2",copy=False)
    return packed(plane)


@torch.no_grad()
def record_run(pipe,frames,seed,steps,w,strength,negative):
    watched={0,steps//2,steps-1}
    samples=[]
    forecasts=None
    residual=None
    calls=0

    def read_control(module,args,out):
        nonlocal residual
        if calls in watched:
            residual=channel_plane(out[0][0],-1)

    def read_unet(module,args,out):
        nonlocal forecasts
        if calls in watched:
            forecasts=out.sample.detach()

    step=pipe.scheduler.step
    @functools.wraps(step)
    def read_step(prediction,tick,state,*args,**kwargs):
        nonlocal calls,forecasts,residual
        if calls in watched:
            batch=forecasts.shape[0]
            if batch!=(2 if w>1 else 1):
                raise ValueError("实际文字分支数与作者采样逻辑不一致")
            pos=forecasts[-1:]
            expected=forecasts[:1]+w*(pos-forecasts[:1]) if batch==2 else pos
            if not torch.equal(expected,prediction):
                raise ValueError("复算 CFG 与采样器收到的整张量不相同")
            raw=state.detach().contiguous().cpu().numpy().tobytes()
            item={"n":calls,"tick":int(tick),"batch":batch,"state_sha256":hashlib.sha256(raw).hexdigest(),
                  "pos":channel_plane(pos,0),"actual":channel_plane(prediction,0),"residual":residual}
            if batch==2:
                item["ref"]=channel_plane(forecasts,0)
            samples.append(item)
            forecasts=residual=None
        calls+=1
        return step(prediction,tick,state,*args,**kwargs)

    hooks=[pipe.controlnet.register_forward_hook(read_control),pipe.unet.register_forward_hook(read_unet)]
    try:
        with patch.object(pipe.scheduler,"step",read_step):
            video=pipe(PROMPT,negative_prompt=negative,width=W,height=H,video_length=F,frames=frames,
                       num_inference_steps=steps,guidance_scale=float(w),controlnet_conditioning_scale=float(strength),
                       smooth_steps=[],eta=0.,generator=torch.Generator("cuda").manual_seed(seed)).videos
    finally:
        for hook in hooks:
            hook.remove()
    if calls!=steps or len(samples)!=len(watched):
        raise ValueError("记录中的采样调用数与日程不同")
    array=video[0].permute(1,2,3,0).float().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError("影片包含非有限值")
    rgb=(array.clip(0,1)*255).round().astype("uint8")
    return {"w":w,"scale":strength,"reference":negative,"samples":samples,"film":[packed(frame) for frame in rgb]}


def run(repo,weights,out,seed,steps):
    pipe,provenance=load_pipeline(repo,weights,seed)
    hints=maps(False)
    record={**provenance,"size":[W,H],"plane_shape":[F,H//8,W//8],"steps":steps,"seed":seed,"prompt":PROMPT,
            "negative":NEGATIVE,"maps":[packed(frame.convert("L")) for frame in hints],"scheduler":"DDIM",
            "eta":0.,"smooth_steps":[],"shared_frame_noise":True,"runs":{}}
    for w,strength,empty in ((2,0,False),(2,1,False),(7.5,0,False),(7.5,1,False),
                             (0,1,False),(1,1,False),(7.5,1,True)):
        key=f"w{w}-d{strength}"+("-empty" if empty else "")
        record["runs"][key]=record_run(pipe,hints,seed,steps,w,strength,"" if empty else NEGATIVE)
        print("完成",key,flush=True)
    if len({v["samples"][0]["state_sha256"] for v in record["runs"].values()})!=1:
        raise ValueError("各次采样用的初始噪声不同")
    out.mkdir(parents=True,exist_ok=True)
    (out/"cfg.json").write_text(json.dumps(record,separators=(",",":")),encoding="utf-8")
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("replay/cfg-video"))
    p.add_argument("--seed",type=int,default=1947)
    p.add_argument("--steps",type=int,choices=range(3,101),metavar="3..100",default=20)
    args=p.parse_args()
    data=run(args.repo,args.weights,args.out,args.seed,args.steps)
    from cfg_report import render
    render(data,args.out)
