"""重演实验 J 前两批训练，记录两个零出口及点图入口的梯度和实际步后效应。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from control_lab import Branch, controlled, grad_norm, teach
from hint_probe import dots, tensor_record
from pixel_lab import BATCH, FRAMES, HEIGHT, WIDTH, TinyVideo


ROOT=Path(__file__).resolve().parents[1]
CHANNELS=(0,1,2)
LAYERS=("fine","coarse","hint","enter","down","middle")
EPS=1e-4


def norms(branch):
    return {name:grad_norm(getattr(branch,name)) for name in LAYERS}


@torch.no_grad()
def finite_differences(base,branch,state,tick,guide,target):
    """在相同参数处用 float64 对整批真实损失做中央差分，仅扰动一枚出口权重。"""
    precise_base=copy.deepcopy(base).double()
    probe=copy.deepcopy(branch).double()
    inputs=(state.double(),tick.double(),guide.double())
    target=target.double()
    def loss():
        return (controlled(precise_base,probe,*inputs)-target).square().mean().item()
    tested=[]
    for o in CHANNELS:
        row=[]
        for i in CHANNELS:
            w=probe.coarse.weight[o,i,0,0]
            saved=w.item()
            w.fill_(saved+EPS)
            plus=loss()
            w.fill_(saved-EPS)
            minus=loss()
            w.fill_(saved)
            row.append([minus,plus,(plus-minus)/(2*EPS)])
        tested.append(row)
    return tested


def selected_volume(tensor,channels):
    # 位置按样片、帧、行、列排列，各位置最后放可选的通道。
    _,_,h,w=tensor.shape
    return tensor[:,channels].permute(0,2,3,1).reshape(BATCH,FRAMES,h,w,len(channels)).contiguous()


@torch.no_grad()
def fixed_response(base,branch,x,t,a,b,step):
    original=base(x,t)
    pa=controlled(base,branch,x,t,a)
    pb=controlled(base,branch,x,t,b)
    difference=pb-pa
    return {"updates":step,"base_max":float((pa-original).abs().max()),
            "edit_max":float(difference.abs().max()),"mean_by_frame":difference[0].abs().mean((-2,-1)).tolist(),
            "base":tensor_record(original[0]),"a":tensor_record(pa[0]),"b":tensor_record(pb[0]),
            "hint_weight_max":float(branch.hint.weight.abs().max()),
            "coarse_weight_max":float(branch.coarse.weight.abs().max())}


def run(root,out):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    base_file=root/"pixel-lab/weights.pt"
    branch_file=root/"control-lab/branch.pt"
    probe_file=root/"control-lab/probe.json"
    trained=torch.load(branch_file,map_location="cpu",weights_only=True)
    fingerprint=hashlib.sha256(base_file.read_bytes()).hexdigest()
    existing=json.loads(probe_file.read_text())
    branch_hash=hashlib.sha256(branch_file.read_bytes()).hexdigest()
    if fingerprint!=trained["base_sha256"] or existing["base_sha256"]!=fingerprint or existing["branch_sha256"]!=branch_hash:
        raise ValueError("实验 J 的权重与所用的单步读数不配套")
    from hint_report import values
    x=torch.tensor(values(existing["fixed"]["x"])).reshape(1,FRAMES,HEIGHT,WIDTH)
    tick=torch.tensor([.5])
    points=[existing["answers"][name]["points"] for name in ("left","moved")]
    a,b=map(dots,points)
    seed=trained["seed"]
    torch.manual_seed(seed)
    base=TinyVideo(True)
    base.load_state_dict(torch.load(base_file,map_location="cpu",weights_only=True)["joint"])
    branch=Branch(base)
    base.eval().requires_grad_(False)
    original={n:p.clone() for n,p in base.named_parameters()}
    snapshots=[]
    effects=[fixed_response(base,branch,x,tick,a,b,0)]
    held={}
    def retain(_module,inputs,output):
        if torch.is_grad_enabled():
            low=inputs[0]
            low.retain_grad()
            output.retain_grad()
            held.update(low=low,residual=output)
    handle=branch.coarse.register_forward_hook(retain)
    def weights():
        return branch.coarse.weight.detach()[:,:,0,0][:,CHANNELS].contiguous().tolist()
    def observe(step,state,t,guide,target,loss):
        if step:
            snapshots[-1]["after"]=weights()
            effects.append(fixed_response(base,branch,x,tick,a,b,step))
        h=held["low"]
        g=held["residual"].grad
        before=weights()
        # 样片 0 另保留 32 个完整输出梯度，与返回粗特征的梯度作逐格对照。
        all_g=g[:FRAMES].permute(0,2,3,1).contiguous()
        returned=selected_volume(h.grad,CHANNELS)[0]
        snapshots.append({"batch":step+1,"loss":float(loss.detach()),"norms":norms(branch),
             "before":before,"h":tensor_record(selected_volume(h.detach(),CHANNELS)),
             "g":tensor_record(selected_volume(g,CHANNELS)),"all_g0":tensor_record(all_g),
             "returned0":tensor_record(returned),
             "dw":branch.coarse.weight.grad[:,:,0,0][list(CHANNELS)][:,CHANNELS].tolist(),
             "db":branch.coarse.bias.grad[list(CHANNELS)].tolist(),
             "finite":finite_differences(base,branch,state,t,guide,target)})
    try:
        teach(base,branch,seed,2,torch.device("cpu"),observe=observe)
    finally:
        handle.remove()
    snapshots[-1]["after"]=weights()
    effects.append(fixed_response(base,branch,x,tick,a,b,2))
    branch.load_state_dict(trained["branch"])
    effects.append(fixed_response(base,branch,x,tick,a,b,trained["updates"]))
    data={"seed":seed,"batch_size":BATCH,"frames":FRAMES,"spatial":[6,8],"channels":list(CHANNELS),
          "lr":.001,"adam_eps":1e-8,"finite_eps":EPS,"points":points,"batches":snapshots,"response":effects,
          "base_delta":max(float((p-original[n]).abs().max()) for n,p in base.named_parameters()),
          "frozen_grads":sum(p.grad is not None for p in base.parameters()),
          "hashes":{"base":fingerprint,"branch":branch_hash,"probe":hashlib.sha256(probe_file.read_bytes()).hexdigest()}}
    out.mkdir(parents=True,exist_ok=True)
    (out/"gate-training.json").write_text(json.dumps(data,separators=(",",":"),allow_nan=False),encoding="utf-8")
    from gate_lesson import render
    render(data,out,guide=(out/"edit-points.html").is_file())
    print([(r["updates"],r["base_max"],r["edit_max"]) for r in effects])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--out",type=Path,default=Path("replay/gates"))
    args=p.parse_args()
    run(args.root,args.out)
