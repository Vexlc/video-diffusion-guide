"""实验 I 延伸：固定已训练权重与扩散时刻，逐帧换输入、逐处开关时间移位。"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from clock_report import Recording
from hint_probe import tensor_record
from pixel_lab import FRAMES, HEIGHT, WIDTH, TinyVideo, mix_neighbors


def switches(key):
    return tuple(bit=="1" for bit in key)


def forward_with_taps(net, states, tick, key, edit=None):
    """按 TinyVideo 的原有计算顺序前向；三个开关仅控制各处通道来源。"""
    def route(x,layer,on):
        sent=mix_neighbors(x,FRAMES,on)
        if edit is not None and edit[0]==layer:
            _,frame,channel,row,col,delta=edit
            sent=sent.clone()  # 关移位时返回原张量；干预不能改写跳跃连接。
            sent[frame,channel,row,col]+=delta
        return sent

    first, middle, last = switches(key)
    features = net.inputs(states,tick)
    skip = net.enter(features)
    routed1 = route(skip,0,first)
    down = net.down(routed1)
    routed2 = route(down,1,middle)
    low = down + net.middle(routed2)
    high = F.interpolate(low,size=(HEIGHT,WIDTH),mode="nearest")
    routed3 = route(high,2,last)
    hidden = net.refine(torch.cat([skip,routed3],dim=1))
    v = net.exit(hidden).reshape_as(states)
    return v,(skip,down,low),(routed1,routed2,routed3)


def check_routes(taps, routed, key):
    """独立按单通道地址检查 PyTorch 的实际输出，包含两个边界帧。"""
    count=0
    for layer,(incoming,outgoing,enabled) in enumerate(zip(taps,routed,switches(key))):
        incoming=incoming[:FRAMES]
        if layer==2:
            incoming=F.interpolate(incoming,size=(HEIGHT,WIDTH),mode="nearest")
        channels=incoming.shape[1]
        for frame in range(FRAMES):
            for channel in range(channels):
                source=frame
                if enabled and channel<channels//4:
                    source=max(0,frame-1)
                elif enabled and channel<channels//2:
                    source=min(FRAMES-1,frame+1)
                if not torch.equal(incoming[source,channel],outgoing[frame,channel]):
                    raise ValueError("通道移位结果与源地址不一致")
        count+=incoming.numel()
    return count


@torch.no_grad()
def export(root, output):
    torch.set_num_threads(2)
    recording_file=root/"pixel-lab/clock-data.js"
    weights=root/"pixel-lab/weights.pt"
    recording=Recording(recording_file)
    fingerprint=hashlib.sha256(weights.read_bytes()).hexdigest()
    if recording.weight_sha256!=fingerprint:
        raise ValueError("输入记录和权重文件不匹配")
    net=TinyVideo(True).eval()
    net.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True)["joint"])
    count=FRAMES*HEIGHT*WIDTH
    def input_at(case):
        start=recording.plane_offset(case,12,0)
        return torch.tensor(recording.states[start:start+count]).reshape(FRAMES,HEIGHT,WIDTH)
    original,donor=input_at(0),input_at(1)
    states=original.repeat(FRAMES+1,1,1,1)
    for j in range(FRAMES):
        states[j+1,j]=donor[j]
    tick=torch.full((len(states),),.5)
    a,sigma,*_=recording.coefficients[12]
    evidence={"base_sha256":fingerprint,"recording_sha256":hashlib.sha256(recording_file.read_bytes()).hexdigest(),
              "step":12,"cases":[1,2],"a":a,"sigma":sigma,"original":tensor_record(original),
              "donor":tensor_record(donor),"taps":{},"runs":{}}
    checked=0
    for bits in itertools.product("01",repeat=3):
        key="".join(bits)
        v,taps,routed=forward_with_taps(net,states,tick,key)
        if key in ("000","111"):
            net.temporal=key=="111"
            if not torch.equal(net(states,tick),v):
                raise ValueError("探针与原模型调用不一致")
        checked+=check_routes(taps,routed,key)
        for layer,tensor in enumerate(taps):
            prefix=key[:layer]
            if prefix not in evidence["taps"]:
                evidence["taps"][prefix]=tensor_record(tensor[:FRAMES])
        evidence["runs"][key]=tensor_record(v)
    evidence["checked"]=checked
    evidence["layers"]={name:tensor_record(param) for name,param in net.state_dict().items()
                        if not name.startswith("enter.")}
    evidence["edits"]=[]
    for key,edit in [("111",(0,2,2,5,7,1.0)),("000",(0,0,4,0,0,-1.0)),
                     ("110",(0,5,14,11,15,.5)),("101",(1,2,10,2,3,-.5)),
                     ("010",(1,0,7,0,0,1.5)),("111",(1,5,30,5,7,.25)),
                     ("011",(2,5,15,11,15,-2.0)),("000",(2,0,31,0,0,1.0))]:
        output_v,_,_=forward_with_taps(net,states[:1],tick[:1],key,edit)
        evidence["edits"].append({"key":key,"edit":edit,"v":tensor_record(output_v)})
    output.mkdir(parents=True,exist_ok=True)
    (output/"temporal.json").write_text(json.dumps(evidence,separators=(",",":")),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out",type=Path,default=Path("replay/temporal-shift"))
    args=parser.parse_args()
    export(args.root,args.out)
    from temporal_report import render
    render(args.out,args.root)
    print(args.out/"temporal-shift.html")
