"""用标准库核算零出口的共享权重梯度，绘制训练时间线和可离线操作的探针。"""
import argparse
import json
import math
from pathlib import Path

from draw import INK, MUTED, RUST, TEAL, rect, svg, text
from hint_report import tile, values


def weight_sum(batch,output=0,channel=0,sample=0,frame=2,row=2,col=3):
    """一个出口权重跨 24 段、6 帧、6×8 个位置共享。用每个位置的 g*h 还原梯度。"""
    h=values(batch["h"])
    g=values(batch["g"])
    count,frames,height,width,channels=batch["h"]["shape"]
    pos=((sample*frames+frame)*height+row)*width+col
    terms=[h[i+channel]*g[i+output] for i in range(0,len(h),channels)]
    start=(sample*frames+frame)*height*width
    first=sample*frames*height*width
    return {"h":h[pos*channels+channel],"g":g[pos*channels+output],"product":terms[pos],
            "frame":math.fsum(terms[start:start+height*width]),"sample":math.fsum(terms[first:first+frames*height*width]),
            "total":math.fsum(terms),"autograd":batch["dw"][output][channel],
            "before":batch["before"][output][channel],"after":batch["after"][output][channel],
            "central":batch["finite"][output][channel][2],"count":len(terms)}


def return_sum(batch,channel=0,frame=2,row=2,col=3):
    """样片 0，本位置的 32 份输出梯度通过转置权重各送回来一项。"""
    pos=(frame*6+row)*8+col
    g=values(batch["all_g0"])[pos*32:(pos+1)*32]
    terms=[weights[channel]*v for weights,v in zip(batch["before"],g)]
    return {"terms":terms,"total":math.fsum(terms),"autograd":values(batch["returned0"])[pos*3+channel]}


def palette(v):
    v=max(-1.,min(1.,v))
    neutral=(239,241,232)
    ink=(173,73,44) if v<0 else (0,105,93)
    return tuple(round(c+abs(v)*(end-c)) for c,end in zip(neutral,ink))


def draw_map(data,w,h,x,y,width,limit):
    return tile(x,y,w,h,width,[palette(v/limit if limit else 0.) for v in data])


def deltas(response):
    base,a,b=(values(response[k]) for k in ("base","a","b"))
    return [p-q for p,q in zip(a,base)],[p-q for p,q in zip(b,a)]


def response_chart(data):
    stages=data["response"]
    limits=[max(s[k] for s in stages) for k in ("base_max","edit_max")]
    parts=[text(12,24,"固定带噪视频，看第 2 帧的预测差",17,TEAL,bold=True)]
    for i,stage in enumerate(stages):
        x=160+119*i
        parts.append(text(x+40,52,f'更新 {stage["updates"]} 次',14,INK,"middle",True))
        for r,(delta,key) in enumerate(zip(deltas(stage),("base_max","edit_max"))):
            top=64+92*r
            parts.extend((draw_map(delta[384:576],16,12,x,top,80,limits[r]),
                          text(x+40,top+79,f'{stage[key]:.3g}',14,INK,"middle")))
    parts.extend((text(12,89,"接入支路",16,TEAL,bold=True),text(12,112,"原提示预测 − 基座预测",12,MUTED),
                  text(12,181,"移动一个点",16,TEAL,bold=True),text(12,204,"新提示预测 − 原提示预测",12,MUTED)))
    early=stages[2]
    delta=deltas(early)[1][384:576]
    parts.extend((draw_map(delta,16,12,12,254,76,early["edit_max"]),
                  text(107,272,"单独放大第 2 次更新的移点差",16,INK,bold=True),
                  text(107,297,f'把色阶上限从 {limits[1]:.3g} 降到 {early["edit_max"]:.3g}，细微的正负差才可见。',14,MUTED)))
    # 在差值热图前画出两张真正送入网络的单像素点图，坐标与浏览器一致。
    hints = [text(12,22,"先看给模型的提示",17,TEAL,bold=True),
             text(12,47,"帧 2，行 5 不变",14,INK),
             text(12,71,"点向右移四格",14,MUTED)]
    for i, points in enumerate(data["points"]):
        x, y = points[2]
        left = 171 + 185*i
        mask = [float((c,r)==(x,y)) for r in range(12) for c in range(16)]
        hints.extend((draw_map(mask,16,12,left,7,88,1),
                      text(left+44,91,f"({x}, {y})",14,INK,"middle")))
    hints.extend((text(298,50,"→",24,TEAL,"middle",True),
                  text(474,37,"其余五帧",15,INK,bold=True),
                  text(474,62,"沿用各自原来的点",14,MUTED)))
    return svg("点图在同一帧向右移四格，再比较接入支路与挪点的两种预测差",426,
               hints+['<g transform="translate(0 106)">']+parts+["</g>"])


def product_chart(batch):
    h,g=values(batch["h"]),values(batch["g"])
    start=2*48*3
    h=h[start:start+48*3:3]
    g=g[start:start+48*3:3]
    sets=(("支路特征 h",h),("出口收到的梯度 g",g),("每格贡献 g × h",[a*b for a,b in zip(h,g)]))
    parts=[text(12,24,"同一张 6×8 网格的三次读法",17,TEAL,bold=True)]
    for i,(title,vals) in enumerate(sets):
        left=21+i*216
        parts.extend((text(left,56,title,15,INK,bold=True),
                      draw_map(vals,8,6,left,70,144,max(map(abs,vals))),
                      f'<rect x="{left+54}" y="106" width="18" height="18" stroke="#d88d21" stroke-width="3" fill="none"/>',
                      text(left,204,f'框内 {vals[19]:.5g}',15,INK,bold=True)))
    parts.append(text(12,238,"橙框为第 2 行、第 3 列。青色为正，棕色为负；每幅图按各自最大绝对值着色。",14,MUTED))
    return svg("样片 0 帧 2 的特征、上游梯度和逐格乘积",250,parts)


def render(data,out,*,guide=False):
    out.mkdir(parents=True,exist_ok=True)
    (out/"gate-response.svg").write_text(response_chart(data),encoding="utf-8")
    (out/"gate-product.svg").write_text(product_chart(data["batches"][0]),encoding="utf-8")
    template=Path(__file__).with_name("gate_training_ui.html").read_text(encoding="utf-8")
    link='<a href="edit-points.html">在已训练模型里自由挪点</a>' if guide else ""
    (out/"gate-training.html").write_text(template.replace("__J_LINK__",link).replace(
        "__GATES__",json.dumps(data,separators=(",",":"),allow_nan=False)),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--batch",type=int,choices=(1,2),default=1)
    p.add_argument("--output",type=int,choices=range(3),default=0)
    p.add_argument("--channel",type=int,choices=range(3),default=0)
    p.add_argument("--sample",type=int,choices=range(24),default=0)
    p.add_argument("--frame",type=int,choices=range(6),default=2)
    p.add_argument("--row",type=int,choices=range(6),default=2)
    p.add_argument("--col",type=int,choices=range(8),default=3)
    p.add_argument("--render",action="store_true")
    a=p.parse_args()
    data=json.loads((a.directory/"gate-training.json").read_text())
    if a.render:
        render(data,a.directory,guide=(a.directory/"edit-points.html").is_file())
    batch=data["batches"][a.batch-1]
    print("单格到整批",json.dumps(weight_sum(batch,a.output,a.channel,a.sample,a.frame,a.row,a.col),ensure_ascii=False))
    back=return_sum(batch,a.channel,a.frame,a.row,a.col)
    print("样片 0 同位置的返回梯度",back["total"],"自动微分",back["autograd"])
