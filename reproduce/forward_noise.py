"""两次加噪和一次直达。用标准库核对一个像素、整张图和成对的随机样本。"""
import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from reproduce import clean_pixel, normal_draw

MODES = ("chain", "reuse", "fresh")
NAMES = {"chain":"逐步抽新噪声", "reuse":"两层共用 ε₁", "fresh":"第二层独立直抽"}
COUNT = 4096
HALF_WINDOW = .06
PIXEL = 15*48+24


def two_steps(y, e1, e2, keep=.75):
    """keep 是第二步保留旧状态的幅度系数，第一步固定为 0.8。"""
    if not 0 <= keep <= 1:
        raise ValueError("旧状态保留幅度须在 0 到 1 之间")
    new = math.sqrt(1-keep**2)
    signal = .8*keep
    sigma = math.sqrt(1-signal**2)
    first = .8*y + .6*e1
    effective = (.6*keep*e1 + new*e2)/sigma
    return {"first":first, "chain":keep*first+new*e2, "effective":effective,
            "direct":signal*y+sigma*effective, "reuse":signal*y+sigma*e1,
            "fresh":signal*y+sigma*e2, "signal":signal, "sigma":sigma, "new":new}


def theory(keep, at):
    """固定干净亮度 0.8，分别计算跨层相关系数及已知第一层时的条件均值与宽度。"""
    cfg=two_steps(.8,0.,0.,keep)
    mean=.8*cfg["signal"]
    sigma=cfg["sigma"]
    return {"mean":mean, "sd":sigma, "chain":[.6*keep/sigma,keep*at,cfg["new"]],
            "reuse":[1.,mean+sigma*(at-.64)/.6,0.], "fresh":[0.,mean,sigma]}


def noise_pairs(seed, count):
    draw=normal_draw(seed)
    return [(draw(),draw()) for _ in range(count)]


def scatter(seed, keep, at):
    points={key:[] for key in MODES}
    for e1,e2 in noise_pairs(seed,COUNT):
        trial=two_steps(.8,e1,e2,keep)
        for key in MODES:
            points[key].append((trial["first"],trial[key]))
    rows={}
    for key,pairs in points.items():
        first,last=zip(*pairs)
        mx,my=fmean(first),fmean(last)
        sx=math.sqrt(fmean((x-mx)**2 for x in first))
        sy=math.sqrt(fmean((y-my)**2 for y in last))
        corr=fmean((x-mx)*(y-my) for x,y in pairs)/(sx*sy)
        selected=[y for x,y in pairs if at-HALF_WINDOW <= x < at+HALF_WINDOW]
        rows[key]={"mean":my,"sd":sy,"corr":corr,"n":len(selected),
                   "window_mean":fmean(selected) if selected else None}
    return points,rows


def image_inputs(seed):
    return [(clean_pixel(*divmod(idx,48)),a,b) for idx,(a,b) in
            enumerate(noise_pairs(seed+34171,30*48))]


def pictures(seed, keep, e1, e2):
    planes={key:[] for key in ("clean","first","chain","direct","reuse")}
    for idx,(y,a,b) in enumerate(image_inputs(seed)):
        if idx==PIXEL:
            a,b=e1,e2
        tile=two_steps(y,a,b,keep)
        planes["clean"].append(y)
        for key in ("first","chain","direct","reuse"):
            planes[key].append(tile[key])
    return planes


def bins(pairs, *, at=None):
    """把全部概率分进 28 格，两端格子也包含范围之外的尾部。"""
    counts=[0]*28
    for x,y in pairs:
        if at is None or at-HALF_WINDOW <= x < at+HALF_WINDOW:
            index=max(0,min(27,math.floor((y+2.3)/.2)))
            counts[index]+=1
    total=sum(counts)
    return [n/total for n in counts] if total else [0.]*28


if __name__=="__main__":
    cmd=argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("--seed",type=int,default=17)
    cmd.add_argument("--keep",type=float,default=.75,help="第二步旧状态的幅度系数，0 到 1")
    cmd.add_argument("--noise",type=float,nargs=2,default=(-1.,1.),metavar=("E1","E2"))
    cmd.add_argument("--at",type=float,default=.04,help="用来筛选第一层读数的窗口中心")
    cmd.add_argument("--group",action="store_true",help="比较 4,096 对样本")
    cmd.add_argument("--out",type=Path,help="用当前设置生成离线网页和两张 SVG")
    args=cmd.parse_args()
    finite=(args.keep,args.at,*args.noise)
    if not all(map(math.isfinite,finite)) or not 0<=args.keep<=1:
        cmd.error("请输入有限数，keep 取 0 到 1")
    if args.group:
        _,rows=scatter(args.seed,args.keep,args.at)
        print(json.dumps({"每种配方的样本数":COUNT,"窗口半宽":HALF_WINDOW,"理论值":theory(args.keep,args.at),"实测":rows},ensure_ascii=False,indent=2))
    else:
        print(json.dumps(two_steps(.8,*args.noise,args.keep),indent=2))
    if args.out:
        from forward_figures import render
        render(args.out,seed=args.seed,keep=args.keep,e1=args.noise[0],e2=args.noise[1],at=args.at)
