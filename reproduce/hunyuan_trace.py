"""实验 O：沿 HunyuanVideo 的原始文生视频路径，计算张量形状和流采样步。"""

import argparse
import json
import math
from pathlib import Path

from flow_coupling import EVAL_COUNT, mae, quantile, velocity


REVISION = "e965abb39f104f618ebd0d9ae27630ee8e55cecc"
SOURCE = f"https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/{REVISION}/"
BUDGETS = (4, 8, 30)
SHIFTS = (1, 3, 9)
RANKS = (.1, .25, .6, .75, .9)


def layout(frames, height, width):
    if min(frames, height, width) < 1 or (frames-1) % 4:
        raise ValueError("高宽须为正，帧数应取 1、5、9 等 4k+1 的整数")
    h, w = ((s+15)//16*16 for s in (height, width))
    t = (frames-1)//4+1
    grid = (t,h//16,w//16)
    return {"requested":(frames,height,width), "canvas":(frames,h,w),
            "latent":(1,16,t,h//8,w//8), "grid":grid, "count":math.prod(grid)}


def address(grid, token, component):
    nt, nh, nw = grid
    if not 0 <= token < nt*nh*nw or not 0 <= component < 64:
        raise ValueError("块号或输出分量超出当前网格")
    t, spatial = divmod(token,nh*nw)
    r, c = divmod(spatial,nw)
    channel, inside = divmod(component,4)
    dy, dx = divmod(inside,2)
    return (t,r,c), (0,channel,t,2*r+dy,2*c+dx)


def knots(steps, shift):
    """发布示例启用 flow-reverse，先从 1 排到 0，再变换刻度。"""
    if steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("步数和 shift 须为正")
    positions = [(steps-i)/steps for i in range(steps+1)]
    return [shift*r/(1+(shift-1)*r) for r in positions]


def solve(sigmas, initial):
    """用实验 K 可精确计算的速度场代替大网络，按源码的 sigma 单位更新。"""
    x = initial
    path = [x]
    for sigma, following in zip(sigmas,sigmas[1:]):
        speed_sigma = -velocity(x,1-sigma)
        x += (following-sigma)*speed_sigma
        path.append(x)
    return path


def guidance_features(strength):
    encoded = strength*1000
    phases = [encoded*math.exp(-math.log(10000)*j/128) for j in range(128)]
    return [math.cos(phi) for phi in phases]+[math.sin(phi) for phi in phases]


def study():
    qs = [(i+.5)/EVAL_COUNT for i in range(EVAL_COUNT)]
    starts = [quantile(q,0) for q in qs]
    targets = [quantile(q,1) for q in qs]
    runs = {}
    for n in BUDGETS:
        runs[str(n)] = {}
        for g in SHIFTS:
            sigmas = knots(n,g)
            ends = [solve(sigmas,z)[-1] for z in starts]
            traces = []
            for q in RANKS:
                path = solve(sigmas,quantile(q,0))
                traces.append({"q":q,"x":path,"ref":[quantile(q,1-s) for s in sigmas],
                               "v_sigma":[-velocity(x,1-s) for x,s in zip(path[:-1],sigmas)]})
            runs[str(n)][str(g)] = {"sigma":sigmas,"mae":mae(ends,targets),"traces":traces}
    return {"runs":runs,"ranks":RANKS,"budgets":BUDGETS,"shifts":SHIFTS,"pairs":EVAL_COUNT,
            "exact":[[quantile(q,i/64) for i in range(65)] for q in RANKS],
            "small":layout(17,64,96),"release":layout(129,720,1280),"revision":REVISION}


def run(output):
    from hunyuan_figures import render
    data = study()
    output.mkdir(parents=True,exist_ok=True)
    (output/"numbers.json").write_text(json.dumps(data,separators=(",",":")),encoding="utf-8")
    render(data,output)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("replay/hunyuan"))
    args = parser.parse_args()
    result = run(args.out)
    print("打开",args.out/"index.html")
    for n, variants in result["runs"].items():
        print(n,"步，shift 依次取 1、3、9 的终点 MAE：",*[round(v["mae"],5) for v in variants.values()])
