"""用第 8 页的两种干净来源，对照一次 DDPM、DDIM 更新和已知分布的精确反向抽样。"""
import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path

from posterior_lab import BACKGROUND, BODY, SIGNAL, SIGMA, estimate, phi
from reproduce import normal_draw


# 16 个等宽组，左右各保留一个越界组。边界属于右边的组。
EDGES=tuple(round(-1.2+.15*i,3) for i in range(17))
WINDOW=.04


def step_parameters(x, target):
    """target 是较干净层的信号系数。用第 8 页的后验权重合并两种真实来源。"""
    if not SIGNAL<target<=1 or not math.isfinite(x) or not -2<=x<=2:
        raise ValueError("本帧 x 须在 −2 到 2 之间，目标信号系数须大于 0.8 且不超过 1")
    guess=estimate(x)
    ratio=SIGNAL/target
    beta=1-ratio**2
    target_var=1-target**2
    within=beta*target_var/SIGMA**2
    # 对已知原值 y，下一格的中心是 cy·y + cx·x。干净端点精确退化为 y。
    cy=1. if target==1 else target*beta/SIGMA**2
    cx=ratio*target_var/SIGMA**2
    body,back=cy*BODY+cx*x,cy*BACKGROUND+cx*x
    p=guess["body"]
    mean=p*body+(1-p)*back
    between=p*(1-p)*(body-back)**2
    return {"x":x,"target":target,"p":p,"clean":guess["clean"],"noise":guess["noise"],
            "ratio":ratio,"beta":beta,"cy":cy,"cx":cx,"body":body,"back":back,"mean":mean,
            "std":math.sqrt(within),"within":within,"between":between,"variance":within+between,
            "ddim":target*guess["clean"]+math.sqrt(target_var)*guess["noise"]}


def draws(step, uniform, fresh_noise):
    """三种走法共用输入和去噪预测；两种随机方法共用同一个新高斯数。"""
    center=step["body"] if uniform<step["p"] else step["back"]
    offset=step["std"]*fresh_noise
    return {"exact":center+offset,"ddpm":step["mean"]+offset,"ddim":step["ddim"]}


def normal_bins(center,std):
    if std==0:
        bins=[0.0]*(len(EDGES)+1)
        bins[bisect_right(EDGES,center)]=1.
        return bins
    cuts=[0.]+[.5*math.erfc((center-cut)/(std*math.sqrt(2))) for cut in EDGES]+[1.]
    return [right-left for left,right in zip(cuts,cuts[1:])]


def distributions(step):
    body=normal_bins(step["body"],step["std"])
    back=normal_bins(step["back"],step["std"])
    p=step["p"]
    return {"exact":[p*b+(1-p)*d for b,d in zip(body,back)],
            "ddpm":normal_bins(step["mean"],step["std"]),"ddim":normal_bins(step["ddim"],0)}


def forward_pairs(target,count,seed):
    """独立按正向链出题，记录先前较干净的读数 v 与后来更吵的读数 x。"""
    draw=normal_draw(seed)
    prior_sigma=math.sqrt(1-target**2)
    ratio=SIGNAL/target
    extra=math.sqrt(1-ratio**2)
    for _ in range(count):
        clean=BODY if draw()>0 else BACKGROUND
        early=target*clean+prior_sigma*draw()
        late=ratio*early+extra*draw()
        yield early,late


def band_probability(low,high):
    """标准正态落入 [low,high] 的概率；右尾用补分布避免两个 1 相减。"""
    if low>0:
        return .5*(math.erfc(low/math.sqrt(2))-math.erfc(high/math.sqrt(2)))
    return .5*(math.erfc(-high/math.sqrt(2))-math.erfc(-low/math.sqrt(2)))


def finite_window(x,target):
    """直接按正向链积分每一档的入窗概率，和反向公式、随机出题均独立。"""
    ratio=SIGNAL/target
    early_sd=math.sqrt(1-target**2)
    extra=math.sqrt(1-ratio**2)
    bins=[0.]*(len(EDGES)+1)
    for y in (BODY,BACKGROUND):
        center=target*y
        if early_sd==0:
            bins[bisect_right(EDGES,y)]+=band_probability((x-WINDOW-ratio*y)/extra,
                                                           (x+WINDOW-ratio*y)/extra)
            continue
        # 积分变量是较干净层的标准噪声。按亮度分组、正态尾部和入窗边沿切分。
        cuts={-10.,10.}
        cuts.update(k/2 for k in range(-19,20))
        cuts.update((v-center)/early_sd for v in EDGES)
        for bound in (x-WINDOW,x+WINDOW):
            for offset in (-9,-5,-2,0,2,5,9):
                cuts.add((bound-ratio*center+offset*extra)/(ratio*early_sd))
        cuts=sorted(z for z in cuts if -10<=z<=10)
        for left,right in zip(cuts,cuts[1:]):
            idx=bisect_right(EDGES,center+early_sd*(left+right)/2)
            # 每段复合 Simpson。两个来源均为 1/2，该因子在归一化时相消。
            width=(right-left)/24
            for i in range(25):
                z=left+i*width
                mean=ratio*(center+early_sd*z)
                prob=band_probability((x-WINDOW-mean)/extra,(x+WINDOW-mean)/extra)
                weight=1 if i in (0,24) else 4 if i%2 else 2
                bins[idx]+=weight*width/3*phi(z)*prob
    total=sum(bins)
    return [mass/total for mass in bins]


def forward_check(pairs,x,target):
    counts=[0]*(len(EDGES)+1)
    kept=0
    for early,late in pairs:
        if abs(late-x)<=WINDOW:
            counts[bisect_right(EDGES,early)]+=1
            kept+=1
    return {"kept":kept,"counts":counts,"expected":finite_window(x,target)}


def cases(xs,targets,count,seed):
    results=[]
    for target in targets:
        observations=list(forward_pairs(target,count,seed))
        for x in xs:
            step=step_parameters(x,target)
            results.append({"step":step,"bins":distributions(step),"forward":forward_check(observations,x,target)})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x",type=float,default=.04,help="第 8 页本帧读数，只看本帧")
    parser.add_argument("--target",type=float,default=.96,help="较干净层的信号系数")
    parser.add_argument("--count",type=int,default=65536,help="正向出题数")
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--uniform",type=float,default=.25,help="精确参照抽来源的均匀数 [0,1)")
    parser.add_argument("--fresh",type=float,default=0.,help="两种随机走法共用的新高斯数")
    parser.add_argument("--out",type=Path,help="生成离线网页与 PDF 的两页插图")
    args=parser.parse_args()
    if not -2<=args.x<=2 or not SIGNAL<args.target<=1 or args.count<1 or not 0<=args.uniform<1 or not math.isfinite(args.fresh):
        parser.error("请检查读数、目标层、题数和抽样输入的范围")
    row=cases([args.x],[args.target],args.count,args.seed)[0]
    print(json.dumps({**row,"one_draw":draws(row["step"],args.uniform,args.fresh)},indent=2,ensure_ascii=False))
    if args.out:
        from reverse_pixel_figures import render
        render(args.out,args.x,args.target,args.count,args.seed,args.uniform,args.fresh)


if __name__=="__main__":
    main()
