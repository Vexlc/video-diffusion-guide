"""两个干净像素能产生同一观测。由高斯密度求最优猜测，再用真实出题记录核对。"""
import argparse
import json
import math
from pathlib import Path

from reproduce import clean_pixel, normal_draw


SIGNAL, SIGMA = .8, .6
BODY, BACKGROUND = clean_pixel(15,24), clean_pixel(5,24)
MODES = ("hidden", "shared", "independent")
SEED = 539
RADIUS = .1


def phi(z):
    return math.exp(-.5*z*z)/math.sqrt(2*math.pi)


def noise_for(x, clean):
    return (x-SIGNAL*clean)/SIGMA


def normalize(body, road):
    return body/(body+road)


def neighbor_prior(mode, neighbor):
    """两类初始各半。共享部位时，邻帧观测会更新本帧的类别概率。"""
    if mode not in MODES:
        raise ValueError("未知的邻帧关系")
    if mode!="shared":
        return .5
    return normalize(phi(noise_for(neighbor,BODY)),phi(noise_for(neighbor,BACKGROUND)))


def estimate(x, mode="hidden", neighbor=.64):
    """给定实际可用的观测，按两种标签的条件概率，分别算干净值和噪声的均值。"""
    prior=neighbor_prior(mode,neighbor)
    z_body,z_road=noise_for(x,BODY),noise_for(x,BACKGROUND)
    body,road=prior*phi(z_body),(1-prior)*phi(z_road)
    p=normalize(body,road)
    noise=p*z_body+(1-p)*z_road
    return {"body":p, "clean":p*BODY+(1-p)*BACKGROUND, "noise":noise,
            "score":-noise/SIGMA, "density":(body+road)/SIGMA,
            "loss":p*(z_body-noise)**2+(1-p)*(z_road-noise)**2}


def slope_from_density(x,mode,neighbor,step=.0001):
    """保持邻帧线索不动，分别求当前读数左右的密度，再计算对数的割线斜率。"""
    right=estimate(x+step,mode,neighbor)["density"]
    left=estimate(x-step,mode,neighbor)["density"]
    return (math.log(right)-math.log(left))/(2*step)


def window_mass(x,clean):
    lo,hi=noise_for(x-RADIUS,clean),noise_for(x+RADIUS,clean)
    probability=.5*(math.erf(hi/math.sqrt(2))-math.erf(lo/math.sqrt(2)))
    # 对窗口里的 z φ(z) 积分；φ 的导数为 −z φ(z)。
    return probability,phi(lo)-phi(hi)


def window_expected(x,mode,neighbor):
    """求有限观察窗的理论平均噪声标签，用作抽样的同口径参照。"""
    total=moment=bright=0.
    for y in (BODY,BACKGROUND):
        mass,first=window_mass(x,y)
        factor=window_mass(neighbor,y)[0] if mode=="shared" else 1.
        total+=mass*factor
        moment+=first*factor
        if y==BODY:
            bright=mass*factor
    return {"body":bright/total,"noise":moment/total}


def sample_window(x,mode,neighbor,count=65536,seed=SEED):
    """实际出题，保留本帧观测和可见邻帧观测都在指定窗口里的题。"""
    draw=normal_draw(seed)
    bright=kept=0
    total=0.
    for _ in range(count):
        y=BODY if draw()>0 else BACKGROUND
        independent=BODY if draw()>0 else BACKGROUND
        z,other_z=draw(),draw()
        observed=SIGNAL*y+SIGMA*z
        other_y=independent if mode=="independent" else y
        other=SIGNAL*other_y+SIGMA*other_z
        if abs(observed-x)>RADIUS or (mode!="hidden" and abs(other-neighbor)>RADIUS):
            continue
        bright+=int(y==BODY)
        kept+=1
        total+=z
    return {"kept":kept,"body_count":bright,"noise":total/kept if kept else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x",type=float,default=.04)
    parser.add_argument("--neighbor",type=float,default=.64)
    parser.add_argument("--mode",choices=MODES,default="shared")
    parser.add_argument("--count",type=int,default=65536)
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--out",type=Path,help="重画两幅讲义插图和离线练习")
    args=parser.parse_args()
    if not all(-2<=v<=2 for v in (args.x,args.neighbor)) or args.count<1:
        parser.error("两个观测读数范围为 −2 到 2，题数须为正整数")
    slope=slope_from_density(args.x,args.mode,args.neighbor)
    record={"point":estimate(args.x,args.mode,args.neighbor),"slope_from_density":slope,
            "exact_window":window_expected(args.x,args.mode,args.neighbor),
            "sample_window":sample_window(args.x,args.mode,args.neighbor,args.count,args.seed)}
    print(json.dumps(record,ensure_ascii=False,indent=2))
    if args.out:
        from posterior_figures import render
        render(args.out)


if __name__=="__main__":
    main()
