"""只用 Python 标准库运行两台六帧轨迹生成器；在同一噪声下开关位置支路。"""
import argparse
import base64
import json
import math
import operator
import random
import struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STEPS=24
TASKS=("diffusion","flow")
ROWS=(3,7)
FIRST_SEED=1700
COUNT=64


def load_models(path):
    saved=json.loads(path.read_text(encoding="utf-8"))
    models={}
    for task,state in saved["models"].items():
        params={}
        for name,packed in state.items():
            shape=packed["shape"]
            vals=[v for v, in struct.iter_unpack("<f",base64.b64decode(packed["data"],validate=True))]
            if len(vals)!=math.prod(shape):
                raise ValueError("权重维度与存储数据不一致")
            params[name]=[vals[i:i+shape[1]] for i in range(0,len(vals),shape[1])] if len(shape)==2 else vals
        models[task]=params
    return models


def linear(p,name,x):
    weight=p[name+".weight"]
    bias=p.get(name+".bias",(0.,)*len(weight))
    return [b+sum(map(operator.mul,row,x)) for row,b in zip(weight,bias)]


def predict(params,x,t,row,gain):
    a,s=math.sin(t*math.pi/2),math.cos(t*math.pi/2)
    features=x+[t,a,s]
    first=list(map(math.tanh,linear(params,"base.first",features)))
    h=list(map(math.tanh,linear(params,"base.second",first)))
    if gain:
        projection=linear(params,"branch.hint",[(row-5)/2])
        hidden=[math.tanh(v+c) for v,c in zip(linear(params,"branch.first",features),projection)]
        branch=list(map(math.tanh,linear(params,"branch.second",hidden)))
        residual=linear(params,"branch.exit",branch)
        h=[v+gain*r for v,r in zip(h,residual)]
    return linear(params,"base.head",h)


def update(task,x,pred,t,dt):
    if task=="flow":
        return [v+dt*u for v,u in zip(x,pred)]
    angle=t*math.pi/2
    a,s=math.sin(angle),math.cos(angle)
    na,ns=math.sin((t+dt)*math.pi/2),math.cos((t+dt)*math.pi/2)
    # v 预测恢复干净值和噪声，再用下一个时刻的系数配比（确定性 DDIM）。
    return [na*(a*v-s*u)+ns*(s*v+a*u) for v,u in zip(x,pred)]


def start(seed):
    rng=random.Random(seed)
    return [rng.gauss(0,1) for _ in range(12)]


def rollout(params,task,noise,row,gain):
    x=list(noise)
    states=[x]
    for k in range(STEPS):
        t=k/STEPS
        x=update(task,x,predict(params,x,t,row,gain),t,1/STEPS)
        states.append(x)
    return states


def centers(x):
    return [(7+5*x[i],5+2*x[i+1]) for i in range(0,len(x),2)]


def cells(x):
    # 用最近像素绘图与计数，坐标不吸附到训练中的四条路线。
    return [(math.floor(col+.5),math.floor(row+.5)) for col,row in centers(x)]


def four(models,seed,row):
    noise=start(seed)
    return {task:[rollout(models[task],task,noise,row,g) for g in (0,1)] for task in TASKS}


def cohort(models,first_seed=FIRST_SEED,count=COUNT):
    """每份噪声分别指定行 3 和 7。整段六个预测行号全对，才给这道题记一分。"""
    scores={}
    for task in TASKS:
        tally=[0,0]
        for seed in range(first_seed,first_seed+count):
            noise=start(seed)
            off=cells(rollout(models[task],task,noise,3,0)[-1])
            for row in ROWS:
                on=cells(rollout(models[task],task,noise,row,1)[-1])
                tally[0]+=all(y==row for _,y in off)
                tally[1]+=all(y==row for _,y in on)
        scores[task]=tally
    return scores


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",type=Path,default=ROOT/"pair-lab/weights.json")
    parser.add_argument("--seed",type=int,default=FIRST_SEED)
    parser.add_argument("--row",type=int,choices=ROWS,default=3)
    parser.add_argument("--cohort",action="store_true",help="将 seed 作为起始种子，重算连续 64 份噪声")
    args=parser.parse_args()
    models=load_models(args.weights)
    if args.cohort:
        print("每列分母",2*COUNT,cohort(models,args.seed))
    else:
        for task,traces in four(models,args.seed,args.row).items():
            for gain,trace in enumerate(traces):
                print(task,"λ =",gain,"六帧中心",cells(trace[-1]))
