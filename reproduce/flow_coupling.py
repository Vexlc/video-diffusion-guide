"""实验 K：用一维双峰分布检查配对标签、生成轨迹和一步蒸馏。仅用标准库。"""

import argparse
import bisect
import json
import math
import random
from pathlib import Path
from statistics import NormalDist, mean


NORMAL = NormalDist()
CENTER, SPREAD = 2.0, 0.5
BUDGETS = (1, 4, 16, 64)
EVAL_COUNT = 256


def variance(t):
    return (1-t)**2 + (t*SPREAD)**2


def arrows(x, t):
    """先在一个高斯分量内取平均，再用后验概率混合左右两支。"""
    var = variance(t)
    slope = (t*SPREAD**2 - (1-t)) / var
    right = (1 + math.tanh(t*CENTER*x / var)) / 2
    left_speed = -CENTER + slope*(x + t*CENTER)
    right_speed = CENTER + slope*(x - t*CENTER)
    return {"p": right, "left": left_speed, "right": right_speed,
            "v": (1-right)*left_speed + right*right_speed}


def velocity(x, t):
    # 与 arrows 中加权平均相同的公式；先平均分量中心，避免反复构造字典。
    var = variance(t)
    center = CENTER * math.tanh(t*CENTER*x / var)
    return center + (t*SPREAD**2 - (1-t))*(x-t*center)/var


def cdf(x, t):
    scale = math.sqrt(variance(t))
    return (NORMAL.cdf((x+t*CENTER)/scale) + NORMAL.cdf((x-t*CENTER)/scale))/2


def quantile(q, t):
    if not 0 < q < 1:
        raise ValueError("分位必须在 0 与 1 之间")
    if t == 0:
        return NORMAL.inv_cdf(q)
    offset = NORMAL.inv_cdf(q)*math.sqrt(variance(t))
    low, high = offset-t*CENTER, offset+t*CENTER
    for _ in range(60):
        mid = (low+high)/2
        if cdf(mid, t) < q:
            low = mid
        else:
            high = mid
    return (low+high)/2


def euler(z, steps):
    if steps < 1:
        raise ValueError("步数至少为 1")
    path = [z]
    for step in range(steps):
        x = path[-1]
        path.append(x + velocity(x, step/steps)/steps)
    return path


def training_draws(seed, count):
    """以与实验 K 相同的顺序抽题；sign 是数据分量标签。"""
    rng = random.Random(seed+101)
    for _ in range(count):
        z = rng.gauss(0, 1)
        sign = rng.choice((-1, 1))
        y = sign*CENTER + SPREAD*rng.gauss(0, 1)
        yield z, y, sign


def local_check(seed, count=120000):
    """实际抽取独立端点，用半径 0.025 的窗口近似条件期望。"""
    probes = [0., .25]
    totals = {x: [0, 0, 0., 0.] for x in probes}
    for z, y, sign in training_draws(seed,count):
        middle = (z+y)/2
        for x in probes:
            if abs(middle-x) <= .025:
                tally = totals[x]
                tally[0] += 1
                tally[1] += sign>0
                tally[2] += y-z
                tally[3] += (y-z)**2
    return [{"x":x, "count":n, "right":r/n, "mean":s/n, "square":s2/n}
            for x,(n,r,s,s2) in totals.items()]


def fit_displacements(starts, ends, bins=96):
    """每个区间用平方损失拟合一个位移均值，将节点间线性连接。"""
    totals = [[0., 0., 0] for _ in range(bins)]
    for z,y in zip(starts, ends):
        index = min(bins-1, max(0, int((z+3.5)*bins/7)))
        cell = totals[index]
        cell[0] += z
        cell[1] += y-z
        cell[2] += 1
    anchors = [(z/n,d/n) for z,d,n in totals if n]
    if len(anchors)<2:
        raise ValueError("不同的训练位置不足")
    return list(zip(*anchors))


def student(z, table):
    knots, displacements = table
    high = min(len(knots)-1, max(1, bisect.bisect_right(knots,z)))
    low = high-1
    w = (z-knots[low])/(knots[high]-knots[low])
    return z + (1-w)*displacements[low] + w*displacements[high]


def mae(a,b):
    return mean(abs(x-y) for x,y in zip(a,b))


def experiment(seed, examples):
    rng = random.Random(seed)
    starts = [rng.gauss(0,1) for _ in range(examples)]
    permutation = list(range(examples))
    rng.shuffle(permutation)
    ranks = [(j+.5)/EVAL_COUNT for j in range(EVAL_COUNT)]
    z_eval = [quantile(q,0) for q in ranks]
    truth = [quantile(q,1) for q in ranks]
    runs = []
    for steps in BUDGETS:
        ends = [euler(z,steps)[-1] for z in starts]
        teacher = [euler(z,steps)[-1] for z in z_eval]
        paired = fit_displacements(starts,ends)
        shuffled = fit_displacements(starts,[ends[i] for i in permutation])
        match = [student(z,paired) for z in z_eval]
        mix = [student(z,shuffled) for z in z_eval]
        runs.append({"steps":steps, "teacher":teacher, "paired":match, "shuffled":mix,
                     "teacher_error":mae(teacher,truth), "paired_error":mae(match,truth),
                     "imitation_error":mae(match,teacher), "shuffled_error":mae(mix,truth)})
    tracks = []
    for q in (.05,.15,.25,.4,.6,.75,.85,.95):
        tracks.append({"q":q, "exact":[quantile(q,i/64) for i in range(65)],
                       "paths":{str(n):euler(quantile(q,0),n) for n in BUDGETS}})
    return {"seed":seed,"examples":examples,"ranks":ranks,"starts":z_eval,"reference":truth,
            "tracks":tracks,"runs":runs,"local":local_check(seed)}


def run(seed, examples, output):
    from flow_figures import render
    result = experiment(seed,examples)
    output.mkdir(parents=True,exist_ok=True)
    (output/"numbers.json").write_text(json.dumps(result,separators=(",",":")),encoding="utf-8")
    render(result,output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=41)
    parser.add_argument("--examples",type=int,default=16384)
    parser.add_argument("--out",type=Path,default=Path("replay/flow-coupling"))
    args = parser.parse_args()
    if args.examples<128:
        parser.error("请至少提供 128 对，让速度表覆盖多个位置")
    data = run(args.seed,args.examples,args.out)
    print("打开",args.out/"index.html")
    for row in data["runs"]:
        print(row["steps"],"步老师 / 一步学生 / 打乱配对学生相对分位参考的 MAE",
              *(round(row[key],5) for key in ("teacher_error","paired_error","shuffled_error")))
