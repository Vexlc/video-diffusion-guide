"""学习笔记本的四道可复算题。坐标从零数，纯 Python，无需权重。"""
import argparse
from math import isfinite, sqrt, tanh
from random import Random


def square(frame, row):
    """六帧小片里的一张灰度图，方块每帧向右移一列。"""
    center = 4 + frame
    return [[0.8 if abs(x-center)<=1 and abs(y-row)<=1 else -0.55
             for x in range(16)] for y in range(12)]


def noise_problem(noise_error):
    original = square(frame=2, row=3)
    rng = Random(17)
    noise = [[rng.gauss(0,1) for x in range(16)] for y in range(12)]
    a = 0.8
    sigma = sqrt(1-a*a)
    mixed = [[a*y + sigma*z for y,z in zip(pixels,snow)]
             for pixels,snow in zip(original,noise)]
    # 已知这次出题用的噪声，给每一格的猜测故意加同样的误差。
    restored = [[(x-sigma*(z+noise_error))/a for x,z in zip(pixels,snow)]
                for pixels,snow in zip(mixed,noise)]
    return original, mixed, restored


def lane_speed(x, t):
    # y 取 -1 或 +1，各占一半；z 是独立标准高斯噪声。
    mean_y = tanh(t*x/(1-t)**2)
    return (mean_y-x)/(1-t)


def lane_path(z, steps):
    if steps < 1:
        raise ValueError("至少走一步")
    path = [z]
    for k in range(steps):
        t = k/steps
        x = path[-1]
        path.append(x + lane_speed(x,t)/steps)
    return path


def video_lanes(seed):
    rng = Random(seed)
    rows = [rng.choice((3,7)) for frame in range(6)]
    # 共享首帧的抽签结果，其余帧保持同一行。
    shared = [rows[0]]*6
    legal = 2                      # 全在第 3 行，或全在第 7 行
    possible = 2**len(rows)         # 每一帧独立二选一
    return rows, shared, legal, possible


def train_exit(updates, strength):
    if updates < 0:
        raise ValueError("训练次数不能为负")
    weight = 0.0
    history = [weight]
    for step in range(updates):
        # 冻结主干 h=0。训练用强度 1，两个条件同样常见，目标值就是条件 c。
        gradient = sum((weight*c-c)*c for c in (-1,1))/2
        weight -= 0.25*gradient
        history.append(weight)
    # 归一化坐标 -1、+1 分别对应第 3、7 行。
    return history, [5+2*strength*weight*c for c in (-1,1)]


def lesson_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--steps", type=int, default=12, help="流匹配采样的欧拉步数，默认 12")
    parser.add_argument("--start", type=float, default=.4, help="跟踪的一格起始噪声，默认 0.4")
    parser.add_argument("--noise-error", type=float, default=.15, help="噪声猜测误差，默认 0.15")
    parser.add_argument("--seed", type=int, default=17, help="逐帧抽行的随机种子，默认 17")
    parser.add_argument("--updates", type=int, default=8, help="控制出口的训练次数，默认 8")
    parser.add_argument("--strength", type=float, default=1, help="训练完成后的控制强度，默认 1")
    return parser


def read_settings(parser):
    args = parser.parse_args()
    if args.steps < 1 or args.updates < 0:
        parser.error("采样步数应为正，训练次数至少为零")
    if not all(isfinite(value) for value in (args.start, args.noise_error, args.strength)):
        parser.error("起始噪声、猜测误差和控制强度须为有限数")
    return args


if __name__=="__main__":
    args = read_settings(lesson_parser(__doc__))
    print("扩散 · 原始、带噪、还原亮度",*[round(a[3][6],5) for a in noise_problem(args.noise_error)])
    print("流匹配 · 起点、一步终点、多步终点",args.start,lane_path(args.start,1)[-1],round(lane_path(args.start,args.steps)[-1],6))
    print("视频 · 独立抽样、整段共享、合法数、全部组合",*video_lanes(args.seed))
    history,positions=train_exit(args.updates,args.strength)
    print("控制 · 训练后的权重",round(history[-1],5),"两个预测行",*[round(v,4) for v in positions])
