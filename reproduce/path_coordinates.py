"""把角度参数化的扩散状态和 v 换成直线混合的状态与速度。只用 Python 标准库。"""
import argparse
import math
from pathlib import Path

from clock_report import Recording


def coefficients(tau):
    """直线两系数为 tau、1-tau。除以它们的长度，得到单位圆上的对应系数。"""
    if not 0 <= tau <= 1:
        raise ValueError("生成进度须在 0 到 1 之间")
    radius = math.hypot(tau, 1-tau)
    return tau/radius, (1-tau)/radius


def line_coordinates(x, v, signal, sigma):
    """x、v 共用同一次圆弧模型调用，可为标量或整段视频张量。"""
    total = signal + sigma
    tau = signal / total
    state = x / total
    speed = (signal - sigma)*x - total*v
    return tau, state, speed



def known_pair(clean, noise, signal):
    """给定训练题的两端，独立用端点差检验由圆弧 v 换算出的直线速度。"""
    if not 0 <= signal <= 1 or not all(map(math.isfinite,(clean,noise,signal))):
        raise ValueError("端点须为有限数，干净系数须在 0 到 1 之间")
    from prediction_lab import exercise
    sigma = math.sqrt(1-signal**2)
    x, labels = exercise(clean,noise,signal)
    v = labels["v_prediction"]
    tau, state, speed = line_coordinates(x,v,signal,sigma)
    return {"x":x,"v":v,"tau":tau,"line":state,"converted":speed,"endpoint_difference":clean-noise}


def next_step(record, case, step, frame, row, col):
    """从原版 DDIM 真正保存的一格数据出发，按匹配的直线网格重做一次更新。"""
    if not 0 <= step < record.steps:
        raise ValueError("请选择仍有下一步的记录")
    read = record.pixel(case, step, frame, row, col)
    tau, state, speed = line_coordinates(read["state"], read["v"], read["a"], read["sigma"])
    next_total = read["next_a"] + read["next_sigma"]
    next_tau = read["next_a"]/next_total
    delta = next_tau-tau
    return {**read, "tau":tau, "line":state, "speed":speed, "delta":delta, "next_tau":next_tau,
            "euler":state+delta*speed, "from_ddim":read["recorded_next"]/next_total,
            "raw_v":state+delta*read["v"], "sign_only":state-delta*read["v"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=Path(__file__).resolve().parents[1]/"pixel-lab/clock-data.js")
    parser.add_argument("--pair", type=float, nargs=2, metavar=("CLEAN","NOISE"), help="用已知训练端点做手算")
    parser.add_argument("--signal", type=float, default=.8, help="手算训练题的干净系数")
    for key, default in (("case",0),("step",12),("frame",2),("row",5),("col",7)):
        parser.add_argument("--"+key, type=int, default=default)
    args = parser.parse_args()
    try:
        if args.pair:
            result = known_pair(*args.pair,args.signal)
            keys = result.keys()
        else:
            result = next_step(Recording(args.record), args.case, args.step, args.frame, args.row, args.col)
            keys = ("state","v","tau","line","speed","next_tau","euler","from_ddim","raw_v","sign_only")
    except ValueError as error:
        parser.error(str(error))
    for key in keys:
        print(f"{key:12} {result[key]:+.9f}")
