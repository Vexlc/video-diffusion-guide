"""只用标准库复算第 19 页四组对照，以及第 78 页的配对实验。"""
import argparse
import json
from statistics import mean

from reproduce import integrate, run_noise, sample_video, schedule, train_control, use_control, velocity


def direction_changes(track):
    """依次看相邻帧的位移。小于等于百万分之一的位置变化当作静止。"""
    moves = [b-a for a,b in zip(track,track[1:]) if abs(b-a)>1e-6]
    return sum(previous*following<0 for previous,following in zip(moves,moves[1:]))


def road_trial(seed, weights):
    """同一段的真实位置、观测和权重都固定，只把输入路标推迟一帧。"""
    truth, marks, base, aligned, delayed = use_control(seed,weights)
    timely_error = [abs(pred-y) for pred,y in zip(aligned,truth)]
    late_error = [abs(pred-y) for pred,y in zip(delayed,truth)]
    return {"truth":truth, "marks":marks, "aligned":aligned, "delayed":delayed,
            "aligned_error":timely_error, "delayed_error":late_error,
            "difference":[late-ontime for late,ontime in zip(late_error,timely_error)],
            "aligned_mae":mean(timely_error), "delayed_mae":mean(late_error),
            "base_mae":mean(abs(pred-y) for pred,y in zip(base,truth))}


def lesson(card, seed):
    if card=="A":
        clean, perfect = run_noise(seed,bias=0)
        _, biased = run_noise(seed,bias=.15)
        witness = 15*48+24
        signal, sigma = schedule(.9)
        mixed = perfect[-1][1][witness]
        return {"truth":clean[witness],"mixed":mixed,"signal":signal,"sigma":sigma,
                "noise":(mixed-signal*clean[witness])/sigma,
                "before":perfect[-1][2][witness],"after":biased[-1][2][witness]}
    if card=="B":
        single = integrate(seed,1,1)
        many = integrate(seed,1,32)
        initial = integrate(seed,0,1)[0]
        speed = velocity(initial,0)[0]
        return {"start_x":initial[0],"initial_speed_x":speed,"first_small_x":initial[0]+speed/32,
                "total":len(many),"before":sum(abs(x)>1 for x,y in single),
                "after":sum(abs(x)>1 for x,y in many)}
    if card=="C":
        single, joint = (sample_video(seed,joint) for joint in (False,True))
        return {"before":direction_changes(single),"after":direction_changes(joint),
                "single":single,"joint":joint}
    if card=="D":
        result = road_trial(seed,train_control(80))
        return {"before":result["aligned_mae"],"after":result["delayed_mae"],
                "frame2":{"truth":result["truth"][2],"aligned":result["aligned"][2],"delayed":result["delayed"][2]},
                "per_frame_change":result["difference"]}
    raise ValueError("题号为 A、B、C 或 D")


def cohort():
    """预先取连续的种子 0 到 63。每段先作配对比较，再汇总 64 次结果。"""
    weights = train_control(80)
    trials = [road_trial(seed,weights) for seed in range(64)]
    differences = [r["delayed_mae"]-r["aligned_mae"] for r in trials]
    summary = {"segments":len(trials), "frames_per_segment":8,
               "aligned_mean":mean(r["aligned_mae"] for r in trials),
               "delayed_mean":mean(r["delayed_mae"] for r in trials),
               "late_worse":sum(d>1e-12 for d in differences),
               "late_better_seeds":[s for s,d in enumerate(differences) if d< -1e-12],
               "tied":sum(abs(d)<=1e-12 for d in differences)}
    return trials, summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card",choices=list("ABCD"),help="只运行一道题；默认依次运行四题")
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--cohort",action="store_true",help="计算全部 64 段，并打印所选种子的逐帧数据")
    args=parser.parse_args()
    if not 0<=args.seed<=999999:
        parser.error("种子范围为 0 至 999999")
    if args.cohort:
        if args.seed>=64 or args.card:
            parser.error("整批模式用种子 0 至 63，无需指定单题")
        trials,summary=cohort()
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print("种子",args.seed,json.dumps(trials[args.seed],ensure_ascii=False,indent=2))
    else:
        for key in args.card or "ABCD":
            print(key,"种子",args.seed,json.dumps(lesson(key,args.seed),ensure_ascii=False))
