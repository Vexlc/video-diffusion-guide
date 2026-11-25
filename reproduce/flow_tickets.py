"""实验 K 延伸：用同一批独立端点核算窗口内的速度回归和过门人数。"""
import argparse
import base64
import json
import math
import struct
from pathlib import Path

from flow_coupling import CENTER, NORMAL, arrows, cdf, training_draws, variance


RECORD = struct.Struct("<ddB")
DEFAULT_COUNT = 120000
BINS = 16


def window_records(samples, x, t, radius, group=0):
    records = []
    for index,(z,y,sign) in enumerate(samples,1):
        position=(1-t)*z+t*y
        if abs(position-x)<=radius and (group==0 or group==sign):
            records.append((index,z,y,sign,position,y-z))
    return records


def summary(records):
    counts=[[0,0] for _ in range(BINS)]
    if not records:
        return {"count":0,"mean":None,"variance":None,"bins":counts}
    labels=[r[-1] for r in records]
    avg=sum(labels)/len(labels)
    for record in records:
        i=max(0,min(BINS-1,math.floor(record[-1])+8))
        counts[i][int(record[3]>0)]+=1
    return {"count":len(labels),"mean":avg,"variance":sum((u-avg)**2 for u in labels)/len(labels),
            "bins":counts}


def density(x,t):
    var=variance(t)
    return sum(math.exp(-(x-t*m)**2/(2*var)) for m in (-CENTER,CENTER))/(2*math.sqrt(2*math.pi*var))


def gate(samples, x, t, half_span):
    early,late=max(0,t-half_span),min(1,t+half_span)
    # q = Φ(z) 沿精确一维 ODE 保持不变。q < F_t(x) 当且仅当轨迹位于门左侧。
    def cutoff(time):
        prob=cdf(x,time)
        if prob==0: return -math.inf
        if prob==1: return math.inf
        return NORMAL.inv_cdf(prob)

    cutoff_early,cutoff_late=cutoff(early),cutoff(late)
    before=after=right=left=ode_before=ode_after=0
    for z,y,_ in samples:
        start=(1-early)*z+early*y < x
        end=(1-late)*z+late*y < x
        ode_before+=z<cutoff_early
        ode_after+=z<cutoff_late
        before+=start
        after+=end
        right+=start and not end
        left+=not start and end
    return {"early":early,"late":late,"before":before,"after":after,"right":right,"left":left,
            "ode_before":ode_before,"ode_after":ode_after,"ode_right":max(0,ode_before-ode_after),
            "ode_left":max(0,ode_after-ode_before),"net":right-left,"empirical_rate":(right-left)/(len(samples)*(late-early)),
            "exact_rate":(cdf(x,early)-cdf(x,late))/(late-early),
            "instantaneous_rate":density(x,t)*arrows(x,t)["v"]}


def inspect(samples,x,t,radius,group,half_span):
    records=window_records(samples,x,t,radius,group)
    result=summary(records)
    return {"location":x,"time":t,"radius":radius,"total":len(samples),"group":group,
            "window":result,"first":records[:1],"theory":arrows(x,t),
            "gate":gate(samples,x,t,half_span)}


def run(seed,count,output):
    from ticket_figures import ticket_page,gate_page,intersect
    output.mkdir(parents=True,exist_ok=True)
    samples=list(training_draws(seed,count))
    rows=window_records(samples,.25,.5,.025)
    totals=summary(rows)
    info=gate(samples,.25,.5,.025)
    for name,content in [("training-crossing",intersect()),("ticket-loss",ticket_page(rows,totals)),
                         ("mass-gate",gate_page(info,count))]:
        (output/(name+".svg")).write_text(content,encoding="utf-8")
    packet=b"".join(RECORD.pack(z,y,int(sign>0)) for z,y,sign in samples)
    template=Path(__file__).with_name("ticket_ui.html").read_text(encoding="utf-8")
    template=template.replace("__MATH__",Path(__file__).with_name("ticket_math.js").read_text(encoding="utf-8"))
    html=template.replace("__TICKETS__",base64.b64encode(packet).decode("ascii")).replace("__SEED__",str(seed))
    (output/"training-tickets.html").write_text(html,encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=41)
    parser.add_argument("--total",type=int,default=DEFAULT_COUNT)
    parser.add_argument("--x",type=float,default=.25)
    parser.add_argument("--time",type=float,default=.5)
    parser.add_argument("--radius",type=float,default=.025)
    parser.add_argument("--half-span",type=float,default=.025)
    parser.add_argument("--group",type=int,choices=[-1,0,1],default=0)
    parser.add_argument("--out",type=Path,help="按本种子和题数重画网页、印图；起始视图统一使用讲义的参数")
    args=parser.parse_args()
    if not (args.total>=1 and math.isfinite(args.x) and 0<=args.time<=1 and
            math.isfinite(args.radius) and args.radius>0 and 0<args.half_span<=1):
        parser.error("题数和窗口宽度须为正，位置须有限，时间在 [0,1]，时间半宽在 (0,1]")
    if args.out:
        run(args.seed,args.total,args.out)
    samples=list(training_draws(args.seed,args.total))
    result=inspect(samples,args.x,args.time,args.radius,args.group,args.half_span)
    print(json.dumps(result,indent=2,ensure_ascii=False))
