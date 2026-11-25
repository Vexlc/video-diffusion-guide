"""用标准库从实际的求解器对照记录重画 PDF 插图和离线成绩单。"""
import argparse
import json
import math
from pathlib import Path

from draw import GRID, INK, MUTED, RUST, TEAL, circle, line, rect, svg, text
from pixel_report import bitmap

NAMES={"stop":"24 步网格","euler":"欧拉法","midpoint":"显式中点法"}
COLORS={"stop":"#65777a","euler":TEAL,"midpoint":RUST}


def rows_for(data,budget):
    return [row for row in data["rows"] if row["budget"]==budget]


def film_chart(rows):
    budget=rows[0]["budget"]
    parts=[text(12,24,f"同一份噪声，每排都问 {budget} 次速度",18,TEAL,bold=True)]
    for f in range(6):
        parts.append(text(183+84*f,50,f"帧 {f}",13,MUTED,"middle"))
    for i,r in enumerate(rows):
        top=61+96*i
        color=COLORS[r["mode"]]
        parts.extend((text(12,top+23,NAMES[r["mode"]],17,color,bold=True),
                      text(12,top+47,f'{r["updates"]} 次更新',14,MUTED)))
        for f,pic in enumerate(r["film"]):
            parts.append(bitmap(pic,150+84*f,top,66))
        y=top+66
        parts.extend((text(12,y+5,f'抵达 τ = {r["at"]:.3g}',14,INK),line(150,y,636,y,GRID,2),
                      line(150,y,150+486*r["at"],y,color,3)))
        for t,_,_ in r["queries"]:
            parts.append(circle(150+486*t,y,3.8,color))
        parts.append(rect(146+486*r["at"],y-4,8,8,color))
    parts.append(text(12,344,"每条尺从 0 到 1。圆点是向网络查询速度的进度，方块是最终抵达的进度。",14,MUTED))
    return svg("相同查询次数下实际解码的六帧、查询时刻与最终进度",356,parts)


def fork_chart(data):
    e,m=(next(r for r in rows_for(data,4) if r["mode"]==mode) for mode in ("euler","midpoint"))
    x0,trial,eu=e["path"][:3]
    mid=m["path"][1]
    low=min(x0,trial,eu,mid)-.01
    high=max(x0,trial,eu,mid)+.012
    def y(v):
        return 203-135*(v-low)/(high-low)
    parts=[text(12,26,"固定视频块 8、潜码分量 0，看前半程",17,TEAL,bold=True),
           text(12,53,"标准化潜码",14,MUTED),line(80,64,80,206,GRID,1.5),line(80,206,414,206,GRID,1.5)]
    for tick in (x0,trial,eu):
        parts.extend((text(72,y(tick)+4,f"{tick:.3f}",12,MUTED,"end"),line(80,y(tick),410,y(tick),GRID,.8)))
    xs=(86,246,406)
    for x,t in zip(xs,("0","¼","½")):
        parts.append(text(x,226,t,15,INK,"middle"))
    parts.extend((line(xs[0],y(x0),xs[2],y(mid),RUST,2.2,"5 3"),
                  line(xs[0],y(x0),xs[1],y(trial),TEAL,2.3),line(xs[1],y(trial),xs[2],y(eu),TEAL,2.3),
                  circle(xs[0],y(x0),4,INK),circle(xs[2],y(eu),4.5,TEAL),
                  rect(xs[2]-4,y(mid)-4,8,8,RUST),
                  f'<path d="M246 {y(trial)-7} l7 7 l-7 7 l-7 -7 Z" stroke="{RUST}" fill="white" stroke-width="2"/>',
                  circle(xs[1],y(trial),3,TEAL),
                  text(438,85,"τ = ½ 时落在哪里？",15,INK,bold=True),
                  text(438,111,f"欧拉   {eu:+.5f}",17,TEAL,bold=True),
                  text(438,137,f"中点   {mid:+.5f}",17,RUST,bold=True),
                  text(438,175,"菱形是两者共同的",14,MUTED),text(438,196,"第二次查询位置。",14,MUTED),
                  text(12,254,"横轴是生成进度 τ。线段只连接已接受的状态。中点法在 ¼ 处额外试探一次。",14,MUTED)))
    return svg("同一份速度场在半程出现两个数值终点及共同的第二次查询位置",267,parts)


def metric_chart(data):
    parts=[text(12,26,"相同权重、相同 224 段题，换两把尺评分",18,TEAL,bold=True)]
    error_max=max(r["latent_rmse"] for r in data["rows"] if r["latent_rmse"] is not None)
    error_top=max(.2,math.ceil(error_max*5)/5)
    score_min=min(r["strict"] for r in data["rows"] if r["mode"]!="stop")
    score_floor=min(180,20*(score_min//20))
    for i,(key,title,lo,hi,ticks) in enumerate((
        ("latent_rmse","距细网格终点的潜码误差",0,error_top,(0,error_top/2,error_top)),
        ("strict","六帧中心全部命中的段数",score_floor,224,(score_floor,(score_floor+224)//2,224)))):
        left=55+i*326
        def xy(b,v):
            return left+(b-2)/22*234,213-(v-lo)/(hi-lo)*121
        parts.append(text(left-33,66,title,15,INK,bold=True))
        for v in ticks:
            _,y=xy(2,v)
            parts.extend((line(left,y,left+240,y,GRID),text(left-7,y+4,f"{v:g}",12,MUTED,"end")))
        for b in data["budgets"]:
            x,_=xy(b,lo)
            parts.append(text(x,234,b,12,MUTED,"middle"))
        for mode in ("euler","midpoint"):
            coords=[xy(r["budget"],r[key]) for r in data["rows"] if r["mode"]==mode]
            for (x1,y1),(x2,y2) in zip(coords,coords[1:]):
                parts.append(line(x1,y1,x2,y2,COLORS[mode],2.2,"5 3" if mode=="midpoint" else ""))
            for x,y in coords:
                if mode=="midpoint":
                    parts.append(f'<rect x="{x-4}" y="{y-4}" width="8" height="8" fill="none" stroke="{RUST}" stroke-width="1.8"/>')
                else:
                    parts.append(circle(x,y,3.2,TEAL))
    parts.extend((text(13,265,"● 实线　欧拉法",14,TEAL,bold=True),text(159,265,"□ 虚线　显式中点法",14,RUST,bold=True),
                  text(389,265,"横轴均为每段的网络查询次数",14,MUTED)))
    return svg("查询预算从 2 到 24 时，终点误差和整段中心全中数量的变化",278,parts)


def render(data,out):
    out.mkdir(parents=True,exist_ok=True)
    for name,figure in (("solver-films",film_chart(rows_for(data,4))),("solver-fork",fork_chart(data)),
                        ("solver-metrics",metric_chart(data))):
        (out/(name+".svg")).write_text(figure,encoding="utf-8")
    template=Path(__file__).with_name("solver_ui.html").read_text(encoding="utf-8")
    results={k:data[k] for k in ("budgets","count","reference","paired")}
    results["rows"]=[{k:r[k] for k in ("mode","budget","strict","frame_hits","latent_rmse","at")} for r in data["rows"]]
    link='<a href="first-film.html#budget">用自己的起点重新生成三排影片</a>' if (out/"first-film.html").is_file() else ""
    template=template.replace("__GENERATOR__",link)
    (out/"solver-scores.html").write_text(template.replace("__RESULTS__",json.dumps(results,separators=(",",":"))),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--budget",type=int,choices=(2,4,8,12,24),default=4)
    p.add_argument("--render",action="store_true")
    args=p.parse_args()
    data=json.loads((args.directory/"solver.json").read_text())
    if args.render:
        render(data,args.directory)
    for row in rows_for(data,args.budget):
        print(NAMES[row["mode"]],"进度",row["at"],"六帧全中",row["strict"],"/",data["count"],
              "终点 RMSE",row["latent_rmse"])
        print("前两次查询 [进度, 潜码, 速度]",row["queries"][:2])
    print("配对结果",data["paired"][str(args.budget)],"细网格核对",data["reference"])
