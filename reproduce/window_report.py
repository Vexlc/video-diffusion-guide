"""从逐帧实测中心重算开窗成绩，画出四条时间带和早晚配对的路线图。仅用标准库。"""
import argparse
import json
from pathlib import Path

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, line, rect, svg, text
from pixel_report import bitmap

NAMES={"off":"全程关闭","early":"前段开启","late":"后段开启","full":"全程开启"}
COLORS={"both":"#206b62","early":"#a4492c","late":"#3874a0","neither":"#d5d8cc"}
CATEGORIES={"both":"两种都中","early":"仅前段中","late":"仅后段中","neither":"两种都偏"}


def judge(data):
    """每题的两条生成路线都与同一个目标逐帧配对。六帧全对才记为全中。"""
    rows=[]
    for i,target in enumerate(data["targets"]):
        counts={mode:sum(a==b for a,b in zip(run["centers"][i],target))
                for mode,run in data["runs"].items()}
        early,late=(counts[m]==len(target) for m in ("early","late"))
        category=("both" if late else "early") if early else ("late" if late else "neither")
        rows.append({"counts":counts,"category":category})
    return rows


def film_figure(data):
    rows=data["runs"]
    parts=[text(12,25,"横条数生成步，右边六格数影片帧",17,TEAL,bold=True)]
    for i in range(6):
        parts.append(text(204+81*i,48,f"帧 {i}",12,MUTED,"middle"))
    for i,(mode,name) in enumerate(NAMES.items()):
        item=rows[mode];top=57+65*i
        begin,end=item["window"]
        active=end-begin if data["strength"] else 0
        parts.append(text(12,top+17,name,15,INK,bold=True))
        for k in range(24):
            parts.append(rect(12+5.6*k,top+25,4.7,9,TEAL if active and begin<=k<end else "#d3dad2"))
        parts.append(text(12,top+49,f"{active} / 24 步用位置提示",11,MUTED))
        for f,frame in enumerate(item["film"]):
            x=170+81*f
            parts.append(bitmap(frame,x,top,68))
            if f==3:
                parts.append(f'<rect x="{x-1.5}" y="{top-1.5}" width="71" height="54" fill="none" stroke="{RUST}" stroke-width="2"/>')
    return svg("在固定噪声与首帧点的四条采样路径上切换控制窗口",321,parts)


def route_grid(data, rows):
    seeds=list(dict.fromkeys(data["seeds"]))
    width=len(rows)//len(seeds)
    parts=[text(12,26,"一格是一道题，颜色记录同一题的两次成绩",17,TEAL,bold=True)]
    left,top=69,70
    dx,dy=575/width,20
    for n,seed in enumerate(seeds):
        parts.append(text(58,top+dy*n+12,str(seed),12,MUTED,"end"))
        for c in range(width):
            color=COLORS[rows[n*width+c]["category"]]
            parts.append(rect(left+c*dx,top+dy*n,dx-1.4,dy-2,color,2))
    example=next((i for i,row in enumerate(rows) if row["category"]=="early"),None)
    if example is not None:
        r,c=divmod(example,width)
        x,y=left+c*dx,top+r*dy
        for shift,color in ((0,INK),(2,"#ffffff")):
            parts.append(f'<rect x="{x+shift}" y="{y+shift}" width="{dx-1.4-2*shift}" height="{dy-2-2*shift}" fill="none" stroke="{color}" stroke-width="1.4"/>')
    for c in range(0,width,4):
        parts.append(text(left+(c+.5)*dx,top-7,str(c+1),12,MUTED,"middle"))
    legend_top=top+dy*len(seeds)+20
    for i,(cat,name) in enumerate(CATEGORIES.items()):
        x=12+162*i
        count=sum(item["category"]==cat for item in rows)
        parts.extend((rect(x,legend_top,15,15,COLORS[cat],2),
                      text(x+21,legend_top+13,f"{name}　{count}",14,INK,bold=True)))
    parts.append(text(12,legend_top+43,"纵轴是起始噪声种子，横轴是留出路线编号。两种设置都完成全部 24 次更新。",14))
    height=legend_top+57
    if example is not None:
        goal=data["targets"][example]
        early=data["runs"]["early"]["centers"][example]
        late=data["runs"]["late"]["centers"][example]
        parts.append(text(12,height+17,"双框内这道题",15,TEAL,bold=True))
        parts.append(text(12,height+39,"目标／前段／后段",12,MUTED))
        for f,(p,a,b) in enumerate(zip(goal,early,late)):
            x=163+88*f
            parts.extend([text(x,height+17,f"帧 {f}",12,MUTED,"middle"),
                          text(x,height+39,f"{p[0]} / {a[0]} / {b[0]}",14,RUST if a!=b else INK,"middle",True)])
        parts.append(text(12,height+62,"上排色格比较六帧的完整坐标，小表放大这道题的列坐标。",13,MUTED))
        height+=73
    return svg("同题早开与晚开的逐段配对成绩",height,parts)


def render(data,out,*,course=False):
    rows=judge(data)
    out.mkdir(parents=True,exist_ok=True)
    (out/"window-films.svg").write_text(film_figure(data),encoding="utf-8")
    (out/"window-routes.svg").write_text(route_grid(data,rows),encoding="utf-8")
    template=Path(__file__).with_name("window_scores_ui.html").read_text(encoding="utf-8")
    nav='<a href="../index.html">离线首页</a> · <a href="first-film.html#when">在浏览器重新生成四排</a>' if course else ''
    # 独立网页内嵌逐帧中心；图表的图例也给出文字，颜色只用于辅助阅读。
    record={k:data[k] for k in ("width","steps","strength","targets","seeds","focus")}
    record["centers"]={key:r["centers"] for key,r in data["runs"].items()}
    record["course"]=course
    if not course:
        (out/"windows.json").write_text(json.dumps(data,separators=(",",":"),allow_nan=False),encoding="utf-8")
    record["windows"]={key:r["window"] for key,r in data["runs"].items()}
    (out/"window-scores.html").write_text(template.replace("__NAV__",nav).replace("__DATA__",json.dumps(record,separators=(",",":"))),
                                          encoding="utf-8")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--record",type=Path,default=Path(__file__).resolve().parents[1]/"flow-hint/windows.json")
    cli.add_argument("--case",type=int,help="从 1 数的题号，先按种子再按路线排列")
    cli.add_argument("--out",type=Path,help="从这份实测记录重画 HTML 和 SVG")
    args=cli.parse_args()
    data=json.loads(args.record.read_text(encoding="utf-8"))
    rows=judge(data)
    case=args.case if args.case is not None else min(7,len(rows))
    if not 1<=case<=len(rows):
        cli.error("题号超出录入范围")
    i=case-1
    print("整段配对", {CATEGORIES[k]:sum(q["category"]==k for q in rows) for k in CATEGORIES})
    print("题号",case,"种子",data["seeds"][i],CATEGORIES[rows[i]["category"]])
    print("目标",data["targets"][i])
    for mode,label in NAMES.items():
        print(label,data["runs"][mode]["centers"][i],rows[i]["counts"][mode],"/ 6")
    if args.out:
        render(data,args.out)
