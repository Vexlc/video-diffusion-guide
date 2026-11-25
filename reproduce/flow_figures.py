"""用实验 K 的公式和实测数据排出 PDF 插图与离线交互图。"""

import json
from pathlib import Path

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, circle, line, rect, svg, text
from flow_coupling import arrows
from workshop_figures import path_line


def speed_panel(x, t):
    a = arrows(x,t)
    parts = [text(12,25,f"同一题面　位置 x = {x:g}，进度 t = {t:g}",17,TEAL,bold=True),
             text(12,55,"这道题来自右团的可能性",14,MUTED),
             rect(205,41,300,19,RUST,3),rect(205+300*(1-a["p"]),41,300*a["p"],19,TEAL,2),
             text(519,56,f'{100*a["p"]:.1f}% 右团',15,TEAL,bold=True)]
    middle, scale = 353, 32
    for row,(key,caption,color) in enumerate([("left","左团的平均标签",RUST),("right","右团的平均标签",TEAL),("v","混合后的指路速度",INK)]):
        top = 103+45*row
        end = middle + scale*a[key]
        parts += [text(12,top+5,caption,15,color,bold=row==2),line(162,top,545,top,GRID),
                  line(middle,top-9,middle,top+9,MUTED),line(middle,top,end,top,color,3)]
        if abs(end-middle)<1:
            parts.append(circle(middle,top,4,color))
        else:
            direction = 1 if end>middle else -1
            parts.append(path_line([(end-6*direction,top-4),(end,top),(end-6*direction,top+4)],color,2))
        parts.append(text(560,top+5,f'{a[key]:+.3f}',16,color,bold=True))
    parts += [text(161,230,"← 速度为负",13,MUTED),text(middle,230,"0",13,MUTED,"middle"),
              text(546,230,"速度为正 →",13,MUTED,"end")]
    return svg("两个分量的平均训练标签与最终条件平均",246,parts)


def axes(parts,left,width,top,height,limit):
    for value in (-2,0,2):
        y = top + (limit-value)*height/(2*limit)
        parts += [line(left,y,left+width,y),text(left-6,y+5,str(value),12,MUTED,"end")]
    for t in (0,.5,1):
        parts.append(text(left+t*width,top+height+17,f"{t:g}",12,MUTED,"middle"))
    def xy(t,value):
        return left+width*t, top+(limit-value)*height/(2*limit)
    return xy


def tracks_panel(data):
    parts = [text(12,26,"沿着八个分位走",17,TEAL,bold=True),text(345,26,"单看第 75 百分位",17,TEAL,bold=True)]
    left=axes(parts,39,258,54,189,3)
    right=axes(parts,367,253,54,189,3)
    for item in data["tracks"]:
        color = RUST if item["q"]==.75 else TEAL
        parts.append(path_line([left(i/64,x) for i,x in enumerate(item["exact"])],color,3 if color==RUST else 1.6))
    focus = next(item for item in data["tracks"] if item["q"]==.75)
    parts.append(path_line([right(i/64,x) for i,x in enumerate(focus["exact"])],TEAL,3))
    for n,color,dash in [(1,RUST,"6 3"),(4,INK,"3 3")]:
        points = [right(i/n,x) for i,x in enumerate(focus["paths"][str(n)])]
        parts.append(path_line(points,color,2,dash))
        parts.extend(circle(x,y,3,color) for x,y in points)
    parts += [text(20,288,"横轴是生成进度 t，纵轴是位置 x。",14,MUTED),
              text(345,288,"青线 精确分位　橙虚线 1 步　黑点 4 步",13,INK)]
    return svg("分位曲线与一步、四步欧拉的实际轨迹",305,parts)


def pairing_panel(data):
    tracks = data["tracks"]
    parts = [text(16,28,"打乱终点，出题线会相交",17,RUST,bold=True), text(344,28,"按分位重新配对后拉直",17,TEAL,bold=True)]
    for col,order in enumerate(([6,0,7,1,5,2,3,4], range(len(tracks)))):
        xy = axes(parts,38+325*col,250,52,193,3)
        for j,k in enumerate(order):
            start = xy(0,tracks[j]["exact"][0])
            end = xy(1,tracks[k]["exact"][-1])
            color = RUST if j==5 else TEAL
            parts += [line(*start,*end,color,3 if j==5 else 1.4),circle(*start,3,color),circle(*end,3,color)]
    parts += [text(15,291,"两幅图用了完全相同的八个起点、八个终点。橙线追踪同一个起点。",14)]
    return svg("只改变配对关系的八组端点",309,parts)


def errors_panel(data):
    parts = [text(14,26,"学生每次只查一次表，两组学生用相同的训练端点",17,TEAL,bold=True)]
    for j,(key,title,color) in enumerate([("paired_error","保留老师配对",TEAL),("shuffled_error","打乱老师配对",RUST)]):
        x = 136+255*j
        parts.append(text(x,57,title,15,color,bold=True))
        for i,run in enumerate(data["runs"]):
            top = 75+37*i
            value = run[key]
            parts += [rect(x,top,166,15,LIGHT,2),rect(x,top,166*min(1,value/2.25),15,color,2),
                      text(x+172,top+13,f"{value:.3f}",14,color,bold=True)]
            if j==0:
                parts.append(text(13,top+13,f'老师 {run["steps"]} 步',14))
    parts += [text(14,245,"条长 = 到分位参考的平均绝对距离，越短越好。各栏共用 0 到 2.25 的尺度。",14,MUTED)]
    return svg("教师步数与一步学生在固定分位上的终点偏差",261,parts)


def render(data,output):
    from flow_tickets import DEFAULT_COUNT, run as render_tickets
    render_tickets(data["seed"],DEFAULT_COUNT,output)
    figures = {"arrows":speed_panel(.25,.5),"tracks":tracks_panel(data),
               "pairing":pairing_panel(data),"errors":errors_panel(data)}
    for name,figure in figures.items():
        (output/(name+".svg")).write_text(figure,encoding="utf-8")
    panels = [[speed_panel(j/4,i/4) for j in range(-4,5)] for i in range(5)]
    payload = {key:data[key] for key in ("ranks","starts","reference","runs","tracks")}
    probes = [[arrows(j/4,i/4) for j in range(-4,5)] for i in range(5)]
    payload.update({"panels":panels,"probes":probes,"seed":data["seed"],"examples":data["examples"]})
    template = Path(__file__).with_name("flow_ui.html").read_text(encoding="utf-8")
    (output/"index.html").write_text(template.replace("__FLOW_DATA__",json.dumps(payload,ensure_ascii=False,
        separators=(",",":"))),encoding="utf-8")
