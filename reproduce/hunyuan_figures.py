"""将源码导读的形状、引导输入和真实计算轨迹画成离线图。"""

import json
import math
from pathlib import Path

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, circle, line, rect, svg, text
from hunyuan_trace import SOURCE, address, guidance_features


def token_grid(data):
    pieces = [text(15,28,"17 帧 · 64 × 96　｜　展开第 2 个潜时间索引（从 0 数）",17,TEAL,bold=True)]
    grid = data["small"]["grid"]
    for r in range(4):
        for c in range(6):
            token = 2*24+r*6+c
            x,y = 18+49*c,51+38*r
            pieces.extend([rect(x,y,45,32,TEAL if token==57 else LIGHT,3),
                           text(x+22,y+22,token,14,"white" if token==57 else INK,"middle",True)])
    pieces += [text(343,73,"块 57 · 每通道 2 × 2 格",16,TEAL,bold=True)]
    for offset in range(4):
        comp = 20+offset
        _, coord = address(grid,57,comp)
        y,x = 89+43*(offset//2),345+134*(offset%2)
        pieces += [rect(x,y,126,38,"#faecd9" if offset==2 else LIGHT,4),
                   text(x+63,y+24,f"分量 {comp} · ({coord[-2]}, {coord[-1]})",14,RUST if offset==2 else INK,"middle")]
    pieces += [text(345,196,"固定潜通道 c = 5，括号为潜格行、列",13,MUTED),
               text(18,236,"格内数字是图像 token 序号。这里每个时间索引有 4 × 6 = 24 个块。",14,MUTED)]
    return svg("第57号视频块连到潜空间中的哪些位置",253,pieces)


def embeds():
    f = guidance_features(6)
    pieces = [text(15,26,"嵌入式引导 g = 6　→　送入编码器的数为 6000",17,TEAL,bold=True),
              text(15,51,"画出 128 对频率中的 4 对。横坐标为余弦，纵坐标为正弦。",14,MUTED)]
    for i,j in enumerate((0,32,64,96)):
        cx,cy=85+162*i,116
        x,y = cx+33*f[j],cy-33*f[128+j]
        pieces += [f'<circle cx="{cx}" cy="{cy}" r="33" fill="none" stroke="{GRID}" stroke-width="2"/>',
                   line(cx-38,cy,cx+38,cy),line(cx,cy-38,cx,cy+38),
                   line(cx,cy,x,y,TEAL,2.5),circle(x,y,4.6,RUST),
                   text(cx,174,f"频率索引 {j}",14,INK,"middle",True),
                   text(cx,194,f"({f[j]:.3f}, {f[j+128]:.3f})",14,MUTED,"middle")]
    return svg("引导系数进入可训练层前的四对正余弦特征",213,pieces)


def clock(data):
    pieces=[text(16,27,"四次问网络，五个 σ 边界",17,TEAL,bold=True),
            text(155,57,"噪声端 σ = 1",14,MUTED),text(633,57,"数据端 σ = 0",14,MUTED,"end")]
    for idx,g in enumerate((1,3,9)):
        top = 96+62*idx
        sigmas = data["runs"]["4"][str(g)]["sigma"]
        pieces += [text(14,top+4,f"shift = {g}",16,INK,bold=True),line(155,top,633,top,GRID,3)]
        for i,s in enumerate(sigmas):
            x=155+478*(1-s)
            pieces += [circle(x,top,5,TEAL if i<4 else RUST)]
        pieces.append(text(155,top+24,"   →   ".join(f"{s:.3f}" for s in sigmas),14,MUTED))
    pieces.append(text(16,275,"青点会发起网络查询，橙点是返回的终点。点的间距表示一次更新跨过的 σ 范围。",14,MUTED))
    return svg("同样四次网络调用，三个shift怎样移动采样点",293,pieces)


def curve(data):
    pieces = [text(13,28,"一份起始噪声，跟到第 75 百分位的终点",17,TEAL,bold=True),
              text(13,51,"横轴为生成进度 τ，纵轴为实验 K 中粒子的位置。",14,MUTED)]
    left,right,top,bottom=58,630,72,218
    def project(t,x):
        return left+(right-left)*t,bottom-(bottom-top)*x/2.2
    for y in (0,1,2):
        _, py=project(0,y)
        pieces += [line(left,py,right,py,GRID,1,"3 3"),text(left-7,py+4,str(y),13,MUTED,"end")]
    for t in (0,.25,.5,.75,1):
        px,_=project(t,0)
        pieces.append(text(px,bottom+18,f"{t:g}",13,MUTED,"middle"))
    def add_path(coords,color,dash,markers):
        points=" ".join(f"{x:.2f},{y:.2f}" for x,y in coords)
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6" stroke-dasharray="{dash or "none"}"/>')
        if markers:
            pieces.extend(circle(x,y,4,color) for x,y in coords)
    rank=data["ranks"].index(.75)
    add_path([project(i/64,v) for i,v in enumerate(data["exact"][rank])],INK,"3 3",False)
    for g,color in [(1,TEAL),(9,RUST)]:
        row=data["runs"]["4"][str(g)]
        add_path([project(1-s,x) for s,x in zip(row["sigma"],row["traces"][rank]["x"])],color,"",True)
    for x,name,color,dash in [(20,"分位参考",INK,"3 3"),(222,"4 步 · shift 1",TEAL,""),(440,"4 步 · shift 9",RUST,"")]:
        pieces += [line(x,260,x+29,260,color,2.6,dash),text(x+37,265,name,14,color)]
    return svg("两种四步日程与精确分位轨迹的对照",282,pieces)


def render(data, out):
    figures={"tokens":token_grid(data),"guidance":embeds(),"clock":clock(data),"curve":curve(data)}
    for name,fig in figures.items():
        (out/(name+".svg")).write_text(fig,encoding="utf-8")
    template=Path(__file__).with_name("hunyuan_ui.html").read_text(encoding="utf-8")
    html=template.replace("__STUDY__",json.dumps(data,separators=(",",":"))).replace("__SOURCE__",SOURCE)
    (out/"index.html").write_text(html,encoding="utf-8")
