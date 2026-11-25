"""为第 8、12 页画出从两种来源到一个观测的关系、局部对数密度与斜率。"""
import math
from pathlib import Path

from draw import GRID, INK, MUTED, RUST, TEAL, circle, grayscale_uri, line, rect, svg, text
from posterior_lab import BACKGROUND, BODY, estimate, noise_for
from reproduce import clean_pixel


def two_sources():
    image=grayscale_uri([clean_pixel(r,c) for r in range(30) for c in range(48)])
    parts=[text(16,20,"可能的干净来源",14,MUTED),text(216,20,"得到同一读数的两种配方",14,MUTED),
           text(478,20,"看到 0.04 后的权重",14,MUTED)]
    body=estimate(.04)["body"]
    for i,(name,row,y,p,color) in enumerate((("车身",15,BODY,body,TEAL),("背景",5,BACKGROUND,1-body,RUST))):
        top=31+64*i
        parts.append(f'<image x="16" y="{top}" width="80" height="50" href="{image}" style="image-rendering:pixelated"/>')
        parts.append(f'<rect x="{16+24*80/48-2}" y="{top+row*50/30-2}" width="6" height="6" fill="none" stroke="#ff982e" stroke-width="2"/>')
        parts.extend((text(109,top+20,name,16,color,bold=True),text(109,top+41,f"y = {y:+.2f}",14),
                      text(216,top+30,f"0.8 × ({y:+.2f}) + 0.6 × ({noise_for(.04,y):+.1f})",15),
                      text(438,top+30,"→",18,MUTED),rect(478,top+29,144,9,"#dae2dc"),
                      rect(478,top+29,144*p,9,color),text(478,top+21,f"{p:.1%}",16,color,bold=True)))
    return svg("车身和背景两种干净值都能产生0.04，等先验下两种解释的概率",151,parts)


def score_curves():
    parts=[]
    left_edge,right_edge=-1,1.2
    for panel,neighbor in enumerate((.64,-.44)):
        color=(TEAL,RUST)[panel]
        left=45+321*panel
        right,top,bottom=left+252,40,131
        def px(x):
            return left+(x-left_edge)/(right_edge-left_edge)*(right-left)
        def py(v):
            return bottom+(v+3)/3*(top-bottom)
        def log_p(x):
            return math.log(estimate(x,"shared",neighbor)["density"])
        score=estimate(.04,"shared",neighbor)["score"]
        parts+=[text(left,20,f"邻帧 {neighbor:+.2f}",16,color,bold=True),
                text(right,20,f"斜率 {score:+.3f}",15,INK,"end",True)]
        for y in (-3,-2,-1,0):
            parts+=[line(left,py(y),right,py(y),GRID,1,"3 3"),text(left-7,py(y)+4,str(y),12,MUTED,"end")]
        parts.append(line(left,bottom,right,bottom,MUTED,1.4))
        for x in (-1,.04,1):
            parts+=[line(px(x),bottom,px(x),bottom+4,MUTED),text(px(x),bottom+18,f"{x:g}",12,MUTED,"middle")]
        points=" ".join(f"{px(x):.2f},{py(log_p(x)):.2f}" for x in (left_edge+i*(right_edge-left_edge)/100 for i in range(101)))
        ref=log_p(.04)
        parts.extend((f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6"/>',
                      line(px(.04),py(ref),px(.04),bottom,MUTED,1,"3 3"),
                      line(px(-.23),py(ref-.27*score),px(.31),py(ref+.27*score),INK,2.8),
                      circle(px(.04),py(ref),3.8,INK)))
    parts.append(text(330,172,"横轴　当前格的带噪读数 x　　　纵轴　固定邻帧后的 log p(x | 邻帧)",14,MUTED,"middle"))
    return svg("同一读数0.04在两种邻帧条件下的对数密度，黑短线显示局部斜率",182,parts)


def render(output):
    output.mkdir(parents=True,exist_ok=True)
    scene=two_sources()
    (output/"two-sources.svg").write_text(scene,encoding="utf-8")
    (output/"score-curves.svg").write_text(score_curves(),encoding="utf-8")
    source=Path(__file__).parent
    screen=(source/"posterior_ui.html").read_text(encoding="utf-8")
    screen=screen.replace("__SCENE__",scene).replace("__MATH__",(source/"posterior_math.js").read_text(encoding="utf-8"))
    if not (output/"index.html").is_file():
        screen=screen.replace('<a href="index.html">三种预测头</a>',"")
    (output/"posterior.html").write_text(screen,encoding="utf-8")
