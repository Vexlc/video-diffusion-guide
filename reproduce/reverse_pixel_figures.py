"""按真实计算值画去噪一步的结果与两种下一层分布。"""
import json
import random
from pathlib import Path

from draw import GRID, INK, MUTED, RUST, TEAL, line, rect, svg, text
from reverse_pixel import EDGES, WINDOW, cases, distributions, draws, step_parameters


def one_step():
    step=step_parameters(.04,.96)
    parts=[text(14,25,"同一个 x = 0.04，重走三次",17,TEAL,bold=True)]
    for i,z in enumerate((-1.,0.,1.)):
        parts.append(text(272+144*i,51,f"新噪声 ξ = {int(z):+d}",14,MUTED,"middle"))
    for row,(method,name) in enumerate((("ddpm","DDPM"),("ddim","DDIM（η = 0）"))):
        y=65+49*row
        parts.extend((text(14,y+17,name,16,INK,bold=True),text(14,y+35,("同一中心，再抽一次","固定起点，确定一步")[row],13,MUTED)))
        for i,z in enumerate((-1.,0.,1.)):
            val=draws(step,.25,z)[method]
            gray=round(max(0,min(1,(val+1)/2))*255)
            x=223+144*i
            parts.extend((rect(x,y,36,32,f"rgb({gray},{gray},{gray})",2),
                          text(x+43,y+22,f"{val:+.4f}",16,(RUST,TEAL)[row],bold=True)))
    return svg("DDPM 与确定性 DDIM 从同一起点取下一格，三列只改变新噪声",161,parts)


def mass_panels():
    parts=[]
    for panel,target in enumerate((.96,1.)):
        step=step_parameters(.04,target)
        dist=distributions(step)
        x=43+327*panel
        top,bottom=60,203
        # 每个小图有自己的纵轴，保证中间层的双峰在纸面仍能读出。
        ceiling=.25 if panel==0 else 1.
        parts.extend((text(x-26,25,"走到下一层" if panel==0 else "直接走到干净层",18,TEAL,bold=True),
                      text(x-26,47,f"信号系数 {target:.2f}　·　每格宽 0.15",14,MUTED)))
        for frac in (0,.5,1):
            y=bottom-frac*(bottom-top)
            parts.extend((line(x,y,x+245,y,GRID,1,"3 3"),text(x-4,y+4,f"{100*ceiling*frac:g}%",11,MUTED,"end")))
        for i in range(16):
            for side,(key,color) in enumerate((("exact",TEAL),("ddpm",RUST))):
                h=(bottom-top)*dist[key][i+1]/ceiling
                parts.append(rect(x+i*15.3+side*6.7,bottom-h,6.7,h,color))
        for i,val in enumerate((-1.2,-.6,0,.6,1.2)):
            left=x+i*61.2
            parts.extend((line(left,bottom,left,bottom+4,MUTED),text(left,bottom+20,f"{val:g}",12,MUTED,"middle")))
    parts.extend((rect(49,240,17,10,TEAL),text(74,250,"精确反向分布",15),
                  rect(278,240,17,10,RUST),text(303,250,"单高斯 DDPM（固定方差）",15)))
    return svg("按像素亮度分组的反向概率，分别在中间层与干净层比较精确参照和单高斯",263,parts)


def render(output,x=.04,target=.96,count=65536,seed=17,uniform=.25,fresh=0.):
    output.mkdir(parents=True,exist_ok=True)
    (output/"sampler-draw.svg").write_text(one_step(),encoding="utf-8")
    (output/"reverse-bins.svg").write_text(mass_panels(),encoding="utf-8")
    xs=sorted({-.44,.04,.64,x})
    targets=sorted({.82,.90,.96,1.,target})
    rng=random.Random(seed+7711)
    payload={"xs":xs,"targets":targets,"x":x,"target":target,"seed":seed,"count":count,
             "edges":EDGES,"window":WINDOW,"initialDraw":[uniform,fresh],"cases":cases(xs,targets,count,seed),
             "tickets":[[rng.random(),rng.gauss(0,1)] for _ in range(256)]}
    template=Path(__file__).with_name("reverse_pixel_ui.html").read_text(encoding="utf-8")
    html=template.replace("__REVERSE_DATA__",json.dumps(payload,separators=(",",":"),allow_nan=False))
    related='<a href="posterior.html">回到第 8 页的来源权重</a>' if (output/"posterior.html").exists() else ""
    (output/"reverse-step.html").write_text(html.replace("__RELATED__",related),encoding="utf-8")
