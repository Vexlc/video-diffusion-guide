"""实验 N：在一个像素上对照三种扩散输出和两条生成路径。只用标准库。"""

import argparse
import json
import math
from pathlib import Path

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, circle, grayscale_uri, line, rect, svg, text
from reproduce import clean_pixel, normal_draw
import author_endpoints


HEADS = ("epsilon", "sample", "v_prediction")
NAMES = ("噪声 ε", "干净值 y", "扩散 v")


def exercise(clean, noise, signal):
    """相同的一对干净值和噪声，一次正向混合，三种可以拿来监督的标签。"""
    sigma = math.sqrt(1 - signal**2)
    state = signal*clean + sigma*noise
    return state, {"epsilon": noise, "sample": clean, "v_prediction": signal*noise-sigma*clean}


def read_prediction(state, output, kind, signal):
    """返回干净值和噪声估计。端点上不能由方程确定的分量记为 None。"""
    sigma = math.sqrt(1 - signal**2)
    if kind == "epsilon":
        return ((state-sigma*output)/signal if signal else None), output
    if kind == "sample":
        return output, ((state-signal*output)/sigma if sigma else None)
    if kind == "v_prediction":
        return signal*state-sigma*output, sigma*state+signal*output
    raise ValueError(f"未知的输出类型 {kind!r}")


def trace(clean, noise, time):
    """两条固定端点的条件路径。生成进度同为 0→1，统一按每单位进度求导。"""
    angle = (1-time)*math.pi/2
    a = 0.0 if time == 0 else math.cos(angle)
    b = math.sin(angle)
    velocity = a*noise-b*clean
    return {
        "arc": (a, b, a*clean+b*noise, -math.pi/2*velocity),
        "line": (time, 1-time, time*clean+(1-time)*noise, clean-noise),
    }


def scene(seed):
    draw = normal_draw(seed+34171)
    clean = [clean_pixel(r,c) for r in range(30) for c in range(48)]
    return clean, [draw() for _ in clean]


def car_row(clean, noise, signal):
    mixed, matched, mistaken = [], [], []
    for y, z in zip(clean, noise):
        x, targets = exercise(y, z, signal)
        v = targets["v_prediction"]
        mixed.append(x)
        matched.append(read_prediction(x,v,"v_prediction",signal)[0])
        mistaken.append(read_prediction(x,v,"epsilon",signal)[0])
    return (clean, mixed, matched, mistaken)


def ledger(clean, noise, signal):
    x, targets = exercise(clean, noise, signal)
    pieces = [text(16,30,f"同一个带噪输入 x = {x:.3f}，只换训练时交的答卷",18,TEAL,bold=True)]
    for xpos, title in [(18,"输出类型"),(181,"标签数值"),(311,"怎样读回干净值"),(566,"读回结果")]:
        pieces.append(text(xpos,67,title,15,MUTED))
    formulas = ["(x − σε̂) / a", "ŷ", "ax − σv̂"]
    for i, (kind, name, formula) in enumerate(zip(HEADS,NAMES,formulas)):
        top = 80+54*i
        result, _ = read_prediction(x,targets[kind],kind,signal)
        pieces += [rect(10,top,640,46,LIGHT,4),text(20,top+29,name,17),
                   text(188,top+29,f"{targets[kind]:+.3f}",17,TEAL,bold=True),
                   text(311,top+29,formula,18),text(570,top+29,f"{result:.3f}" if result is not None else "不定",17,RUST,bold=True)]
    return svg("三种监督标签及对应的干净值换算",247,pieces)


def contact_sheet(clean, noise):
    pieces = []
    for i, title in enumerate(["干净原图", "带噪输入", "按 v 读回", "误当成 ε 读回"]):
        pieces.append(text(119+151*i,26,title,15,INK,"middle",True))
    for row, signal in enumerate((.8, .2)):
        top = 40+124*row
        pieces.append(text(8,top+40,f"a={signal}",13,TEAL,bold=True))
        for col, pixels in enumerate(car_row(clean,noise,signal)):
            left = 49+151*col
            pieces.append(f'<image x="{left}" y="{top}" width="140" height="87.5" href="{grayscale_uri(pixels)}" style="image-rendering:pixelated"/>')
            if col >= 2:
                mae = sum(abs(y-yhat) for y,yhat in zip(clean,pixels))/len(clean)
                pieces.append(text(left+70,top+107,f"误差 {mae:.3f}",14,TEAL if col==2 else RUST,"middle"))
    return svg("两档信号水平用同一个真值 v 演算，比较正确与错误的换算",282,pieces)


def path_figure(clean, noise, time, *, compare_start=False):
    # 左图在系数平面中画路径；右图把同一对标量代入，横轴改为生成进度。
    pieces = [text(18,26,"两种系数的配方",17,TEAL,bold=True),text(350,26,"代入以后，这个像素的值",17,TEAL,bold=True)]
    values = [trace(clean,noise,i/100) for i in range(101)]
    extent = max(1, abs(clean)+abs(noise))
    planes = [(77,248,51,222),(374,636,51,222)]
    for panel,(left,right,top,bottom) in enumerate(planes):
        pieces += [line(left,top,left,bottom,MUTED,1.2),line(left,bottom,right,bottom,MUTED,1.2)]
        for ratio in [0,.5,1]:
            x = left+(right-left)*ratio
            pieces += [line(x,bottom,x,bottom+4,MUTED),text(x,bottom+19,f"{ratio:g}",13,MUTED,"middle")]
        y_values = [0,.5,1] if panel == 0 else [-extent,0,extent]
        for j,val in enumerate(y_values):
            y = bottom-j*(bottom-top)/2
            pieces += [line(left,y,right,y,GRID,1,"3 3"),text(left-6,y+4,f"{val:g}",12,MUTED,"end")]
        pieces += [text(left,top-11,"噪声系数" if panel==0 else "像素值 x",13,MUTED),
                   text((left+right)/2,bottom+39,"干净系数" if panel==0 else "生成进度 τ",14,MUTED,"middle")]
        for key,color,dash in [("line",RUST,"6 4"),("arc",TEAL,"")]:
            def xy(t, row):
                a,b,state,_ = row[key]
                vx,vy = (a,b) if panel==0 else (t, .5+.5*state/extent)
                return left+(right-left)*vx,bottom-(bottom-top)*vy
            coords = [xy(i/100,row) for i,row in enumerate(values)]
            points = " ".join(f"{x:.2f},{y:.2f}" for x,y in coords)
            pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6" stroke-dasharray="{dash or "none"}"/>')
            px,py = xy(time,trace(clean,noise,time))
            pieces.append(f'<circle id="mark-{panel}-{key}" cx="{px:.2f}" cy="{py:.2f}" r="{6 if key=="line" else 4}" fill="{color}" stroke="white" stroke-width="1"/>')
    pieces += [line(50,286,88,286,TEAL,3),text(96,291,"圆弧系数",15),
               line(242,286,280,286,RUST,3,"6 4"),text(287,291,"直线系数",15),
               text(454,291,"圆点标出所选进度",14,MUTED)]
    if compare_start:
        # 在同一根亮度尺上标出另一起点的两个中点，便于在纸上读出差距。
        other = trace(clean,1.,.5)
        pieces += [line(18,306,642,306),text(18,332,"改从 +1 出发",15,INK,bold=True),
                   text(18,355,"τ = ½ 时的像素值",13,MUTED),line(218,347,605,347,MUTED,1.5)]
        for val in (.8,1.,1.3):
            px = 218+(val-.7)*600
            pieces += [line(px,344,px,352,MUTED),text(px,371,f"{val:.1f}",13,MUTED,"middle")]
        for key,name,color in [("line","直线",RUST),("arc","圆弧",TEAL)]:
            value = other[key][2]
            px = 218+(value-.7)*600
            pieces += [circle(px,347,5.5,color),text(px,334,f"{name} {value:.3f}",14,color,"middle",True)]
    return svg("系数平面和单个像素随进度的实际变化",381 if compare_start else 307,pieces)


def run(seed, output):
    output.mkdir(parents=True,exist_ok=True)
    clean, noise = scene(seed)
    y,z,a,t = .8,-1.,.8,.5
    x,targets = exercise(y,z,a)
    row = {"seed":seed, "clean":y, "noise":z, "signal":a, "state":x, "targets":targets,
           "halfway":trace(y,z,t), "car_mae":{}}
    for signal in (.8,.2):
        wrong = car_row(clean,noise,signal)[-1]
        row["car_mae"][str(signal)] = sum(abs(v-w) for v,w in zip(clean,wrong))/len(clean)
    figures = {"heads":ledger(y,z,a), "car":contact_sheet(clean,noise),
               "paths":path_figure(y,z,t,compare_start=True)}
    for name, chart in figures.items():
        (output/(name+".svg")).write_text(chart,encoding="utf-8")
    (output/"numbers.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
    template = Path(__file__).with_name("prediction_ui.html").read_text(encoding="utf-8")
    reference = author_endpoints.load_record()
    author_endpoints.check(reference)
    endpoint_figure = author_endpoints.figure(reference)
    (output/"author-endpoints.svg").write_text(endpoint_figure,encoding="utf-8")
    template = (template.replace("__ENDPOINT_FIGURE__",endpoint_figure)
                .replace("__ENDPOINT_RECORD__",json.dumps(reference,separators=(",",":"),ensure_ascii=False))
                .replace("__AUTHOR_SOURCE__",author_endpoints.SOURCE))
    inputs = json.dumps({"clean":clean,"noise":noise},separators=(",",":"))
    html = template.replace("__CAR_DATA__",inputs).replace("__PATH_SVG__",path_figure(y,z,t))
    html = html.replace("__POSITIVE_SVG__",path_figure(y,1.,t))
    (output/"index.html").write_text(html,encoding="utf-8")
    from posterior_figures import render as render_posterior
    render_posterior(output)
    from reverse_pixel_figures import render as render_reverse_step
    render_reverse_step(output)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("replay/prediction-lab"))
    parser.add_argument("--seed",type=int,default=17,help="小车噪声种子")
    args = parser.parse_args()
    print(json.dumps(run(args.seed,args.out),ensure_ascii=False,indent=2))
