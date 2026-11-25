"""用原始读数画出系数图和四排短片，并生成离线交互页面。"""
import hashlib
import json
import math
import shutil
from pathlib import Path

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, circle, line, rect, svg, text
from hint_report import signed, tile, values
from clock_report import Recording
from path_coordinates import line_coordinates, next_step
from pixel_report import bitmap
from prediction_lab import exercise

MODES = {"ddim":"原版 DDIM","matched":"换算 · 匹配刻度","raw":"直接拿 v 走","uniform":"换算 · 等距刻度"}


def geometry_diagram():
    y, z, a, b = .8, -1., .8, .6
    x, labels = exercise(y, z, a)
    v = labels["v_prediction"]
    tau, state, speed = line_coordinates(x, v, a, b)
    def point(px, py):
        return 63+195*px, 243-195*py
    def dot(key, px, py, fill):
        cx, cy = point(px,py)
        return f'<circle id="{key}" cx="{cx}" cy="{cy}" r="6" fill="{fill}" stroke="white" stroke-width="2"/>'
    curve = " ".join("%.3f,%.3f" % point(math.sin(i*math.pi/160),math.cos(i*math.pi/160)) for i in range(81))
    lx, ly = point(a,b)
    pieces = [text(12,23,"同一比例的两组混合系数",17,TEAL,bold=True),
              text(328,23,"给同一个像素配两次",17,TEAL,bold=True),
              line(63,243,270,243,MUTED,1.2),line(63,243,63,38,MUTED,1.2),
              text(69,40,"噪声系数",13,MUTED),text(163,265,"干净系数",13,MUTED,"middle"),
              text(51,243,"0",13,MUTED),text(50,53,"1",13,MUTED),text(257,260,"1",13,MUTED),
              f'<polyline points="{curve}" stroke="{TEAL}" fill="none" stroke-width="2.8"/>',
              line(63,48,258,243,RUST,2.6,"6 4"),
              f'<line id="ray" x1="63" y1="243" x2="{lx}" y2="{ly}" stroke="{MUTED}" stroke-width="1.6"/>',
              dot("lineDot",tau,1-tau,RUST),dot("arcDot",a,b,TEAL)]
    pairs = [(TEAL,"圆弧系数",(a,b),x,v,"v"),(RUST,"直线系数",(tau,1-tau),state,speed,"u")]
    for i,(color,title,(c,d),s,velocity,symbol) in enumerate(pairs):
        top=41+i*112
        pieces.extend((rect(315,top,333,102,LIGHT,5),text(330,top+23,title,16,color,bold=True)))
        for key,dy,value in (("coeff",49,f"干净 × {c:.3f}  +  噪声 × {d:.3f}"),
                             ("read",79,f"像素 {s:+.5f}　　 {symbol} = {velocity:+.3f}")):
            pieces.append(text(330,top+dy,value,15,INK).replace("<text ",f'<text id="{key}{i}" ',1))
    pieces.extend((circle(27,286,5,TEAL),text(41,291,"平方和为 1",14,INK),
                   circle(170,286,5,RUST),text(184,291,"两数之和为 1",14,INK)))
    return svg("同一射线上的圆弧与直线配方，以及两种状态和速度的读数",303,pieces)


def grid_frames(tensor, case):
    data = values(tensor)
    _, frames, height, width = tensor["shape"]
    start = case*frames*height*width
    clip = data[start:start+frames*height*width]
    pixels = [round(max(0,min(255,127.5*(value+1)))) for value in clip]
    rows = [pixels[i:i+width] for i in range(0,len(pixels),width)]
    return [rows[i:i+height] for i in range(0,len(rows),height)]


def films_figure(data):
    parts = [text(12,24,"相同的一份雪花，各自走 24 步",17,TEAL,bold=True)]
    for f in range(6):
        parts.append(text(190+81*f,49,f"帧 {f}",13,MUTED,"middle"))
    for i,(mode,caption) in enumerate(MODES.items()):
        row = data["rows"][mode]
        top = 60+i*67
        parts.extend((text(12,top+23,caption,15,INK,bold=True),
                      text(12,top+44,"终点参照" if i==0 else f'误差 {row["rmse"]:.3g}',13,MUTED)))
        for f, frame in enumerate(grid_frames(row["end"],0)):
            parts.append(bitmap(frame,155+81*f,top,70))
    frame, row_index, col_index = 2, 3, 8
    start = frame*12*16
    reference = values(data["rows"]["ddim"]["end"])[start:start+192]
    parts.extend((line(12,333,646,333,GRID),text(12,360,"帧 2 减去原版",16,TEAL,bold=True),
                  text(12,384,"两图共用色阶 ±0.04",13,MUTED),text(12,409,"蓝色较暗，橙色较亮",13,MUTED)))
    for i,(mode,caption) in enumerate((("matched","匹配刻度"),("uniform","等距刻度"))):
        frame_values = values(data["rows"][mode]["end"])[start:start+192]
        diff = [x-y for x,y in zip(frame_values,reference)]
        left, top = 177+242*i, 351
        parts.extend((tile(left,top,16,12,80,[signed(d,.04) for d in diff]),
                      f'<rect x="{left+5*col_index}" y="{top+5*row_index}" width="5" height="5" fill="none" stroke="#2b1111" stroke-width="1.3"/>',
                      text(left+90,374,caption,15,INK,bold=True),
                      text(left+90,398,f'框内 {diff[row_index*16+col_index]:.3g}',13,MUTED)))
    return svg("四种实际采样输出，同一帧的差图显示均分直线刻度造成的微小变化",427,parts)


def one_step_figure(record):
    p = next_step(record,0,12,2,5,7)
    parts = [text(12,22,"同一个生成过程，用两把尺量进度",17,TEAL,bold=True)]
    for lane, title in enumerate(("圆弧已走的角度比例","配比中的干净份额 τ")):
        y = 58+58*lane
        parts.extend((text(12,y+5,title,14,INK,bold=True),line(186,y,635,y,MUTED,1.5)))
        for n in range(record.steps+1):
            if n==record.steps:
                t=1.
            else:
                a,b,*_=record.coefficients[n]
                t=n/record.steps if lane==0 else a/(a+b)
            x=186+449*t
            mark=n in (0,6,12,18,24)
            parts.append(line(x,y-5,x,y+5,TEAL if mark else GRID,2 if mark else 1))
            if mark:
                parts.append(text(x,y+23,f"{max(0,t):.3f}".rstrip("0").rstrip("."),13,MUTED,"middle"))
            if n in (12,13):
                parts.append(circle(x,y,4,TEAL if n==12 else RUST))
    parts.append(text(12,172,"已更新次数 12 → 13，读同一帧同一格",16,TEAL,bold=True))
    for i,(label,key,caption) in enumerate((("当前直线读数","line","X"),("每单位进度的速度","speed","û"),("下一格的直线读数","euler","X′"))):
        x=12+218*i
        parts.extend((rect(x,186,205,85,LIGHT,4),text(x+12,211,label,15,INK),
                      text(x+12,247,f'{caption} = {p[key]:+.6f}',17,TEAL,bold=True)))
    parts.append(text(12,300,f'步长 Δτ = {p["next_tau"]:.6f} − {p["tau"]:.6f} ≈ {p["delta"]:.6f}',16,INK))
    return svg("配对时间网格与同一个视频像素的欧拉一步",316,parts)


def render(data,record,out,*,guide=False):
    if data["hashes"]["record"]!=hashlib.sha256(record.read_bytes()).hexdigest():
        raise ValueError("四排影片与单步放映记录不匹配")
    out.mkdir(parents=True,exist_ok=True)
    if record.resolve()!=(out/record.name).resolve():
        shutil.copyfile(record,out/record.name)
    diagram=geometry_diagram()
    (out/"path-coeffs.svg").write_text(diagram,encoding="utf-8")
    (out/"path-step.svg").write_text(one_step_figure(Recording(record)),encoding="utf-8")
    (out/"path-films.svg").write_text(films_figure(data),encoding="utf-8")
    template=Path(__file__).with_name("path_ui.html").read_text(encoding="utf-8")
    nav = ('<nav><a href="../index.html">离线首页</a><a href="../video-diffusion-guide.pdf#page=75">PDF 第 75 页</a>'
           '<a href="two-clocks.html#update">打开原版采样录像</a></nav>') if guide else ""
    html=template.replace("__GUIDE__",nav).replace("__GEOMETRY__",diagram)
    html=html.replace("__RESULTS__",json.dumps(data,separators=(",",":")))
    (out/"path-clock.html").write_text(html,encoding="utf-8")
