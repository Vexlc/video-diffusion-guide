"""从雪花到运动 · 四个可独立运行的标准库实验。python first_lab.py --help 查看参数。"""
import argparse
from math import isfinite
from pathlib import Path
import base64, struct, zlib
from html import escape
from math import sqrt, tanh
from random import Random

GRID, INK, MUTED, RUST, TEAL, WIDTH = '#c8d7d0', '#193540', '#495f66', '#ad492c', '#00695f', 660

def square(frame, row):
    """六帧小片里的一张灰度图，方块每帧向右移一列。"""
    center = 4 + frame
    return [[0.8 if abs(x-center)<=1 and abs(y-row)<=1 else -0.55
             for x in range(16)] for y in range(12)]

def noise_problem(noise_error):
    original = square(frame=2, row=3)
    rng = Random(17)
    noise = [[rng.gauss(0,1) for x in range(16)] for y in range(12)]
    a = 0.8
    sigma = sqrt(1-a*a)
    mixed = [[a*y + sigma*z for y,z in zip(pixels,snow)]
             for pixels,snow in zip(original,noise)]
    # 已知这次出题用的噪声，给每一格的猜测故意加同样的误差。
    restored = [[(x-sigma*(z+noise_error))/a for x,z in zip(pixels,snow)]
                for pixels,snow in zip(mixed,noise)]
    return original, mixed, restored

def lane_speed(x, t):
    # y 取 -1 或 +1，各占一半；z 是独立标准高斯噪声。
    mean_y = tanh(t*x/(1-t)**2)
    return (mean_y-x)/(1-t)

def lane_path(z, steps):
    if steps < 1:
        raise ValueError("至少走一步")
    path = [z]
    for k in range(steps):
        t = k/steps
        x = path[-1]
        path.append(x + lane_speed(x,t)/steps)
    return path

def video_lanes(seed):
    rng = Random(seed)
    rows = [rng.choice((3,7)) for frame in range(6)]
    # 共享首帧的抽签结果，其余帧保持同一行。
    shared = [rows[0]]*6
    legal = 2                      # 全在第 3 行，或全在第 7 行
    possible = 2**len(rows)         # 每一帧独立二选一
    return rows, shared, legal, possible

def train_exit(updates, strength):
    if updates < 0:
        raise ValueError("训练次数不能为负")
    weight = 0.0
    history = [weight]
    for step in range(updates):
        # 冻结主干 h=0。训练用强度 1，两个条件同样常见，目标值就是条件 c。
        gradient = sum((weight*c-c)*c for c in (-1,1))/2
        weight -= 0.25*gradient
        history.append(weight)
    # 归一化坐标 -1、+1 分别对应第 3、7 行。
    return history, [5+2*strength*weight*c for c in (-1,1)]

def text(left, top, content, size=15, color=INK, anchor="start", bold=False):
    return (f'<text x="{left}" y="{top}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{700 if bold else 400}">{escape(str(content))}</text>')

def line(left, top, right, bottom, color=GRID, width=1, dash=""):
    return (f'<line x1="{left}" y1="{top}" x2="{right}" y2="{bottom}" stroke="{color}" '
            f'stroke-width="{width}" stroke-dasharray="{dash or "none"}"/>')

def rect(left, top, width, height, fill, radius=0):
    return f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="{radius}" fill="{fill}"/>'

def circle(left, top, radius, color):
    return f'<circle cx="{left}" cy="{top}" r="{radius}" fill="{color}"/>'

def svg(title, height, pieces):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" role="img" '
            f'aria-label="{escape(title)}" font-family="MiSans, Noto Sans CJK SC, Microsoft YaHei, sans-serif">'
            f'<title>{escape(title)}</title>' + rect(0, 0, WIDTH, height, "#fbfcf8")
            + "".join(pieces) + '</svg>')

def png_uri(rows):
    def block(name, payload):
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name+payload))
    header = struct.pack(">IIBBBBB", len(rows[0]), len(rows), 8, 0, 0, 0, 0)
    stream = b"".join(b"\x00"+bytes(row) for row in rows)
    png = b"\x89PNG\r\n\x1a\n"+block(b"IHDR", header)+block(b"IDAT", zlib.compress(stream))+block(b"IEND", b"")
    return "data:image/png;base64,"+base64.b64encode(png).decode()

def bitmap(rows, left, top, width):
    return (f'<image x="{left}" y="{top}" width="{width}" height="{width*len(rows)/len(rows[0])}" '
            f'style="image-rendering:pixelated" href="{png_uri(rows)}"/>')

def gray(values):
    return [[round((max(-1,min(1,v))+1)*127.5) for v in row] for row in values]

def noise_plot(result):
    parts = []
    for i,(pixels,label) in enumerate(zip(result,("干净的帧 2","混入雪花","按猜测还原"))):
        left = 24+216*i
        parts.extend([text(left,24,label,16,TEAL,bold=True),bitmap(gray(pixels),left,35,160),
                      text(left,180,f"框内 {pixels[3][6]:.4f}",15,INK)])
        parts.append(f'<rect x="{left+60}" y="65" width="10" height="10" fill="none" stroke="#ffad46" stroke-width="2"/>')
    return svg("同一格亮度在加噪和还原后怎样变化",190,parts)

def flow_plot(steps,start):
    parts = []
    extent = max(1.6,abs(start)*1.1)
    for i,n in enumerate((1,steps)):
        left = 45+321*i
        px = lambda k: left+247*k/n
        py = lambda value: 132-78*value/extent
        parts.append(text(left,23,f"问 {n} 次速度",17,TEAL,bold=True))
        for y in (-1,0,1):
            parts.extend([line(left,py(y),px(n),py(y),"#ccd9d2",1,"4 3"),
                          text(left-8,py(y)+5,str(y),13,MUTED,"end")])
        for z in (-1.4,-.8,-.4,.4,.8,1.4,start):
            track = lane_path(z,n)
            color = RUST if z==start else TEAL
            points = " ".join(f"{px(k):.3f},{py(x):.3f}" for k,x in enumerate(track))
            parts.extend([f'<polyline fill="none" points="{points}" stroke="{color}" stroke-width="{2.7 if z==start else 1.4}"/>',
                          circle(px(n),py(track[-1]),3.5,color)])
        parts.extend([text(left,230,"噪声端 0",13,MUTED),text(px(n),230,"成片端 1",13,MUTED,"end"),
                      text(left,257,f"起点 {start:g}　→　{lane_path(start,n)[-1]:.4f}",15,RUST,bold=True)])
    return svg("同一批纵坐标分别用一步和多步沿平均速度向两条目标行生成",270,parts)

def video_plot(result):
    independent,shared,legal,possible = result
    parts = []
    for i,(rows,label) in enumerate(((independent,"每帧各抽"),(shared,"整段共用"))):
        top = 34+76*i
        parts.append(text(10,top+25,label,15,TEAL,bold=True))
        for f,row in enumerate(rows):
            left = 117+88*f
            parts.extend([bitmap(gray(square(f,row)),left,top,69),
                          text(left+34,top+67,f"行 {row}",13,INK,"middle")])
    parts.append(text(12,200,f"独立抽行共有 {possible} 种结果，只有 {legal} 种全程保持同一行。",15,INK))
    return svg("六帧方块的逐帧抽行和整段共用行号",212,parts)

def control_plot(result):
    history,positions = result
    parts = [text(18,24,"出口权重 w 从零开始学习",16,TEAL,bold=True),
             text(345,24,"用同一权重，分别要求上下两行",16,TEAL,bold=True)]
    px = lambda k: 48+240*k/max(1,len(history)-1)
    py = lambda w: 151-93*w
    for w in (0,1):
        parts.extend([line(48,py(w),288,py(w),"#c5d6ce",1,"5 3"),text(38,py(w)+4,str(w),13,MUTED,"end")])
    points = " ".join(f"{px(k):.3f},{py(w):.3f}" for k,w in enumerate(history))
    parts.extend([f'<polyline points="{points}" fill="none" stroke="{TEAL}" stroke-width="2"/>',
                  circle(px(len(history)-1),py(history[-1]),4,TEAL),
                  text(18,181,f"{len(history)-1} 次更新后 w = {history[-1]:.4f}",15,INK)])
    lo,hi=min(2,*positions),max(8,*positions)
    screen_y = lambda row: 47+99*(row-lo)/(hi-lo)
    for i,(target,actual) in enumerate(zip((3,7),positions)):
        left=350+149*i
        parts.extend([rect(left,43,121,109,"#e8efe9",3),line(left,screen_y(target),left+121,screen_y(target),RUST,1.8,"5 3"),
                      circle(left+58,screen_y(actual),6,TEAL),text(left,179,f"预测行 {actual:.3f}",14,INK)])
    parts.append(text(18,206,"虚线是请求的行，青点是冻结主干加上位置修正后的连续坐标。",14,MUTED))
    return svg("标量控制支路的学习曲线与两种请求的实际响应",218,parts)

def lesson_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--steps", type=int, default=12, help="流匹配采样的欧拉步数，默认 12")
    parser.add_argument("--start", type=float, default=.4, help="跟踪的一格起始噪声，默认 0.4")
    parser.add_argument("--noise-error", type=float, default=.15, help="噪声猜测误差，默认 0.15")
    parser.add_argument("--seed", type=int, default=17, help="逐帧抽行的随机种子，默认 17")
    parser.add_argument("--updates", type=int, default=8, help="控制出口的训练次数，默认 8")
    parser.add_argument("--strength", type=float, default=1, help="训练完成后的控制强度，默认 1")
    return parser

def read_settings(parser):
    args = parser.parse_args()
    if args.steps < 1 or args.updates < 0:
        parser.error("采样步数应为正，训练次数至少为零")
    if not all(isfinite(value) for value in (args.start, args.noise_error, args.strength)):
        parser.error("起始噪声、猜测误差和控制强度须为有限数")
    return args

def personal_report(args):
    frames = noise_problem(args.noise_error)
    restored = frames[-1][3][6]
    path = lane_path(args.start, args.steps)
    endpoint = path[-1]
    lanes = video_lanes(args.seed)
    control = train_exit(args.updates, args.strength)
    changes = sum(left != right for left, right in zip(lanes[0], lanes[0][1:]))
    figures = [
        ("noise", "扩散", "先看橙框里的一格亮度", noise_plot(frames),
         f"猜测误差 δ = {args.noise_error:g}。混入噪声时，干净亮度乘 0.8，噪声乘 0.6。还原时多减了 "
         f"0.6 × δ，再除以 0.8。于是橙框从 0.8 变为 0.8 − 0.75 × δ = {restored:.4f}。", "2、7–8"),
        ("flow", "流匹配", "沿橙线看同一份噪声走到哪里", flow_plot(args.steps, args.start),
         f"横轴从噪声端走向成片端，纵轴记录位置。每步的位置变化 = 当时的平均速度 × 步长。"
         f"从 {args.start:g} 出发，左图只走一步，终点为 0。右图把路分成 {args.steps} 步，终点为 {endpoint:.4f}。"
         "目标只有 −1 和 +1 两行，速度由这两行的概率精确算出。", "3、10–11、89"),
        ("video", "视频", "手指沿着每一排走完六帧", video_plot(lanes),
         f"种子 {args.seed}。方块每帧向右移一列。第一排每帧重新抽行，这一次换了 {changes} 次行。"
         "第二排沿用首帧行号，整段都在同一行。逐帧独立抽行共有 64 种等概率结果，只有 2 种符合整段规则。", "4、13–14"),
        ("control", "ControlNet", "看控制出口怎样学会回应位置提示", control_plot(control),
         f"这里把控制出口缩成一个权重 w。主干固定，训练 {args.updates} 次后 w = {control[0][-1]:.4f}。"
         f"采样强度 λ = {args.strength:g}，请求的两行为 3 和 7。代入 5 ± 2λw，实际预测为 "
         f"{control[1][0]:.4f} 和 {control[1][1]:.4f}。橙色虚线标请求，青点标预测。", "5、16、85–87"),
    ]
    sections = []
    for key, topic, question, plot, caption, pages in figures:
        trace = ""
        if key == "flow":
            count = args.steps
            steps = range(count) if count <= 24 else (0, 1, count//2, count-1)
            note = "下表列出全部更新。" if count <= 24 else "下表选取头两次、中间一次与最后一次更新。"
            rows = []
            for k in steps:
                t = k/count
                x = path[k]
                speed = lane_speed(x, t)
                cells = (k+1, f"{t:.4f}", f"{x:.4f}", f"{speed:.4f}", f"{speed/count:.4f}", f"{path[k+1]:.4f}")
                rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in cells) + "</tr>")
            trace = (f'<details><summary>把橙线的一步拆开算</summary><p>每步时长为 1/{count}。'
                     f'从左往右算，用起点加上位移得到新位置。{note}表中数值保留四位小数。</p>'
                     '<div class="numbers" tabindex="0" role="region" aria-label="橙线的速度与位置">'
                     '<table><thead><tr><th>步</th><th>进度 t</th><th>起点 x</th>'
                     '<th>速度 u</th><th>位移 u × 时长</th><th>新位置</th></tr></thead><tbody>'
                     + "".join(rows) + '</tbody></table></div></details>')
        sections.append(f'<section id="{key}"><small>{topic} · 讲义第 {pages} 页</small>'
                        f'<h2>{question}</h2><figure tabindex="0" aria-label="{question}">{plot}</figure>'
                        f'<p>{caption}</p>{trace}</section>')
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>我的四道生成模型实验</title><style>
*{box-sizing:border-box}body{margin:0;background:#f0f4ef;color:#17373a;font:17px/1.65 system-ui,sans-serif}
main{max-width:780px;margin:auto;padding:clamp(12px,3vw,34px)}h1,h2{line-height:1.35}
small,a{color:#005e54}nav a{display:inline-block;margin-right:1em}section{background:white;margin:20px 0;
padding:clamp(12px,3vw,24px);border-top:3px solid #00695f}h2{font-size:1.45em;margin:6px 0}
figure{margin:12px 0;overflow-x:auto}svg{display:block;width:100%;min-width:500px;height:auto}
figure:focus-visible,a:focus-visible,summary:focus-visible,.numbers:focus-visible{outline:3px solid #ae532c}
summary{color:#005e54;cursor:pointer;font-weight:650}.numbers{overflow-x:auto}
table{border-collapse:collapse;min-width:510px;width:100%;font-variant-numeric:tabular-nums}
td,th{text-align:right;padding:5px 7px;border-bottom:1px solid #cadaD2}th{background:#eaf1ec}
@media print{section{break-inside:avoid}nav{display:none}main{max-width:none}}
</style><main><h1>我的四道生成模型实验</h1><p>这四张图来自这次 Python 运行。
先看每题的图，再读下面的数字。小屏可横滑看完整图。</p>
<nav aria-label="跳到实验"><a href="#noise">去噪</a><a href="#flow">流匹配</a>
<a href="#video">视频</a><a href="#control">控制</a></nav>""" + "".join(sections) + "</main></html>\n"

def personal_main():
    parser = lesson_parser("用本次参数画四张图，保存成可双击打开的离线网页。")
    parser.add_argument("--out", type=Path, default=Path("my-first-lab.html"), help="要保存的 HTML 文件")
    args = read_settings(parser)
    report = personal_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print("请用浏览器打开", args.out.resolve())

if __name__ == "__main__":
    personal_main()
