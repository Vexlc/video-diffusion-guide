"""从四道纯 Python 练习生成可携带的 Jupyter 笔记本、离线预览和印刷图。"""
import argparse
import base64
from html import escape
import inspect
import json
from pathlib import Path

from draw import GRID, INK, MUTED, RUST, TEAL, WIDTH, circle, line, rect, svg, text
from pixel_report import bitmap, png_uri
from lane_lessons import (lane_path, lane_speed, lesson_parser, noise_problem, read_settings,
                          square, train_exit, video_lanes)


class Picture:
    def __init__(self, markup):
        self.markup = markup

    def _repr_svg_(self):
        return self.markup

    def __repr__(self):
        return "实验输出图"


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


LESSONS = [
    ("noise","扩散","雪花猜多一点，亮块会变暗多少？","2、7–8",
     "只看六帧方块中的帧 2。干净图 y 混入标准高斯噪声 z，得到 x = a·y + σ·z。取 a = 0.8、σ = 0.6。我们知道出题用的 z，故意把它猜为 z + δ，再按 ŷ = (x − σ·猜测)/a 还原。每格的偏差都是 −σδ/a。",
     [noise_problem],"noise_error = 0.15\nPicture(noise_plot(noise_problem(noise_error)))",
     "先猜 δ 改成 0 后橙框会读到什么。再试 −0.15，检查橙框会变亮还是变暗。这里用已知噪声做校验，训练去噪网络的流程见第 29 页。","https://arxiv.org/abs/2006.11239","DDPM 论文"),
    ("flow","流匹配","一份噪声会停在中间，还是走向一条车道？","3、10–11、89",
     "把两条目标行写成 y = −1 或 +1，各占一半。起点 z 是独立的标准高斯数，练习路径为 x = (1−t)z + ty。在同一题面，平均终点是 tanh(tx/(1−t)²)，平均速度是 (平均终点−x)/(1−t)。这组分布允许用公式精确计算速度，省去了拟合网络的误差。",
     [lane_speed,lane_path],"steps = 12\nstart = 0.4\nPicture(flow_plot(steps, start))",
     "先把 steps 改为 2，再改为 12，盯着橙线的终点。把 start 改为 0，找出对称中心为什么会一直停住。图中的起点是特意选来观察轨迹的几个数。","https://arxiv.org/abs/2210.02747","Flow Matching 论文"),
    ("video","视频","每一帧都在合法行，连起来有多少段会跳？","4、13–14",
     "数据里每段方块向右走，整段任选第 3 或第 7 行。逐帧独立抽签和整段共用一次抽签，每一帧落在任一行的概率都是 ½。六次独立抽签会产生 2⁶ 条等概率序列，其中全上、全下两条符合整段规则。",
     [video_lanes],"seed = 17\nPicture(video_plot(video_lanes(seed)))",
     "改 seed 可以换一卷。两排的首帧相同，横着读第二至六帧。独立抽签整段合法的概率为 2/64。这个抽签实验隔离了帧间依赖，第 14 页再看实际联合去噪。","https://arxiv.org/abs/2204.03458","Video Diffusion Models 论文"),
    ("control","ControlNet","出口起初是零，两个位置提示何时能分开？","5、16、85–87",
     "主干 h 冻结在 0，条件 c 取 −1、+1，两个目标分别是 c 本身。出口只有一个参数 w，从 0 开始学。训练用强度 1，最小化 ½·平均[(wc−c)²]，所以梯度等于 w−1。生成时才用 h + λwc 读数，乘 2 再加 5 就是屏幕行号。",
     [train_exit],"updates = 8\nstrength = 1\nPicture(control_plot(train_exit(updates, strength)))",
     "先设 updates = 0，试着改变强度。然后恢复 8 次更新，再把 strength 设成 0。两者都能让位置回到第 5 行，权重曲线却不同。此处的单参数回归演示控制出口，视频网络的多尺度出口见第 86 页。","https://arxiv.org/abs/2302.05543","ControlNet 论文")
]


def src(*functions):
    return "\n\n".join(inspect.getsource(f).strip() for f in functions)


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


def render(destination, *, course=False):
    destination.mkdir(parents=True,exist_ok=True)
    imports = ("import base64, struct, zlib\nfrom html import escape\nfrom math import sqrt, tanh\nfrom random import Random\n"
               f"\nGRID, INK, MUTED, RUST, TEAL, WIDTH = {GRID!r}, {INK!r}, {MUTED!r}, {RUST!r}, {TEAL!r}, {WIDTH}\n\n")
    plots = src(text,line,rect,circle,svg,png_uri,bitmap,gray,noise_plot,flow_plot,video_plot,control_plot)
    prelude = imports + src(square,Picture) + "\n\n" + plots
    script = ('"""从雪花到运动 · 四个可独立运行的标准库实验。python first_lab.py --help 查看参数。"""\n'
              "import argparse\nfrom math import isfinite\nfrom pathlib import Path\n" + imports
              + src(square,noise_problem,lane_speed,lane_path,video_lanes,train_exit) + "\n\n"
              + plots + "\n\n" + src(lesson_parser,read_settings,personal_report,personal_main)
              + '\n\nif __name__ == "__main__":\n    personal_main()\n')
    script_bytes = script.encode("utf-8")
    (destination/"first_lab.py").write_bytes(script_bytes)
    title = "四道可以改数的生成模型练习"
    intro = "从头运行一次。第一格准备绘图工具，后面四题可以单独改参数重跑。笔记本内含全部代码和默认结果，计算与绘图只用 Python 标准库。"
    cells = []
    def md(key,body):
        cells.append({"id":key,"cell_type":"markdown","metadata":{},"source":body})
    def code(key,body,outputs,extra=None):
        count=1+sum(c["cell_type"]=="code" for c in cells)
        cells.append({"id":key,"cell_type":"code","metadata":extra or {},"source":body,"execution_count":count,"outputs":outputs})
    md("introduction","# "+title+"\n\n"+intro+"\n\n### 准备绘图工具\n\n先运行下一格。它负责把数字画出来，四道题的计算都写在各题下面。")
    code("plot-tools",prelude,[],{"jupyter":{"source_hidden":True}})
    values = [noise_plot(noise_problem(.15)),flow_plot(12,.4),video_plot(video_lanes(17)),control_plot(train_exit(8,1))]
    defaults = ("误差 +0.15","12 步 · 起点 0.4","种子 17","训练 8 次 · 强度 1")
    # 每个对照只改变默认单元格里的一项设置。
    alternatives = (
        [("误差为零","noise_error = 0.15","noise_error = 0",noise_plot(noise_problem(0))),
         ("误差 −0.15","noise_error = 0.15","noise_error = -0.15",noise_plot(noise_problem(-.15)))],
        [("只走 2 步","steps = 12","steps = 2",flow_plot(2,.4)),
         ("从中线 0 出发","start = 0.4","start = 0",flow_plot(12,0))],
        [("种子 4","seed = 17","seed = 4",video_plot(video_lanes(4)))],
        [("从零开始 · 尚未训练","updates = 8","updates = 0",control_plot(train_exit(0,1))),
         ("已训练 · 强度 0","strength = 1","strength = 0",control_plot(train_exit(8,0)))],
    )
    parts = []
    for i,(item,figure) in enumerate(zip(LESSONS,values)):
        key,topic,question,pages,description,functions,example,action,url,reference = item
        md(key+"-question",f"## {topic} · {question}\n\n讲义第 {pages} 页。{description}\n\n{action}\n\n[{reference}]({url})")
        body=src(*functions)+"\n\n"+example
        code(key+"-code",body,[{"output_type":"execute_result","execution_count":i+2,"metadata":{},
                                "data":{"text/plain":"实验输出图","image/svg+xml":figure}}])
        runs = [(defaults[i],example,figure)]
        for label,old,new,plot in alternatives[i]:
            runs.append((label,example.replace(old,new),plot))
        buttons = []
        panels = []
        for j,(label,params,plot) in enumerate(runs):
            buttons.append(f'<button type="button" data-pick="{j}" aria-pressed="{str(j==0).lower()}" '
                           f'aria-controls="{key}-run-{j}">{label}</button>')
            active = "" if j==0 else " hidden"
            panels.append(f'<div id="{key}-run-{j}" data-run="{j}"{active}><figure tabindex="0" aria-label="{escape(label)}的输出图">{plot}</figure>'
                          f'<details><summary>查看这组参数和 Python 代码</summary>'
                          f'<pre><code>{escape(src(*functions))}\n\n{escape(params)}</code></pre></details></div>')
        parts.append(f'<section id="{key}"><small>{topic}　/　PDF 第 {pages} 页</small><h2>{question}</h2>'
                     f'<p>{description}</p><div class="variants" role="group" aria-label="{topic}参数">{"".join(buttons)}</div>'
                     f'{"".join(panels)}<p>{action}</p><p><a href="{url}">{reference}</a></p></section>')
    nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"}},"cells":cells}
    notebook_bytes = (json.dumps(nb,ensure_ascii=False,indent=1)+"\n").encode("utf-8")
    (destination/"first-lab.ipynb").write_bytes(notebook_bytes)
    (destination/"lane-flow.svg").write_text(values[1],encoding="utf-8")
    template=Path(__file__).with_name("notebook_ui.html").read_text(encoding="utf-8")
    nav='<a href="../index.html">离线首页</a>　<a href="../video-diffusion-lab.html#studyMap">讲义阅读地图</a>' if course else ""
    download = "data:application/octet-stream;base64,"+base64.b64encode(notebook_bytes).decode("ascii")
    script_url = "data:application/octet-stream;base64," + base64.b64encode(script_bytes).decode("ascii")
    (destination/"index.html").write_text(template.replace("__NAV__",nav).replace("__DOWNLOAD__",download)
                                         .replace("__SCRIPT__",script_url).replace("__LESSONS__","\n".join(parts)),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("replay/first-notebook"))
    args=parser.parse_args()
    render(args.out)
