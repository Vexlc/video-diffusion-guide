"""用真实的六帧成片，计算正放、倒放、换条件再生成时的速度和逐像素差。只用标准库。"""
import argparse
import base64
import json
from pathlib import Path

from draw import INK, MUTED, RUST, TEAL, rect, svg, text
from pixel_report import bitmap

ROOT = Path(__file__).resolve().parents[1]
W, H, FRAMES = 16, 12, 6
PANEL = (("right",False,"向右生成"),("right",True,"倒放第一排"),
         ("left",False,"改为向左"),("left_end",False,"再移动起点"))
ORDER = list(range(FRAMES))


def display(run, reverse):
    """倒放时只重排解码后的图像及其中心，采样记录沿用原来的生成时间。"""
    order = ORDER[::-1] if reverse else ORDER
    raw = base64.b64decode(run["film"],validate=True)
    if len(raw)!=FRAMES*H*W:
        raise ValueError("需要六帧 12 × 16 的灰度影片")
    return {"order":order, "frames":[raw[f*W*H:(f+1)*W*H] for f in order],
            "centers":[run["centers"][f] for f in order], "x":run["x"], "u":run["u"]}


def measure(run, frame, step, fps):
    center = run["centers"][frame]
    velocity = None
    if frame+1<FRAMES:
        following = run["centers"][frame+1]
        velocity = [(after-before)*fps for before,after in zip(center,following)]
    return {"source_frame":run["order"][frame], "center":center, "time_seconds":frame/fps,
            "pixels_per_second":velocity, "x":run["x"][step], "u":run["u"][step],
            "next":run["x"][step+1]}


def compare(a,b):
    """两张成片按播放次序逐帧对齐，在相同画面坐标比较 8 位显示亮度。"""
    hits, changes, sums, peaks = [], [], [], []
    for i,(left,right) in enumerate(zip(a["frames"],b["frames"])):
        delta = [abs(x-y) for x,y in zip(left,right)]
        hits.append(a["centers"][i]==b["centers"][i])
        changes.append(sum(d!=0 for d in delta))
        sums.append(sum(delta))
        loc = delta.index(max(delta))
        peaks.append({"col":loc%W,"row":loc//W,"a":left[loc],"b":right[loc],"diff":delta[loc]})
    return {"center_hits":sum(hits), "changes":changes, "sum_abs":sums, "peaks":peaks,
            "different":sum(changes), "mae":sum(sums)/(FRAMES*W*H)}


def rows(raw):
    return [list(raw[y*W:(y+1)*W]) for y in range(H)]


def gallery(group):
    parts = [text(12,24,"横读一排是同一段影片，竖读一列对齐播放时刻",17,TEAL,bold=True)]
    for f in ORDER:
        parts.append(text(183+82*f,48,f"播放格 {f}",14,MUTED,"middle"))
    for i,(key,reverse,title) in enumerate(PANEL):
        run = display(group[key],reverse)
        top = 59 + 72*i
        parts.extend((text(12,top+18,title,16,INK,bold=True),
                      text(12,top+43,f"u = {run['u'][12]:+.4f}".replace("-","−"),14,TEAL)))
        for f,frame in enumerate(run["frames"]):
            left = 151+82*f
            parts.extend((bitmap(rows(frame),left,top,63),text(left+31,top+63,
                     f"列 {run['centers'][f][0]}",13,MUTED,"middle")))
    return svg("同一噪声三次生成，向右片另作一次倒放。每格的列号由成片实测。",347,parts)


def difference_strip(group):
    a,b = display(group["right"],True),display(group["left_end"],False)
    scores = compare(a,b)
    parts = [text(12,24,"倒放第一排与重新指定起点的向左片",17,TEAL,bold=True),
             text(12,52,"橙格标出",14,INK),text(12,74,"亮度不同处",14,INK)]
    for f,(left,right) in enumerate(zip(a["frames"],b["frames"])):
        x,y,size = 151+82*f,35,4
        parts.append(rect(x,y,W*size,H*size,"#e0e8e3",1))
        for j,(v,w) in enumerate(zip(left,right)):
            if v!=w:
                parts.append(rect(x+(j%W)*size,y+(j//W)*size,size,size,RUST))
        parts.append(text(x+32,102,f"{scores['changes'][f]} 格",14,INK,"middle"))
    return svg("沿六个播放格逐像素比较，橙色表示两段影片在同一位置的显示亮度不等",112,parts)


def render(record,out,*,course=False):
    out.mkdir(parents=True,exist_ok=True)
    default = record["seeds"].get("9317",next(iter(record["seeds"].values())))
    for name,fig in (("playback-orders",gallery(default)),("playback-differences",difference_strip(default))):
        (out/(name+".svg")).write_text(fig,encoding="utf-8")
    template = Path(__file__).with_name("playback_ui.html").read_text(encoding="utf-8")
    links = ('<nav><a href="../index.html">离线首页</a><a href="motion.html#times">回到两只钟</a>'
             '<a href="first-film.html">在本机运行生成器</a><a href="../page-index.html#p68">第 68 页源码</a></nav>') if course else ""
    html = template.replace("__LINKS__",links).replace("__RECORD__",json.dumps(record,separators=(",",":")))
    (out/"playback.html").write_text(html,encoding="utf-8")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record",type=Path,default=ROOT/"flow-hint/playback.json")
    parser.add_argument("--seed",type=int,default=9317)
    parser.add_argument("--frame",type=int,choices=range(FRAMES),default=2)
    parser.add_argument("--step",type=int,choices=range(24),default=12)
    parser.add_argument("--fps",type=int,choices=(3,6,12),default=6)
    args = parser.parse_args()
    seeds = json.loads(args.record.read_text())["seeds"]
    if str(args.seed) not in seeds:
        parser.error("此记录可用的种子为 "+", ".join(seeds))
    group = seeds[str(args.seed)]
    for key,reverse,title in PANEL:
        print(title,json.dumps(measure(display(group[key],reverse),args.frame,args.step,args.fps),ensure_ascii=False))
    print("倒放与移点生成",json.dumps(compare(display(group["right"],True),display(group["left_end"],False)),ensure_ascii=False))
