"""读取两次真实 DDIM 采样的显示像素，复算片长或导出可播放的六帧动画。仅用标准库。"""
import argparse
import base64
from fractions import Fraction
import json
from pathlib import Path
import struct
import zlib

from draw import GRID, INK, MUTED, RUST, TEAL, line, svg, text
from pixel_report import bitmap

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "pixel-lab/timing.json"


def frames(data, panel, case):
    cases, count, height, width = data["shape"]
    if not 1 <= case <= cases:
        raise ValueError(f"起点编号应为 1 到 {cases}")
    raw = base64.b64decode(data["panels"][panel], validate=True)
    size = width*height
    if len(raw) != cases*count*size:
        raise ValueError("像素记录长度与影片尺寸不符")
    start = (case-1)*count*size
    return [raw[start+i*size:start+(i+1)*size] for i in range(count)]


def measure(data, case, frame, row, col, fps):
    _, count, height, width = data["shape"]
    if not (0 <= frame < count and 0 <= row < height and 0 <= col < width and fps > 0):
        raise ValueError("帧号、采样位置或帧率越界")
    movies = {key:frames(data,key,case) for key in ("partial","6","24")}
    different = [sum(a!=b for a,b in zip(s,l)) for s,l in zip(movies["6"],movies["24"])]
    return {"起点":case, "选定坐标 [帧,行,列]":[frame,row,col],
            "各片此处亮度":{key:film[frame][row*width+col] for key,film in movies.items()},
            "两段成片各帧不同格数":different, "总不同格数":sum(different),
            "总格数":count*height*width, "帧起始秒":str(Fraction(frame,fps)),
            "末帧起始秒":str(Fraction(count-1,fps)), "片长秒":str(Fraction(count,fps))}


def animation(film, width, height, fps):
    """将每个源像素放大 12 倍；APNG 的每帧时长按 1/fps 秒写入。"""
    scale = 12
    w, h = width*scale, height*scale
    def block(name, payload):
        return struct.pack(">I",len(payload))+name+payload+struct.pack(">I",zlib.crc32(name+payload))
    parts = [b"\x89PNG\r\n\x1a\n",block(b"IHDR",struct.pack(">IIBBBBB",w,h,8,0,0,0,0)),
             block(b"acTL",struct.pack(">II",len(film),1))]
    seq = 0
    for i,pixels in enumerate(film):
        control = struct.pack(">IIIIIHHBB",seq,w,h,0,0,1,fps,0,0)
        parts.append(block(b"fcTL",control))
        seq += 1
        scanlines = []
        for start in range(0,len(pixels),width):
            row = b"".join(bytes([gray])*scale for gray in pixels[start:start+width])
            scanlines.append((b"\0"+row)*scale)
        compressed = zlib.compress(b"".join(scanlines))
        if i==0:
            parts.append(block(b"IDAT",compressed))
        else:
            parts.append(block(b"fdAT",struct.pack(">I",seq)+compressed))
            seq += 1
    return b"".join(parts)+block(b"IEND",b"")


def plate(data):
    _, count, height, width = data["shape"]
    pieces = [text(12,24,"同一份雪花出发　·　每排都是六帧",17,TEAL,bold=True),
              text(655,47,"框内",13,RUST,"end")]
    for f in range(count):
        pieces.append(text(174+77*f+30,47,f"帧 {f}",14,TEAL if f==2 else MUTED,"middle",f==2))
    for i,(key,label,note) in enumerate((("start","共同起点","还没调用网络"),("partial","长日程 · 6/24","才走四分之一"),
                                        ("24","长日程 · 24/24","抵达终点"),("6","短日程 · 6/6","重新从起点走完"))):
        top = 53+i*57+(5 if i==3 else 0)
        if i==3:
            pieces.append(line(12,top-6,644,top-6,GRID,1.2,"5 4"))
        pieces.extend([text(12,top+18,label,16,INK,bold=True),text(12,top+39,note,13,MUTED)])
        for f,pixels in enumerate(frames(data,key,1)):
            left = 174+77*f
            rows = [pixels[j:j+width] for j in range(0,len(pixels),width)]
            pieces.append(bitmap(rows,left,top,60))
            if f==2:
                px, py = left+10*60/width,top+5*60/width
                pieces.append(f'<rect x="{px-.7}" y="{py-.7}" width="5.15" height="5.15" fill="none" stroke="#ffa542" stroke-width="1.5"/>')
                pieces.append(text(655,top+28,str(pixels[5*width+10]),14,RUST,"end",True))
    return svg("同一六帧噪声的两条日程，标出帧2的同一像素",283,pieces)


def render(data, destination, *, course=False):
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"timing.svg").write_text(plate(data),encoding="utf-8")
    cases,_,h,w = data["shape"]
    clips = {}
    for case in range(1,cases+1):
        for budget in (6,24):
            film = frames(data,str(budget),case)
            for fps in (3,6):
                raw = animation(film,w,h,fps)
                clips[f"{case}-{budget}-{fps}"] = base64.b64encode(raw).decode("ascii")
    template = Path(__file__).with_name("timing_ui.html").read_text(encoding="utf-8")
    lesson = {"shape":data["shape"],"panels":data["panels"],"clips":clips}
    nav = ('<nav><a href="../index.html">离线首页</a>　<a href="two-clocks.html">逐步查看原始采样记录</a>'
           '　<a href="../page-index.html#p6">第 6 页的公式与代码</a></nav>') if course else ""
    html = template.replace("__NAV__",nav).replace("__LESSON__",json.dumps(lesson,separators=(",",":")))
    (destination/"timing.html").write_text(html,encoding="utf-8")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record",type=Path,default=DEFAULT)
    parser.add_argument("--case",type=int,default=1)
    parser.add_argument("--frame",type=int,default=2)
    parser.add_argument("--row",type=int,default=5)
    parser.add_argument("--col",type=int,default=10)
    parser.add_argument("--fps",type=int,choices=(3,6),default=3)
    parser.add_argument("--steps",type=int,choices=(6,24),default=6,help="选择要导出的成片")
    parser.add_argument("--animation",type=Path,help="导出播放一次的动态 PNG")
    args = parser.parse_args()
    data = json.loads(args.record.read_text(encoding="utf-8"))
    try:
        print(json.dumps(measure(data,args.case,args.frame,args.row,args.col,args.fps),indent=2,ensure_ascii=False))
        if args.animation:
            _,_,h,w = data["shape"]
            film = frames(data,str(args.steps),args.case)
            args.animation.parent.mkdir(parents=True,exist_ok=True)
            args.animation.write_bytes(animation(film,w,h,args.fps))
    except ValueError as ex:
        parser.error(str(ex))
