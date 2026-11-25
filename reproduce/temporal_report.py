"""读取实验 I 的固定网络记录，核对通道地址和整帧干预，并生成离线图文。"""
import argparse
import json
import math
import os
from html import escape
from pathlib import Path

from draw import INK, MUTED, TEAL, rect, svg, text
from hint_report import signed, tile, values


LAYERS = ("编码器入口之后","低分辨率卷积之前","上采样之后")


def source_frame(frame, channel, channels, enabled, frames=6):
    if not enabled or channel>=channels//2:
        return frame
    if channel<channels//4:
        return max(0,frame-1)
    return min(frames-1,frame+1)


def possible_sources(key):
    """每次移位取相邻帧的通道，随后的逐帧卷积混合它们；边界钳到最近一帧。"""
    sources=[{f} for f in range(6)]
    for mode in key:
        if mode=="1":
            sources=[sources[max(f-1,0)] | sources[f] | sources[min(f+1,5)] for f in range(6)]
    return sources


def differences(run):
    _,frames,h,w=run["shape"]
    raw=values(run)
    plane=h*w
    result=[]
    for replaced in range(frames):
        start=(replaced+1)*frames*plane
        result.append([sum(abs(raw[i+start]-raw[i]) for i in range(f*plane,(f+1)*plane))/plane
                       for f in range(frames)])
    return result


def address(data,key,layer,frame,channel,row,col):
    if len(key)!=3 or set(key)-{"0","1"} or layer not in range(3):
        raise ValueError("开关须为三个 0 或 1，交接序号为 0、1、2")
    spec=data["taps"][key[:layer]]
    frames,channels,h,w=spec["shape"]
    scale=2 if layer==2 else 1
    if not (0<=frame<frames and 0<=channel<channels and 0<=row<h*scale and 0<=col<w*scale):
        raise ValueError("帧号、通道、行或列超出该层范围")
    origin=source_frame(frame,channel,channels,key[layer]=="1")
    r,c=row//scale,col//scale
    features=values(spec)
    def at(f):
        return features[((f*channels+channel)*h+r)*w+c]
    return {"source_frame":origin,"local_value":at(frame),"received":at(origin),
            "stored_row":r,"stored_col":c}


def feature_sheet(data):
    raw=values(data["taps"][""])
    parts=[text(12,26,"打开第一处移位，追踪帧 2 的三个通道",17,TEAL,bold=True)]
    for f in range(6):
        parts.append(text(133+64*f,57,f"帧 {f}",14,MUTED,"middle"))
    parts.append(text(567,57,"帧 2 收到",15,TEAL,"middle",True))
    for group,channel in enumerate((2,6,12)):
        y=72+76*group
        origin=source_frame(2,channel,16,True)
        parts += [text(14,y+20,f"通道 {channel}",15,INK,bold=True),
                  text(14,y+41,["读上一帧","读下一帧","读本帧"][group],13,MUTED)]
        for f in range(7):
            src=origin if f==6 else f
            off=(src*16+channel)*192
            x=106+64*f if f<6 else 539
            colors=[signed(v,1) for v in raw[off:off+192]]
            parts.append(tile(x,y,16,12,54,colors))
            if f==origin or f==6:
                parts.append(f'<rect x="{x-2}" y="{y-2}" width="58" height="44.5" fill="none" stroke="{TEAL}" stroke-width="2"/>')
                parts.append(text(x+27,y+60,f"{raw[off+5*16+7]:+.4f}",13,INK,"middle",True))
                parts.append(f'<rect x="{x+7*54/16}" y="{y+5*54/16}" width="3.4" height="3.4" fill="none" stroke="#2b2625" stroke-width="1.4"/>')
        parts.append(text(503,y+27,"→",20,TEAL,"middle"))
    parts.append(text(14,308,"每个方格是一整张特征图。黑框取第 5 行、第 7 列；相同颜色刻度为 −1 到 +1。",14,MUTED))
    return svg("六帧特征来源与帧2接收到的三个通道",324,parts)


def strength_color(value):
    a=min(1, math.log1p(100*value)/math.log(101))
    return tuple(round(245+(c-245)*a) for c in (6,105,93))


def reach_sheet(data):
    parts=[text(12,25,"行：替换哪一帧输入　　列：查看哪一帧输出",17,TEAL,bold=True)]
    for m,(key,label) in enumerate([("000","三处全关"),("100","只开第一处"),("111","三处全开")]):
        left=38+215*m
        matrix=differences(data["runs"][key])
        reachable=possible_sources(key)
        parts.append(text(left+72,55,label,16,TEAL,"middle",True))
        for f in range(6):
            parts.append(text(left+12+24*f,79,f,13,MUTED,"middle"))
        for j,row in enumerate(matrix):
            y=87+24*j
            parts.append(text(left-13,y+17,j,13,MUTED,"middle"))
            for f,measure in enumerate(row):
                x=left+24*f
                if j in reachable[f]:
                    color="#%02x%02x%02x"%strength_color(measure)
                    parts.append(rect(x,y,22,22,color))
                else:
                    parts.append(rect(x,y,22,22,"#eaedea"))
                    parts.append(text(x+11,y+16,"—",12,"#7f8986","middle"))
        parts.append(text(left+72,255,"可达 "+str(sum(len(group) for group in reachable))+" / 36",14,INK,"middle",True))
    parts.append(text(12,286,"灰格横线：结构上无路。绿色越深，实测变化越大；颜色按 log(1 + 100D) 映射。",14,MUTED))
    return svg("冻结权重、替换一帧输入，在三种开关下测量六乘六响应",302,parts)


def render(directory, book_root=None):
    data=json.loads((directory/"temporal.json").read_text(encoding="utf-8"))
    for name,content in [("channel-routes",feature_sheet(data)),("frame-response",reach_sheet(data))]:
        (directory/(name+".svg")).write_text(content,encoding="utf-8")
    here=Path(__file__).parent
    template=(here/"temporal_ui.html").read_text(encoding="utf-8")
    for tag,filename in [("__OPS__","hint_network.js"),("__PREDICT__","temporal_network.js")]:
        template=template.replace(tag,(here/filename).read_text(encoding="utf-8"))
    root=book_root if book_root is not None else directory.parent
    for marker,file,label in (("__BOOK__","video-diffusion-guide.pdf","讲义第 63 页"),
                              ("__HOME__","index.html","回到离线学习包")):
        relative=Path(os.path.relpath(root/file,directory)).as_posix()
        fragment="#page=63" if marker=="__BOOK__" else ""
        link=f'<a href="{escape(relative,quote=True)}{fragment}">{label}</a>' if (root/"index.html").is_file() else ""
        template=template.replace(marker,link)
    payload={k:v for k,v in data.items() if k!="edits"}
    template=template.replace("__DATA__",json.dumps(payload,separators=(",",":")))
    (directory/"temporal-shift.html").write_text(template,encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--switches",default="111",help="从入口到解码器的三处移位开关，如 100")
    parser.add_argument("--layer",type=int,default=0)
    parser.add_argument("--frame",type=int,default=2)
    parser.add_argument("--channel",type=int,default=2)
    parser.add_argument("--row",type=int,default=5)
    parser.add_argument("--col",type=int,default=7)
    parser.add_argument("--swap",type=int,choices=range(6),default=0)
    parser.add_argument("--render",action="store_true",help="重画两幅插图与离线网页")
    args=parser.parse_args()
    data=json.loads((args.directory/"temporal.json").read_text(encoding="utf-8"))
    try:
        cell=address(data,args.switches,args.layer,args.frame,args.channel,args.row,args.col)
    except ValueError as error:
        parser.error(str(error))
    if args.render:
        render(args.directory)
    print(json.dumps({"cell":cell,"response_to_swapped_frame":differences(data["runs"][args.switches])[args.swap]},
                     indent=2))
