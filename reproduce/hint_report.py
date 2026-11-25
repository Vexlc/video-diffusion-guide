"""从 PyTorch 保存的特征重建实验 J 的印刷图与可编辑网页。仅需标准库。"""
import argparse
import base64
import json
import struct
import zlib
from pathlib import Path

from draw import INK, MUTED, TEAL, RUST, rect, svg, text


def values(tensor):
    return [item for item, in struct.iter_unpack("<f", base64.b64decode(tensor["data"]))]


def tile(left, top, w, h, display_width, colors):
    def block(kind, data):
        return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data))
    pixels = bytes(component for pixel in colors for component in pixel)
    rows = b"".join(b"\0"+pixels[i:i+3*w] for i in range(0,len(pixels),3*w))
    image = (b"\x89PNG\r\n\x1a\n"+block(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+
             block(b"IDAT",zlib.compress(rows))+block(b"IEND",b""))
    uri = base64.b64encode(image).decode("ascii")
    return (f'<image x="{left}" y="{top}" width="{display_width}" height="{display_width*h/w}" '
            f'style="image-rendering:pixelated" href="data:image/png;base64,{uri}"/>')


def grey(v):
    k = round(127.5*(max(-1,min(1,v))+1))
    return k,k,k


def signed(v, limit):
    end = (13,103,147) if v < 0 else (190,78,26)
    t = min(1,abs(v)/limit)
    return tuple(round(247+t*(c-247)) for c in end)


def film_figure(data):
    a,b = (data["answers"][key] for key in ("left","moved"))
    clean_a,clean_b = values(a["clean"]),values(b["clean"])
    labels = ["A 的提示点","B 的提示点","A 猜的干净片","B 猜的干净片","亮度变化 B − A"]
    parts = [text(12,24,"只在帧 2 挪一个点，再看六帧怎样回应",17,TEAL,bold=True)]
    for row,label in enumerate(labels):
        y = 58+55*row
        parts.append(text(12,y+28,label,15,INK,bold=True))
        for f in range(6):
            x = 148+83*f
            if row==0:
                parts.append(text(x+30,48,f"帧 {f}",13,TEAL if f==2 else MUTED,"middle",f==2))
            if row<2:
                point=(a,b)[row]["points"][f]
                colors=[(238,244,235) if (i%16,i//16)==tuple(point) else (34,55,59) for i in range(192)]
            else:
                va,vb=clean_a[f*192:(f+1)*192],clean_b[f*192:(f+1)*192]
                colors = ([grey(t) for t in (va if row==2 else vb)] if row<4
                          else [signed(v-u,1.25) for u,v in zip(va,vb)])
            parts.append(tile(x,y,16,12,60,colors))
            if f==2:
                parts.append(f'<rect x="{x-2}" y="{y-2}" width="64" height="49" fill="none" stroke="{TEAL}" stroke-width="1.8"/>')
    parts.append(text(12,345,"平均绝对变化",14,INK,bold=True))
    for f in range(6):
        first=f*192
        mae=sum(abs(a-b) for a,b in zip(clean_a[first:first+192],clean_b[first:first+192]))/192
        x=148+83*f
        parts.extend([text(x+30,345,f"{mae:.4f}",14,TEAL,"middle",f==2),
                      rect(x,352,60,5,"#e6eae5"),rect(x,352,60*mae/.08,5,TEAL)])
    parts.append(text(12,385,"差图共用 ±1.25 色标：蓝色变暗，橙色变亮。条长共用 0 到 0.08 的刻度。",14,MUTED))
    return svg("移动一个点前后的六帧、干净估计和有符号差图",401,parts)


def feature_figure(data):
    frame,channel,row,col=2,0,2,3
    first=(frame*32+channel)*48
    def plane(tensor):
        return values(tensor)[first:first+48]
    base=plane(data["fixed"]["low"])
    a,b=(plane(data["answers"][k]["coarse"]) for k in ("left","moved"))
    parts = [text(12,25,"在同一个粗网格上，选通道 0、帧 2",17,TEAL,bold=True)]
    layers=[("基座 h",base),("A 修正 r",a),("B 修正 r",b),
            ("h + A 修正",[x+y for x,y in zip(base,a)]),("h + B 修正",[x+y for x,y in zip(base,b)])]
    for index,(label,cells) in enumerate(layers):
        x=15+index*130
        parts += [text(x+53,56,label,14,INK,"middle",True),tile(x,70,8,6,104,[signed(v,2) for v in cells]),
                  f'<rect x="{x+13*col}" y="{70+13*row}" width="13" height="13" fill="none" stroke="#292326" stroke-width="2.3"/>',
                  text(x+52,171,f"{cells[row*8+col]:+.4f}",15,INK,"middle",True)]
    parts.append(text(15,198,"黑框是第 2 行、第 3 列。共用 ±2 色标，超出范围的值以端点色显示。",14,MUTED))
    return svg("同一粗尺度通道上的基座、修正和逐格相加结果",214,parts)


def render(output):
    data=json.loads((output/"probe.json").read_text(encoding="utf-8"))
    for name, draw in [("hint-move",film_figure),("hint-features",feature_figure)]:
        (output/(name+".svg")).write_text(draw(data),encoding="utf-8")
    source=Path(__file__).parent
    network = (source/"hint_network.js").read_text(encoding="utf-8")
    template = (source/"hint_ui.html").read_text(encoding="utf-8")
    payload = {k:data[k] for k in ("fixed","layers","a","sigma")}
    payload["points"] = data["answers"]["left"]["points"]
    page=template.replace("__NETWORK__",network).replace("__DATA__",json.dumps(payload,separators=(",",":")))
    (output/"edit-points.html").write_text(page,encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    render(parser.parse_args().directory)
