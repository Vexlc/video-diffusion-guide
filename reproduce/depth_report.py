"""不依赖模型与图像库，从 ControlVideo 记录计算 RGB 差异并核对一次真实特征相加。"""
import argparse
import base64
import json
from pathlib import Path
import struct
import zlib

from draw import INK, MUTED, TEAL, rect, svg, text
from hint_report import signed, tile, values

ROOT=Path(__file__).resolve().parents[1]
KEYS=("h","r","joined")
LIMITS=(1.3,.11,1.3)


def unpack(s):
    return zlib.decompress(base64.b64decode(s,validate=True))


def png(raw,w,h,channels):
    def chunk(kind,data):
        return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data))
    stride=w*channels
    if len(raw)!=stride*h or channels not in (1,3):
        raise ValueError("读到的画面尺寸或通道数不匹配")
    scan=b"".join(b"\0"+raw[k:k+stride] for k in range(0,len(raw),stride))
    binary=(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2 if channels==3 else 0,0,0,0))+
            chunk(b"IDAT",zlib.compress(scan))+chunk(b"IEND",b""))
    return "data:image/png;base64,"+base64.b64encode(binary).decode("ascii")


def differences(a,b):
    """以 8 位 RGB 的三个通道为单位取绝对差，同时统计至少一通道不同的像素。"""
    if len(a)!=len(b) or not len(a) or len(a)%3:
        raise ValueError("两幅 RGB 图需有相同的尺寸")
    d=bytes(abs(x-y) for x,y in zip(a,b))
    changed=sum(bool(r or g or b) for r,g,b in struct.iter_unpack("BBB",d))
    return {"mae":sum(d)/len(d),"changed":changed}


def readout(trace,frame,col,row):
    """读 CFG 正面文字分支、最高分辨率的第一个出口、通道 0。"""
    frames,h,w=trace["h"]["shape"]
    if not (0<=frame<frames and 0<=col<w and 0<=row<h):
        raise ValueError("选中的特征坐标超出已记录范围")
    index=(frame*h+row)*w+col
    vals={k:struct.unpack_from("<f",base64.b64decode(trace[k]["data"]),4*index)[0] for k in KEYS}
    vals["fp16_sum"]=struct.unpack("<e",struct.pack("<e",vals["h"]+vals["r"]))[0]
    return vals


def pixel(raw,w,col,row,channels):
    p=(row*w+col)*channels
    return list(raw[p:p+channels])


def payload(record):
    w,h=record["size"]
    data={"maps":{},"runs":{},"delta":{},"size":[w,h],"seed":record["seed"],"steps":record["steps"]}
    for label,images in record["maps"].items():
        data["maps"][label]=[png(unpack(frame),w,h,1) for frame in images]
    for label,run in record["runs"].items():
        data["runs"][label]={"film":[png(unpack(frame),w,h,3) for frame in run["film"]],"trace":run["trace"]}
    for scale in (0,1):
        a,b=[record["runs"][name+str(scale)]["film"] for name in ("A","B")]
        rows=[differences(unpack(left),unpack(right)) for left,right in zip(a,b)]
        if any(v["mae"]!=mean or v["changed"]!=count for v,mean,count in
               zip(rows,record["delta"][str(scale)]["mae"],record["delta"][str(scale)]["changed"])):
            raise ValueError("逐像素复算与采集时的成绩不一致")
        data["delta"][str(scale)]=rows
    return data


def gallery(data):
    parts=[text(12,24,"只改帧 2 的深度图，再看五帧成片",17,TEAL,bold=True)]
    film=lambda key:data["runs"][key]["film"]
    rows=[("深度图 A",data["maps"]["A"]),("深度图 B",data["maps"]["B"]),("关闭 · A",film("A0")),
          ("开启 · A",film("A1")),("开启 · B",film("B1"))]
    for f in range(5):
        parts.append(text(149+111*f,47,"帧 "+str(f),14,MUTED,"middle"))
    for i,(label,frames) in enumerate(rows):
        top=55+81*i
        parts.append(text(10,top+40,label,15,INK,bold=True))
        for f,uri in enumerate(frames):
            left=98+111*f
            parts.append(f'<image x="{left}" y="{top}" width="102" height="76.5" href="{uri}"/>')
            if i==1 and f==2:
                parts.append(f'<rect x="{left}" y="{top}" width="102" height="76.5" fill="none" stroke="#ae4f2c" stroke-width="2"/>')
    return svg("两排条件图与三排实际 ControlVideo 成片，橙框中的深度球向右移了 48 像素",464,parts)


def feature_sheet(data):
    col,row,frame=16,12,2
    parts=[text(12,25,"第一次预测　·　帧 2 的同一格",17,TEAL,bold=True)]
    for i,title in enumerate(("主干特征 h","支路修正 r","解码器读到 h′")):
        parts.append(text(161+191*i,51,title,16,TEAL,"middle",True))
    for j,label in enumerate(("A","B")):
        trace=data["runs"][label+"1"]["trace"]
        top=61+142*j
        parts.append(text(10,top+55,"条件 "+label,15,INK,bold=True))
        vals=readout(trace,frame,col,row)
        for i,(key,limit) in enumerate(zip(KEYS,LIMITS)):
            left=83+191*i
            planes=values(trace[key]);w,h=32,24
            cells=planes[frame*w*h:(frame+1)*w*h]
            parts.append(tile(left,top,w,h,156,[signed(x,limit) for x in cells]))
            unit=156/w
            parts.append(f'<rect x="{left+col*unit}" y="{top+row*unit}" width="{unit}" height="{unit}" fill="none" stroke="#111" stroke-width="1.6"/>')
            parts.append(text(left+78,top+134,f"{vals[key]:+.5f}".replace("-","−"),15,INK,"middle",True))
    parts.append(text(12,355,"蓝色为负，橙色为正。左右两列共用 ±1.30 的色阶，中列单独使用 ±0.11。",14,MUTED))
    return svg("两条条件下冻结主干的值相同，支路修正经真实加法进入解码器",370,parts)


def render(record,out,*,course=False):
    data=payload(record)
    out.mkdir(parents=True,exist_ok=True)
    for name,content in (("depth-films",gallery(data)),("depth-sum",feature_sheet(data))):
        (out/(name+".svg")).write_text(content,encoding="utf-8")
    nav=('<nav><a href="../index.html">离线首页</a><a href="../page-index.html#p85">成片的公式与代码</a>'
         '<a href="../page-index.html#p86">特征加法的代码</a></nav>') if course else ""
    html=Path(__file__).with_name("depth_ui.html").read_text(encoding="utf-8")
    (out/"index.html").write_text(html.replace("__NAV__",nav).replace("__DATA__",json.dumps(data,separators=(",",":"))),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record",type=Path,default=ROOT/"depth-lab/record.json")
    parser.add_argument("--frame",type=int,choices=range(5),default=3)
    parser.add_argument("--scale",type=int,choices=(0,1),default=1)
    parser.add_argument("--feature",nargs=2,type=int,metavar=("COL","ROW"),default=[16,12])
    parser.add_argument("--pixel",nargs=2,type=int,metavar=("COL","ROW"),default=[152,102])
    parser.add_argument("--render",type=Path,help="另生成可双击的离线页面与两幅图")
    args=parser.parse_args()
    record=json.loads(args.record.read_text(encoding="utf-8"))
    w,h=record["size"];col,row=args.pixel
    if not(0<=col<w and 0<=row<h): parser.error("像素坐标超出画幅")
    answer={};films=[]
    for label in ("A","B"):
        run=record["runs"][label+str(args.scale)]
        raw=unpack(run["film"][args.frame]);films.append(raw)
        answer[label]={"depth":pixel(unpack(record["maps"][label][args.frame]),w,col,row,1)[0],
                       "rgb":pixel(raw,w,col,row,3),"feature":readout(run["trace"],args.frame,*args.feature)}
    answer["diff"]=differences(*films)
    print(json.dumps(answer,ensure_ascii=False,indent=2))
    if args.render:render(record,args.render)
