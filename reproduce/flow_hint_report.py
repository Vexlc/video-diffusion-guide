"""仅用标准库读实际的条件流轨迹，绘制 PDF 插图、离线播放器和单格核对记录。"""
import argparse
import base64
import json
import struct
from pathlib import Path

from draw import INK, MUTED, RUST, TEAL, line, rect, svg, text
from pixel_report import bitmap
import motion_readout


def number(tensor,*address):
    offset=0
    if len(address)!=len(tensor["shape"]):
        raise ValueError("地址维数与张量不同")
    for i,size in zip(address,tensor["shape"]):
        if not 0<=i<size:
            raise ValueError("坐标超出记录范围")
        offset=offset*size+i
    return struct.unpack_from("<f",base64.b64decode(tensor["data"]),4*offset)[0]


def cell(data,strength,case,n,token,component,block):
    """在同一状态读基座速度和接入支路后的速度，再核对一次欧拉更新。"""
    rec=data["records"][str(float(strength))]
    args=(case,n,token,component)
    x=number(rec["x"],*args)
    v=number(rec["v"],*args)
    before,r,after=[number(rec["features"],case,n,block,token,k) for k in range(3)]
    return {"x":x,"v":v,"base_same_x":number(rec["free"],*args),"delta_x":v/data["steps"],
            "next":number(rec["x"],case,n+1,token,component),"before":before,"residual":r,"after":after}


def film(rec,case,n):
    raw=base64.b64decode(rec["pixels"])
    start=(case*25+n)*6*192
    if not (0<=case<12 and 0<=n<=24):
        raise ValueError("影片编号或进度超出范围")
    return [[list(raw[start+(f*12+r)*16:start+(f*12+r+1)*16]) for r in range(12)] for f in range(6)]


def film_sheet(data):
    parts=[text(12,24,"同一份噪声，24 步后解码的六帧",17,TEAL,bold=True)]
    for f in range(6):
        parts.append(text(184+83*f,49,f"帧 {f}",13,MUTED,"middle"))
    for row,(scale,case) in enumerate(((0.,1),(1.,1),(1.,3),(1.,2))):
        ex=data["examples"][case]
        pics=film(data["records"][str(scale)],case,24)
        top=58+63*row
        parts.extend((text(12,top+20,f"点 (7, {ex['point'][1]})",15,INK,bold=True),
                      text(12,top+42,f"{ex['label']}　λ = {scale:g}",14,TEAL)))
        for f,pic in enumerate(pics):
            parts.append(bitmap(pic,150+83*f,top,68))
    return svg("真实解码影片，依次改变位置支路开关、首帧点的行、标签方向",309,parts)


def first_films(data):
    parts=[text(10,23,"同一份噪声，两次生成",17,TEAL,bold=True)]
    rec=data["records"]["1.0"]
    for row,case in enumerate((1,3)):
        top=35+58*row
        y=data["examples"][case]["point"][1]
        parts.append(text(12,top+27,f"起点 (7,{y})",15,INK,bold=True))
        for f,pic in enumerate(film(rec,case,24)):
            left=123+87*f
            parts.append(bitmap(pic,left,top,62))
            if row==1:
                parts.append(text(left+31,top+63,f"帧 {f}",12,MUTED,"middle"))
    return svg("同一噪声与向右标签，第 3 行和第 7 行出发的两段六帧影片",164,parts)


def wiring(data):
    parts=[text(12,25,"两条路径同时读当前潜码、进度和方向",17,TEAL,bold=True),
           text(12,78,"冻结基座",15,TEAL,bold=True),text(12,188,"训练支路",15,RUST,bold=True)]
    for i,x in enumerate((105,345)):
        join=x+158
        parts.extend((rect(x,55,85,36,"#e6f1ea",4),text(x+42,79,f"块 {i+1}",16,TEAL,"middle",True),
                      line(x+85,73,join-18,73,TEAL,2),rect(join-17,57,34,32,"#d7ece2",4),
                      text(join,80,"+",24,TEAL,"middle",True),
                      rect(x,164,85,36,"#f6e9dc",4),text(x+42,188,f"副本 {i+1}",16,RUST,"middle",True),
                      line(x+85,182,join,182,RUST,2),line(join,182,join,147,RUST,2),
                      rect(join-52,118,104,29,"#f6e9dc",3),text(join,138,"零初始化出口",14,RUST,"middle",True),
                      line(join,118,join,91,RUST,2),
                      f'<path d="M{join-4} 98 l4 -7 l4 7" fill="none" stroke="{RUST}" stroke-width="2"/>',
                      text(join+10,111,f"λr{i+1}",14,RUST)))
        if i==0:
            for y,color in ((73,TEAL),(182,RUST)):
                start=join+17 if y==73 else join
                parts.extend((line(start,y,342,y,color,2),
                              f'<path d="M336 {y-4} l6 4 l-6 4" fill="none" stroke="{color}" stroke-width="2"/>'))
    parts.extend((line(520,73,555,73,TEAL,2),text(553,78,"→ 速度头",15,TEAL,bold=True),
                  text(12,231,"亮点图投影后送入第一个副本。副本之间传递隐藏特征，各自的出口向上交修正。",15,MUTED)))
    return svg("两个冻结 Transformer 块及对应的可训练副本，每个副本单独通过零初始化出口接入基座",247,parts)


def score_figure(data):
    parts=[text(12,26,"同一批留出题，各强度独立生成",17,TEAL,bold=True)]
    for index,(key,title) in enumerate((("first","首帧中心命中"),("direction","首末位移方向正确"),("strict","六帧中心全部命中"))):
        left=10+215*index
        parts.append(text(left,61,title,15,INK,bold=True))
        for i,scale in enumerate(data["scales"]):
            row=data["scores"][str(float(scale))]
            y=80+34*i
            parts.extend((text(left,y+11,f"λ {scale:g}",13,MUTED),rect(left+37,y,109,12,"#e5eae4",2),
                          rect(left+37,y,109*row[key]/row["n"],12,TEAL if i else RUST,2),
                          text(left+155,y+12,row[key],14,INK,bold=True)))
    parts.append(text(12,196,"每栏分母 224 = 28 条未训练路线 × 同一组 8 份起始噪声。",14,MUTED))
    return svg("首帧、方向、整段的独立评分",210,parts)


def failure_figure(data):
    error=data["first_error"]
    if error is None:
        return svg("留出题全部六帧命中",70,[text(12,40,"本次留出题的六帧中心全部命中。",17,TEAL)])
    parts=[text(12,25,"第一段未全中的留出片 · 六帧逐格查",17,TEAL,bold=True)]
    for i,(pic,target,pred) in enumerate(zip(error["film"],error["reference"],error["measured"])):
        left=18+i*107
        parts.append(bitmap(pic,left,40,86))
        color=RUST if target!=pred else INK
        parts.extend((text(left,127,f"应 ({target[0]}, {target[1]})",13,MUTED),
                      text(left,149,f"得 ({pred[0]}, {pred[1]})",14,color,bold=target!=pred)))
    return svg("未经挑选的第一段未逐帧命中影片和六个实测中心",161,parts)


def render(data,out):
    out.mkdir(parents=True,exist_ok=True)
    for name,img in (("first-films",first_films(data)),("combined-films",film_sheet(data)),("branch-blocks",wiring(data)),
                     ("combined-scores",score_figure(data)),("first-miss",failure_figure(data))):
        (out/(name+".svg")).write_text(img,encoding="utf-8")
    motion_readout.render(data,out)
    template=Path(__file__).with_name("flow_hint_ui.html").read_text(encoding="utf-8")
    (out/"index.html").write_text(template.replace("__FLOW_HINT__",json.dumps(data,ensure_ascii=False,
                                  separators=(",",":"),allow_nan=False)),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--case",type=int,default=1,choices=range(12))
    parser.add_argument("--step",type=int,default=12,choices=range(24))
    parser.add_argument("--token",type=int,default=8,choices=range(18))
    parser.add_argument("--component",type=int,default=0,choices=range(4))
    parser.add_argument("--block",type=int,default=0,choices=range(2))
    parser.add_argument("--strength",type=float,choices=(0.,.5,1.),default=1.)
    parser.add_argument("--render",action="store_true")
    args=parser.parse_args()
    data=json.loads((args.directory/"numbers.json").read_text(encoding="utf-8"))
    if args.render:
        render(data,args.directory)
    print(json.dumps(cell(data,args.strength,args.case,args.step,args.token,args.component,args.block),indent=2))
