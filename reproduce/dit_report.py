"""以 Python 标准库将视频 Transformer 的真实神经网络结果做成图册。"""

import argparse
import json
from pathlib import Path

from draw import INK, LIGHT, MUTED, RUST, TEAL, line, rect, svg, text
from pixel_report import contact, png_uri


def patch_map(data):
    parts=[text(14,26,"先按时间组，再按行、左右块，把潜格排成一列",17,TEAL,bold=True)]
    for g in range(3):
        left=20+215*g
        parts.append(text(left,57,f"组 {g} · 电影帧 {2*g}、{2*g+1}",15,INK,bold=True))
        for r in range(3):
            for b in range(2):
                ident=6*g+2*r+b
                x,y=left+85*b,70+35*r
                color=RUST if ident==8 else TEAL
                parts += [rect(x,y,80,30,"#f7e8d8" if ident==8 else LIGHT,3),
                          line(x+40,y,x+40,y+5,"#a0bcb0",1.5),
                          line(x+40,y+25,x+40,y+30,"#a0bcb0",1.5),
                          text(x+40,y+21,f"块 {ident}",14,color,"middle",True)]
    values=data["probe"]["tokens"][8]
    parts += [text(17,210,"展开块 8",16,RUST,bold=True),text(149,210,"通道 0 的两格",14,MUTED),text(397,210,"通道 1 的两格",14,MUTED)]
    for j,value in enumerate(values):
        x=143+j*119
        parts += [rect(x,222,106,40,LIGHT,4),text(x+53,248,f"{value:+.3f}",16,INK,"middle",True)]
    return svg("十八个视频潜块的地址与第八号块的四个实测输入",281,parts)


def tint(weight):
    return "#"+"".join(f"{round(a+(b-a)*weight):02x}" for a,b in zip((240,246,237),(0,98,87)))


def attention_map(data):
    weights=data["probe"]["answers"]["left"]["maps"][0][0][8]
    parts=[text(14,27,"块 8 发起查询　·　第 1 层，第 1 头，条件向左",17,TEAL,bold=True)]
    for g in range(3):
        x=17+216*g
        subtotal=sum(weights[6*g:6*g+6])
        parts += [text(x,58,f"帧组 {g}　合计 {100*subtotal:.1f}%",15,INK,bold=True)]
        for r in range(3):
            for b in range(2):
                i=6*g+2*r+b
                left,top=x+b*92,69+r*53
                v=weights[i]
                fg="#ffffff" if v>.57 else INK
                parts += [rect(left,top,87,48,tint(v),4),text(left+9,top+16,f"键 {i}",12,fg),
                          text(left+76,top+38,f"{100*v:.1f}%",15,fg,"end",True)]
    parts.append(text(16,260,"数字是这一个查询对 18 个键的份额。色阶固定为 0% 至 100%，逐格四舍五入。",14,MUTED))
    return svg("展开一个真实注意力头的一整行权重",278,parts)


def intervention(data):
    p=data["probe"]
    parts=[text(12,26,"保留帧组 1 的输入，换掉两边，只测中间六个 token",17,TEAL,bold=True)]
    for i in range(6):
        parts.append(text(179+83*i,56,f"块 {6+i}",14,MUTED,"middle"))
    peak=max(max(p["by_token"]["full"]),1e-8)
    for row,(key,caption,color) in enumerate([("full","允许跨组",TEAL),("local","封住跨组",RUST)]):
        y=72+row*63
        parts.append(text(12,y+26,caption,15,color,bold=True))
        for i,change in enumerate(p["by_token"][key]):
            x=141+i*83
            parts += [rect(x,y,76,42,LIGHT,3),text(x+38,y+25,f"{change:.3f}",15,color,"middle",True),
                      rect(x+4,y+33,68*min(1,change/peak),5,color,2)]
    parts.append(text(12,218,"每格 = 四个速度分量在换片前后的平均绝对变化。读数越大，答案改变越多。",14,MUTED))
    return svg("在同一个中间状态替换首尾组前后的速度变化",235,parts)


def score_chart(data):
    parts=[text(12,25,"只改 CFG 系数，128 份噪声各下两道相反指令",17,TEAL,bold=True),
           text(116,58,"首末位移朝指定方向",14,TEAL,bold=True),
           text(388,58,"六帧都中同一条指定方向的合法路线",14,RUST,bold=True)]
    for i,key in enumerate(data["scales"]):
        row=data["stats"][key]
        y=75+32*i
        parts.append(text(14,y+14,"w = "+key,14,INK))
        for x,field,color in [(116,"direction",TEAL),(388,"strict",RUST)]:
            val=row[field]
            parts += [rect(x,y,152,16,LIGHT,2),rect(x,y,152*val/(2*data["pairs"]),16,color,2),
                      text(x+160,y+14,f'{val} / {2*data["pairs"]}',13,color,bold=True)]
    parts.append(text(12,294,"两栏用同一刻度。方向正确的一段片，仍可能在中间跳格、变形或走错行。",14,MUTED))
    return svg("六种引导强度下宽松与严格的成功计数",311,parts)


def render(data,output):
    output.mkdir(parents=True,exist_ok=True)
    ex=data["examples"]
    pictures={"tokens":patch_map(data),"attention":attention_map(data),"intervention":intervention(data),
              "guidance":contact("预先选定的第一份雪花，比较两道指令的实际生成片",[
                  ("w=0　两题相同",ex["0"][0]),("w=1　向左",ex["1"][0]),("w=1　向右",ex["1"][1]),
                  ("w=4　向左",ex["4"][0]),("w=4　向右",ex["4"][1])]),"scores":score_chart(data)}
    for name,diagram in pictures.items():
        (output/(name+".svg")).write_text(diagram,encoding="utf-8")
    payload={k:data[k] for k in ("stats","probe","pairs","steps","scales")}
    payload["films"]={name:[[png_uri(frame) for frame in video] for video in videos] for name,videos in ex.items()}
    template=Path(__file__).with_name("dit_ui.html").read_text(encoding="utf-8")
    link = ' <a href="message.html">只改一个值，追踪一条跨帧消息</a>' if (output/"message.html").is_file() else ""
    template = template.replace("__MESSAGE_LINK__", link)
    (output/"index.html").write_text(template.replace("__DIT_DATA__",json.dumps(payload,separators=(",",":"))),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("numbers",type=Path)
    args=parser.parse_args()
    render(json.loads(args.numbers.read_text(encoding="utf-8")),args.numbers.parent)
