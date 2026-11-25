"""用训练记录和解码器权重排出实验 L；生成网页只需要标准库。"""

import argparse
import json
import math
from pathlib import Path

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, line, rect, svg, text
from pixel_report import bitmap, contact, png_uri


def shade(value,limit):
    base = (246,246,235)
    end = (0,105,95) if value>=0 else (173,73,44)
    weight = min(1,abs(value)/limit)
    return "#"+"".join(f"{round(a+(b-a)*weight):02x}" for a,b in zip(base,end))


def latent_grid(data,limit):
    parts = [text(12,24,"同一段片的六帧像素与 72 个潜变量均值",17,TEAL,bold=True)]
    for g,group in enumerate(data["latent"][0]):
        x = 18+215*g
        parts.append(text(x,54,f"第 {2*g}、{2*g+1} 帧",15,INK,bold=True))
        for k in range(2):
            parts.append(bitmap(data["examples"]["truth"][0][2*g+k],x+k*83,65,77))
        parts += [line(x,135,x+159,135,TEAL,2),text(x+80,155,"共同编码 ↓",14,TEAL,"middle")]
        for c,matrix in enumerate(group):
            left = x+c*84
            for i,row in enumerate(matrix):
                for j,value in enumerate(row):
                    parts.append(rect(left+j*17,169+i*17,16,16,shade(value,limit)))
            parts.append(text(left+33,240,f"通道 {c}",13,INK,"middle"))
    parts += [text(15,271,"负值",13,RUST),text(245,271,"正值",13,TEAL)]
    for j in range(37):
        parts.append(rect(51+5*j,258,6,13,shade((j/18-1)*limit,limit)))
    parts.append(text(291,271,f"色阶 −{limit} 到 +{limit}，米白为零。每小格存一个数。",13,MUTED))
    return svg("每两帧共同编码成两个三行四列的特征图",288,parts)


def performance(data):
    metrics = data["stats"]["generation"]
    parts = [text(14,26,"同一批 256 份初始噪声，同一个轻正则解码器",17,TEAL,bold=True),
             text(148,57,"六帧车心全中 ↑",15,TEAL,bold=True),text(406,57,"整片像素 MSE ↓",15,RUST,bold=True)]
    for i,(key,label) in enumerate([("prior","直接抽 VAE 先验"),("start","FM 起点解码"),("flow1","FM · 1 步"),("flow4","FM · 4 步"),("flow24","FM · 24 步")]):
        s = metrics[key]
        y = 74+35*i
        parts += [text(12,y+14,label,14),rect(148,y,150,15,LIGHT),rect(148,y,150*s["route_hits"]/256,15,TEAL),
                  text(307,y+14,f'{s["route_hits"]} / 256',14,TEAL,bold=True),
                  rect(406,y,156,15,LIGHT),rect(406,y,156*s["clip_mse"]/max(.2,max(v["clip_mse"] for v in metrics.values())),15,RUST),
                  text(571,y+14,f'{s["clip_mse"]:.4f}',14,RUST,bold=True)]
    parts.append(text(14,275,"左栏逐段判定整条路线，右栏取每段与最近合法模板的误差再平均。",14,MUTED))
    return svg("直接抽先验与潜空间流匹配的整片表现",292,parts)


def render(data,output):
    output.mkdir(parents=True,exist_ok=True)
    ex = data["examples"]
    limit = max(1,math.ceil(max(abs(v) for clip in data["latent"] for group in clip for ch in group for row in ch for v in row)))
    def film(title,rows):
        return contact(title,[(name,ex[key][0]) for name,key in rows])
    base,hybrid = ex["light_mean"][0],ex["hybrid"][0]
    difference = [[[abs(a-b) for a,b in zip(row_a,row_b)] for row_a,row_b in zip(frame_a,frame_b)]
                  for frame_a,frame_b in zip(base,hybrid)]
    figures = {"grid":latent_grid(data,limit),
        "posterior":film("固定同一段留出影片，只换编码方式或训练好的权重",[
            ("干净参考","truth"),("轻正则 / 均值","light_mean"),("轻正则 / 抽一次","light_draw"),
            ("强正则 / 均值","strong_mean"),("强正则 / 抽一次","strong_draw")]),
        "splice":contact("把 B 的中间一组潜变量交给 A 的解码器输入",[
            ("A 的均值重建",base),("B 的均值重建",ex["donor"][0]),("A、B、A 混合",hybrid),
            ("混合与 A 的差",difference)]),
        "sample":film("固定第一份噪声，分别求解后只解码一次",[
            ("标准化噪声起点","start"),("欧拉 1 步","flow1"),("欧拉 4 步","flow4"),("欧拉 24 步","flow24")]),
        "scores":performance(data)}
    for key,value in figures.items():
        (output/(key+".svg")).write_text(value,encoding="utf-8")
    payload = {key:data[key] for key in ("latent","first_sigma","first_eps","decoder","stats","hold_count")}
    payload.update(limit=limit,frames={key:[[png_uri(frame) for frame in clip] for clip in clips] for key,clips in ex.items()})
    template = Path(__file__).with_name("latent_ui.html").read_text(encoding="utf-8")
    template=template.replace("__PAIR_DECODER__",Path(__file__).with_name("pair_decoder.js").read_text(encoding="utf-8"))
    (output/"index.html").write_text(template.replace("__LATENT_DATA__",json.dumps(payload,separators=(",",":"))),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("numbers",type=Path)
    args = parser.parse_args()
    render(json.loads(args.numbers.read_text(encoding="utf-8")),args.numbers.parent)
