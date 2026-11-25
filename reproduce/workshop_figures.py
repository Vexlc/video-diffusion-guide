"""把真实的实验结果排成可打印的图。"""

import math

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, circle, grayscale_uri, line, rect, svg, text, tiny_car
from workshop_models import decode
from reverse_paths import ROUTES


def path_line(points, color, width=2, dash="none"):
    pairs = " ".join(f"{left:.2f},{top:.2f}" for left, top in points)
    return f'<polyline points="{pairs}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'


def ticket():
    parts = [text(15, 26, "一张训练题：只看运动系数这一维", 17, TEAL, bold=True)]
    start, end, time = -0.5, 1.1, 0.25
    middle = (1-time)*start + time*end
    scale = lambda value: 65 + (value+0.6)*295
    parts += [line(36, 107, 628, 107, GRID, 3)]
    for value, title, color, top in [(start, "起点 z = −0.5", MUTED, 150), (middle, "此刻 x = −0.1", TEAL, 75), (end, "终点 y = 1.1", RUST, 150)]:
        left = scale(value)
        parts.extend([circle(left, 107, 6, color), line(left, 107, left, top+(-15 if top>107 else 7), color),
                      text(left, top, title, 15, color, "middle", True)])
    parts.extend([rect(15, 177, 305, 75, LIGHT, 5), rect(335, 177, 305, 75, "#f9eee4", 5),
                  text(30, 201, "题面 / 预测器能看", 16, TEAL, bold=True), text(30, 231, "位置 −0.1，时间 0.25", 17),
                  text(351, 201, "答案 / 只在训练时对分", 16, RUST, bold=True), text(351, 231, "速度 1.1 − (−0.5) = 1.6", 17)])
    return svg("流匹配单题：起点、插值点、终点与速度标签", 265, parts)


def histogram(points):
    counts = [0]*22
    for center, speed in points:
        counts[min(21, max(0, int((speed+2.2)*5)))] += 1
    return counts


def motion_distribution(fm, stage, clip=0):
    parts = [text(14, 27, "256 段生成片段的运动系数（纵轴：段数）", 18, TEAL, bold=True),
             text(15, 53, "实柱：模型生成　　虚线：独立目标样本　　负值向左，正值向右", 14)]
    counts, target = histogram(stage["points"]), histogram(fm["target"])
    peak = max(max(target), *(max(histogram(entry["points"])) for entry in fm["stages"]))
    project = lambda count: 208-133*count/peak
    for count in (0, peak):
        parts += [line(53, project(count), 637, project(count)), text(45, project(count)+5, count, 12, MUTED, "end")]
    width = 26
    for index, count in enumerate(counts):
        left = 59+width*index
        parts.append(rect(left, project(count), width-3, 208-project(count), TEAL))
    parts.append(path_line([(70+width*index, project(count)) for index, count in enumerate(target)], RUST, 2.3, "5 3"))
    for tick in (-2, -1, 0, 1, 2):
        left = 57+(tick+2.2)/.2*width
        parts += [line(left, 208, left, 214), text(left, 231, tick, 12, MUTED, "middle")]
    parts += [text(15, 266, f"放大第 {clip+1} 个样本：用两个数画成八帧", 17, TEAL, bold=True)]
    track = decode(stage["points"][clip])
    for index, value in enumerate(track):
        left = 16+79*index
        parts.extend([rect(left, 280, 74, 43, LIGHT, 3), line(left, 309, left+74, 309),
                      tiny_car(left+37+15*value, 308, TEAL, 0.9), text(left+37, 341, f"帧 {index}", 12, MUTED, "middle")])
    parts.append(text(15, 365, "端格含越界值。先看分布是否分成两团，再看一辆车，不要只挑最好看的一段。", 13))
    return svg("生成的运动分布和一个八帧样本", 378, parts)


def velocity_profile(fm, index):
    stages = fm["stages"]
    parts = [text(13, 25, f"固定生成时间 τ = {(index+1)*0.25:.2f}，沿同一条街查看路牌", 17, TEAL, bold=True)]
    highest = max(2, math.ceil(max(abs(value) for stage in stages for value in stage["profiles"][index])))
    pos = lambda value: 335 + value*130
    height = lambda value: 131-75*value/highest
    for value in (-highest, 0, highest):
        parts += [line(51, height(value), 616, height(value)), text(45, height(value)+4, value, 12, MUTED, "end")]
    parts += [line(pos(0), 46, pos(0), 210), text(14, 48, "速度", 13)]
    for value in (-2, -1, 0, 1, 2):
        parts.append(text(pos(value), 228, value, 12, MUTED, "middle"))
    parts.append(text(615, 228, "位置", 12))
    for number, stage in enumerate(stages):
        color, dash = [(MUTED, "3 3"), (RUST, "7 3"), (TEAL, "none")][number]
        points = [(pos((sample-42)/20), height(value)) for sample, value in enumerate(stage["profiles"][index])]
        parts.extend([path_line(points, color, 2.1, dash), line(54+number*193, 254, 77+number*193, 254, color, 2.2, dash),
                      text(83+number*193, 258, f"{stage['count']:,} 道题", 13)])
    return svg("速度随位置的剖面，不同训练量的对照", 274, parts)


def attention_map(weights, ambiguous):
    parts = [text(16, 27, "本帧的一块，去上一帧的三个位置找线索", 18, TEAL, bold=True),
             text(16, 51, "每个格子是一票的份额；每一行的未舍入值加起来是 1。", 14)]
    names = ["车", "车", "路"] if ambiguous else ["树", "车", "路"]
    for col, meaning in enumerate(names):
        parts.append(text(225+154*col, 86, f"位置 {col+1} · {meaning}", 15, MUTED, "middle"))
    for row, name in enumerate(("车", "路", "树")):
        top = 98+47*row
        parts.append(text(19, top+27, "查询："+name, 16, INK, bold=True))
        for col, value in enumerate(weights[row]):
            shade = tuple(round(240+(limit-240)*value) for limit in (0, 97, 86))
            fill = "#%02x%02x%02x" % shade
            parts += [rect(151+154*col, top, 148, 41, fill, 3),
                      text(225+154*col, top+26, f"{100*value:.1f}%", 16, "white" if value>.67 else INK, "middle", True)]
    return svg("时序注意力权重矩阵", 245, parts)


def photo(pixels, left, top, width):
    return (f'<image x="{left}" y="{top}" width="{width}" height="{width*30/48}" '
            f'style="image-rendering:pixelated" href="{grayscale_uri(pixels)}"/>')


def control_contact(arrays):
    parts = []
    for index, (key, caption) in enumerate([("truth", "① 干净参考"), ("observed", "② 带噪输入"),
                                            ("mask", "③ 对齐的遮罩"), ("base", "④ 只用老模型"),
                                            ("aligned", "⑤ 加对齐遮罩"), ("late", "⑥ 加上一帧遮罩")]):
        left, top = 12+217*(index%3), 8+160*(index//3)
        parts.extend([text(left, top+16, caption, 16, TEAL, bold=True), photo(arrays[key], left, top+26, 191)])
    return svg("合成影片的输入、灰度遮罩和三种还原估计", 322, parts)


def changed_plate(arrays, metrics):
    region = [1 if first != second else -1 for first, second in zip(arrays["mask"], arrays["late_mask"])]
    parts = [text(15, 28, "先找两张遮罩哪里不同，再看那里的误差", 17, TEAL, bold=True)]
    for index, (pixels, title) in enumerate([(arrays["mask"], "本帧遮罩"), (arrays["late_mask"], "上一帧遮罩"), (region, "白色 = 发生变化")]):
        left = 15+216*index
        parts += [photo(pixels, left, 44, 180), text(left+90, 178, title, 14, INK, "middle")]
    parts.append(text(15, 215, "同一张错位结果，两种误差统计", 17, TEAL, bold=True))
    maximum = max(metrics["all"]["late"], metrics["changed"]["late"], .001)
    for row, (key, caption) in enumerate([("all", "全片所有像素"), ("changed", "仅遮罩变化像素")]):
        top = 236+40*row
        value = metrics[key]["late"]
        parts += [text(14, top+16, caption, 14), rect(158, top, 368*value/maximum, 22, TEAL if not row else RUST, 2),
                  text(540, top+17, f"{value:.3f}", 16, INK, bold=True)]
    return svg("全片误差与遮罩变化区域误差", 313, parts)


def route_counts(records):
    parts = [text(15, 28, "同一模型，每种设置重复 96 次", 18, TEAL, bold=True)]
    for index, (key, title) in enumerate([("DDIM_fixed", "DDIM · 固定起点"), ("DDPM_fixed", "DDPM · 固定起点"),
                                          ("DDIM_fresh", "DDIM · 每次换起点"), ("DDPM_fresh", "DDPM · 每次换起点")]):
        left, top = 11+(index%2)*323, 46+(index//2)*166
        parts.extend([rect(left, top, 315, 156, LIGHT, 4), text(left+11, top+22, title, 15, TEAL, bold=True)])
        for route, count in enumerate(records[key]["counts"]):
            baseline = top + 41 + route*27
            parts.extend([text(left+11, baseline+8, ROUTES[route], 13), rect(left+115, baseline-3, count*1.52, 12, RUST if index%2 else TEAL),
                          text(left+281, baseline+8, count, 14, INK, "end", True)])
    return svg("四种设置下，九十六个样本落入四条候选路线的计数", 382, parts)


def sampler_film(records):
    parts = [text(15, 26, "把同一份雪花复印三份，再分别走完 24 步", 18, TEAL, bold=True)]
    for number, sampler in enumerate(("DDIM", "DDPM")):
        top = 48+number*150
        parts.append(text(13, top+16, sampler+"　/　相同起点，分别重跑", 15, TEAL, bold=True))
        for repeat, case in enumerate(records[sampler+"_fixed"]["cases"][:3]):
            row = top+23+repeat*37
            parts.append(text(14, row+21, f"第 {repeat+1} 次", 12))
            for frame, value in enumerate(case["track"]):
                left = 74+72*frame
                parts.extend([rect(left, row, 66, 30, LIGHT, 2), line(left, row+22, left+66, row+22),
                              tiny_car(left+33+16*value, row+21, TEAL if not number else RUST, .74)])
    parts.append(text(15, 365, "每一行都是一次完整生成；从左到右是视频帧，不是去噪步骤。", 14))
    return svg("相同起点下，两种采样器各三次生成结果", 382, parts)
