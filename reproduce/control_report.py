"""把实验 J 的实测帧、梯度与全组成绩写成离线图解。只需标准库。"""

import argparse
import json
from pathlib import Path

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, line, rect, svg, text
from pixel_report import contact, png_uri


def gates():
    parts = [text(16, 30, "一个采样步中，两个尺度分别接收一次修正", 18, TEAL, bold=True)]
    for top, name, shape in [(57, "细尺度", "12 × 16，16 通道"), (156, "粗尺度", "6 × 8，32 通道")]:
        parts += [text(16,top+18,name,15,TEAL,bold=True), text(16,top+41,shape,13,MUTED),
                  rect(143,top,132,51,LIGHT,7),text(209,top+30,"冻结基座特征",15,INK,"middle",True),
                  text(290,top+33,"+",22,TEAL,"middle",True),
                  rect(309,top,182,51,"#f7e9df",7),text(400,top+22,"可训练副本的修正",15,RUST,"middle",True),
                  text(400,top+42,"1 × 1 出口 × 强度",13,RUST,"middle"),
                  text(506,top+32,"→",20,TEAL,"middle",True),
                  rect(529,top,117,51,LIGHT,7),text(588,top+32,"送往解码器",14,TEAL,"middle",True)]
    parts += [line(16,133,644,133,GRID),text(16,243,"两行都读同一段六帧影片。从左向右算，每个位置的各个通道分别相加。",14)]
    return svg("细尺度和粗尺度的两条加法连接", 261, parts)


def selected_guide(examples, case, mode):
    frames = examples["guide"][case ^ 1 if mode=="wrong" else case]
    black = [[0 for _ in row] for row in frames[0]]
    return [black if (mode=="first" and time>0) or (mode=="ends" and time not in (0, 5)) else frame
            for time, frame in enumerate(frames)]


def sensitivity(scores):
    rows = [("off", "双出口 × 0"), ("half", "双出口 × 0.5"), ("dense", "双出口 × 1"),
            ("double", "双出口 × 2"), ("fine", "只留细尺度"), ("coarse", "只留粗尺度")]
    parts = [text(13, 27, "给足六帧路标，固定噪声，只改出口系数", 17, TEAL, bold=True),
             text(585, 55, "中间误差 / 格", 13, MUTED, "middle")]
    for index, (name, label) in enumerate(rows):
        item = scores[name]
        top = 66+index*32
        color = TEAL if index<4 else RUST
        parts += [text(14,top+14,label,14),rect(130,top,318,16,LIGHT,3),
                  rect(130,top,318*item["exact"]/96,16,color,3),
                  text(462,top+14,f'{item["exact"]} / 96',14,color,bold=True),
                  text(585,top+14,f'{item["middle_l1"]:.3f}',14,INK,"middle")]
    parts += [line(12, 187, 642, 187, GRID), text(13, 286, "条长 = 六帧全部命中。右列只对帧 1–4 算平均位置误差，越小越好。", 14, MUTED)]
    return svg("控制强度与单独关闭出口的严格命中数和中间帧误差", 302, parts)


def render(data, output):
    output.mkdir(parents=True, exist_ok=True)
    ex = data["examples"]
    figures = {"gates": gates(), "scale": sensitivity(data["scores"]),
               "pair": contact("同一起点、同一份噪声，分别要求往左和往右", [
                   ("共用的首帧提示", selected_guide(ex, 0, "first")), ("只给首帧 / 左题", ex["first"][0]),
                   ("只给首帧 / 右题", ex["first"][1]), ("给两端 / 左题", ex["ends"][0]),
                   ("给两端 / 右题", ex["ends"][1])]),
               "wrong": contact("保持原题向左，只把送进支路的路标改成向右", [
                   ("原题参考", ex["truth"][0]), ("向左的六个点", ex["guide"][0]),
                   ("喂向左点的结果", ex["dense"][0]), ("换成向右的点", selected_guide(ex, 0, "wrong")),
                   ("喂向右点的结果", ex["wrong"][0])])}
    for name, figure in figures.items():
        (output/(name+".svg")).write_text(figure, encoding="utf-8")
    frames = {name:[[png_uri(frame) for frame in clip] for clip in clips] for name, clips in ex.items()
              if name!="opposite"}
    blank = png_uri([[0 for _ in row] for row in ex["guide"][0][0]])
    scores = {name:{key:value for key,value in item.items() if key!="centers"} for name,item in data["scores"].items()}
    payload = json.dumps({"frames":frames, "scores":scores, "blank":blank}, separators=(",", ":"))
    template = Path(__file__).with_name("control_ui.html").read_text(encoding="utf-8")
    hint_link = ""
    if (output/"probe.json").is_file():
        from hint_report import render as render_probe
        render_probe(output)
        hint_link = '<p><a href="edit-points.html">自己移动提示点，比较一次预测与两层修正 →</a></p>'
    template = template.replace("__HINT_LINK__", hint_link)
    (output/"index.html").write_text(template.replace("__CONTROL_PAYLOAD__", payload), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("numbers", type=Path)
    args = parser.parse_args()
    render(json.loads(args.numbers.read_text(encoding="utf-8")), args.numbers.parent)
