"""不依赖 PyTorch：把浮点评分与实际输出的显示像素做成图册。"""

import argparse
import base64
import json
import struct
import zlib
from pathlib import Path

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, circle, line, rect, svg, text
from workshop_figures import path_line


def png_uri(rows):
    def block(name, payload):
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name+payload))
    header = struct.pack(">IIBBBBB", len(rows[0]), len(rows), 8, 0, 0, 0, 0)
    stream = b"".join(b"\x00"+bytes(row) for row in rows)
    png = b"\x89PNG\r\n\x1a\n"+block(b"IHDR", header)+block(b"IDAT", zlib.compress(stream))+block(b"IEND", b"")
    return "data:image/png;base64,"+base64.b64encode(png).decode()


def bitmap(rows, left, top, width):
    return (f'<image x="{left}" y="{top}" width="{width}" height="{width*len(rows)/len(rows[0])}" '
            f'style="image-rendering:pixelated" href="{png_uri(rows)}"/>')


def contact(title, groups, selected=None):
    pieces = [text(12, 26, title, 17, TEAL, bold=True)]
    for frame in range(6):
        pieces.append(text(187+82*frame, 52, f"帧 {frame}", 12, MUTED, "middle"))
    for index, (caption, frames) in enumerate(groups):
        top = 62 + index*68
        pieces.append(text(12, top+31, caption, 14, INK, bold=True))
        for frame, pixels in enumerate(frames):
            left = 150+82*frame
            pieces.append(bitmap(pixels, left, top, 74))
            if selected == frame:
                pieces.append(f'<rect x="{left-2}" y="{top-2}" width="78" height="60" fill="none" stroke="{RUST}" stroke-width="2"/>')
    return svg(title, 66+len(groups)*68, pieces)


def evidence_plate(data, index):
    probe, examples = data["probe"], data["examples"]
    return contact("同一段已知短片，同一份雪花；看两种网络能恢复什么",
                   [("干净参考", probe["truth"][index]), ("给网络的输入", probe["noisy"][index]),
                    ("逐帧估计", examples["single"]["probe"][index]),
                    ("跨帧估计", examples["joint"]["probe"][index])])


def stages_plate(data, index):
    stages = data["examples"]["joint"]
    return contact("固定初始噪声，每行是一次生成过程的快照", [
        ("最初的雪花", stages["start"][index]), ("第 1 次猜原片", stages["0"][index]),
        ("第 13 次猜原片", stages["12"][index]), ("第 24 次猜原片", stages["23"][index])])


def comparison_plate(data, index):
    examples = data["examples"]
    return contact("从同一份纯噪声出发；这里不存在待找回的原片", [
        ("未训练 / 跨帧", examples["untrained"][index]), ("已训练 / 逐帧", examples["single"]["23"][index]),
        ("已训练 / 跨帧", examples["joint"]["23"][index])])


def neighbor_plate(data, index):
    examples = data["examples"]["neighbor"]
    parts = []
    for row, (title, frames) in enumerate([("原来的六帧", data["probe"]["noisy"][index]),
                                           ("五帧换自别片", examples["foreign"][index])]):
        top = row*63+13
        parts.append(text(12, top+31, title, 14, bold=True))
        for frame, grid in enumerate(frames):
            left = 151+82*frame
            parts.append(bitmap(grid, left, top, 70))
            if frame==2:
                parts.append(f'<rect x="{left-2}" y="{top-2}" width="74" height="57" fill="none" stroke="{RUST}" stroke-width="2"/>')
    parts.append(text(12, 163, "只核对橙框位置：输入不变，答案会不会变？", 16, TEAL, bold=True))
    for column, (caption, grid) in enumerate([("真实第 2 帧", data["probe"]["truth"][index][2]),
            ("跨帧 / 原邻帧", examples["intact"][index]), ("跨帧 / 换邻帧", examples["swapped"][index])]):
        left = 38+209*column
        parts += [bitmap(grid, left, 179, 151), text(left+76, 316, caption, 14, INK, "middle")]
    return svg("保持第2帧不变仅更换邻帧的干预", 334, parts)


def loss_plate(data):
    history = data["history"]
    end = max(1, history[-1]["step"])
    parts = [text(12, 25, "用同一批留出噪声题检查训练进度", 17, TEAL, bold=True)]
    windows = [(0, min(100, end), "开始的变化"), (min(250, end), end, "放大末段差距")]
    series = [("single", "逐帧网络", RUST, "6 3"), ("joint", "跨帧网络", TEAL, "none")]
    for column, (start, stop, title) in enumerate(windows):
        subset = [entry for entry in history if start <= entry["step"] <= stop] or history[-1:]
        peak = max(entry[key] for entry in subset for key in ("single", "joint"))*1.08
        left = 48+320*column
        pos_step = lambda step: left+245*(step-start)/max(1, stop-start)
        pos_loss = lambda value: 173-92*value/max(1e-9, peak)
        parts.append(text(left, 57, title+" · MSE", 14, TEAL, bold=True))
        for mark in (0, peak/2, peak):
            parts += [line(left, pos_loss(mark), left+245, pos_loss(mark)),
                      text(left-5, pos_loss(mark)+4, f"{mark:.2f}", 12, MUTED, "end")]
        for step in sorted({start, stop}):
            parts.append(text(pos_step(step), 194, f"{step} 步", 12, MUTED, "middle"))
        for key, caption, color, dash in series:
            points = [(pos_step(entry["step"]), pos_loss(entry[key])) for entry in subset]
            parts += [path_line(points, color, 2, dash)]
            parts.extend(circle(horizontal, vertical, 2.5, color) for horizontal, vertical in points)
    for column, (key, caption, color, dash) in enumerate(series):
        parts += [line(160+column*158, 226, 185+column*158, 226, color, 2, dash),
                  text(192+column*158, 230, caption, 13)]
    return svg("早期与后期分别缩放的留出集 v 预测均方误差", 243, parts)


def render(data, output):
    output.mkdir(parents=True, exist_ok=True)
    names = ["probe", "sampling", "models", "neighbors"]
    functions = [evidence_plate, stages_plate, comparison_plate, neighbor_plate]
    panels = [{name: function(data, index) for name, function in zip(names, functions)} for index in range(8)]
    figures = {**panels[0], "loss": loss_plate(data)}
    for name, figure in figures.items():
        (output/(name+".svg")).write_text(figure, encoding="utf-8")
    previews = [[png_uri(frame) for frame in clip] for clip in data["examples"]["joint"]["23"]]
    payload = {"panels": panels, "previews": previews, "scores": data["scores"], "setup": data["setup"], "loss": figures["loss"]}
    template = Path(__file__).with_name("pixel_ui.html").read_text(encoding="utf-8")
    recording_path = output/"clock-data.js"
    clock_link = ""
    if recording_path.is_file():
        from clock_report import render as render_clocks
        render_clocks(recording_path, output)
        clock_link = '<p><a href="two-clocks.html">逐格查看六帧与 24 次更新，验算一个像素 →</a></p>'
    if (output/"temporal.json").is_file():
        from temporal_report import render as render_temporal
        render_temporal(output)
        clock_link += '<p><a href="temporal-shift.html">拆开三处跨帧连接，追踪一个通道 →</a></p>'
    template = template.replace("__CLOCK_LINK__", clock_link)
    (output/"index.html").write_text(template.replace("__PIXEL_PAYLOAD__", json.dumps(payload, ensure_ascii=False,
        separators=(",", ":")).replace("</script", "<\\/script")), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("numbers", type=Path)
    args = parser.parse_args()
    render(json.loads(args.numbers.read_text(encoding="utf-8")), args.numbers.parent)
