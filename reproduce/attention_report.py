"""在随包的真实神经网络记录上，复算一个视频块收到的消息及一次干预。只用标准库。"""
import argparse
import json
from math import exp, fsum, sqrt
from pathlib import Path

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, line, rect, svg, text
from pixel_report import bitmap, png_uri

ROOT = Path(__file__).resolve().parents[1]


def similarity(row, query, source, mode):
    q = row["queries"][query]
    scores = [fsum(x*y for x, y in zip(q, k)) / sqrt(len(q)) for k in row["keys"]]
    allowed = [mode == "full" or j//6 == query//6 for j in range(len(scores))]
    peak = max(score for score, use in zip(scores, allowed) if use)
    masses = [exp(score - peak) if use else 0. for score, use in zip(scores, allowed)]
    total = fsum(masses)
    return {"dot": fsum(x*y for x, y in zip(q, row["keys"][source])), "score": scores[source],
            "peak": peak, "mass": masses[source], "total": total, "weight_from_qk": masses[source]/total}


def readout(data, query=8, source=13, mode="full"):
    if not (0 <= source < 18 and 0 <= query < 18):
        raise ValueError("块号范围是 0 到 17")
    row = data["runs"][mode]
    weights, values = row["weights"][query], row["values"]
    products = [a*v for a, v in zip(weights, values)]
    return {**similarity(row, query, source, mode), "weight": weights[source], "value": values[source], "contribution": products[source],
            "group_sums": [fsum(products[i:i+6]) for i in (0, 6, 12)], "sum": fsum(products),
            "received": row["messages"][query], "received_plus_one": row["edited_messages"][source][query],
            "velocity": row["velocity"][query], "velocity_plus_one": row["edited_velocity"][source][query]}


def film_diagram(data):
    reading = readout(data)
    parts = []
    for group in range(3):
        left = 13 + 218*group
        parts.append(text(left+93, 21, f"时间组 {group} · 帧 {2*group}、{2*group+1}", 16, INK, "middle", True))
        for f in range(2):
            parts.append(bitmap(data["frames"][2*group+f], left+94*f, 30, 89))
        for offset in range(6):
            index = 6*group+offset
            x, y = left+18+76*(offset%2), 109+26*(offset//2)
            bg = TEAL if index == 8 else RUST if index == 13 else LIGHT
            parts.extend([rect(x, y, 69, 23, bg, 3),
                          text(x+34, y+17, f"块 {index}", 14, "white" if index in (8, 13) else MUTED, "middle", index in (8, 13))])
        total = reading["group_sums"][group]
        parts.append(text(left+93, 210, f"六块合计 {total:+.4f}".replace("-", "−"), 16, TEAL, "middle", True))
    parts += [line(11, 224, 649, 224, GRID),
              text(15, 250, "块 13 发送", 17, RUST, bold=True),
              text(115, 250, f'{reading["weight"]:.4f} × ({reading["value"]:.4f}) ≈ {reading["contribution"]:.4f}'.replace("-", "−"), 17, INK),
              text(451, 250, "→", 19, TEAL, "middle"), text(474, 250, "块 8 收进这份消息", 17, TEAL, bold=True)]
    return svg("六帧原片、三个时间组和一次实测的跨组取值。绿色块接收，橙色块发送。", 262, parts)


def render(data, out, *, course=False):
    out.mkdir(parents=True, exist_ok=True)
    (out/"message-route.svg").write_text(film_diagram(data), encoding="utf-8")
    nav = ('<nav><a href="../index.html">离线首页</a><a href="../page-index.html#p13">第 13 页的公式与源码</a>'
           '<a href="index.html">查看其他注意力头</a></nav>') if course else ""
    template = Path(__file__).with_name("attention_ui.html").read_text(encoding="utf-8")
    compact = {"frames": [png_uri(frame) for frame in data["frames"]], "runs": data["runs"]}
    (out/"message.html").write_text(template.replace("__NAV__", nav).replace(
        "__RECORD__", json.dumps(compact, separators=(",", ":"))), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT/"dit-lab/message.json")
    parser.add_argument("--query", type=int, choices=range(18), default=8, metavar="0..17")
    parser.add_argument("--source", type=int, choices=range(18), default=13, metavar="0..17")
    parser.add_argument("--mode", choices=("full", "local"), default="full")
    parser.add_argument("--out", type=Path, help="另生成独立网页与讲义配图")
    args = parser.parse_args()
    data = json.loads(args.record.read_text(encoding="utf-8"))
    print(json.dumps(readout(data, args.query, args.source, args.mode), ensure_ascii=False, indent=2))
    if args.out:
        render(data, args.out)
