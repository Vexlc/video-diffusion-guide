"""用 Python 标准库从五帧 ControlVideo 实测记录复算 CFG，绘制离线对照。"""
import argparse
import base64
import json
from pathlib import Path
import struct

from depth_report import differences, png, unpack
from draw import INK, MUTED, TEAL, svg, text

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    ("固定深度为 1，提高文字引导", "w2-d1", "w7.5-d1"),
    ("固定深度为 0，提高文字引导", "w2-d0", "w7.5-d0"),
    ("固定文字引导为 2，打开深度", "w2-d0", "w2-d1"),
    ("固定文字引导为 7.5，打开深度", "w7.5-d0", "w7.5-d1"),
    ("作者界面填 0 或 1", "w0-d1", "w1-d1"),
    ("把负面提示换成空文字", "w7.5-d1", "w7.5-d1-empty"),
)
KEYS = ("ref", "pos", "actual", "residual")


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def guided(ref, pos, w):
    """按作者 FP16 张量的运算顺序，在普通 CPU 上复算一个分量。"""
    if ref is None:
        return pos
    difference = half(pos - ref)
    return half(ref + half(w * difference))


def readout(record, case, step, frame, col, row):
    frames, height, width = record["plane_shape"]
    if not (0 <= frame < frames and 0 <= row < height and 0 <= col < width):
        raise ValueError("选中的坐标超出帧或潜网格范围")
    run = record["runs"][case]
    sample = next((s for s in run["samples"] if s["n"] == step), None)
    if sample is None:
        raise ValueError("可读的步号为 " + ", ".join(str(s["n"]) for s in run["samples"]))
    offset = 2 * ((frame * height + row) * width + col)
    values = {k: struct.unpack_from("<e", unpack(sample[k]), offset)[0] for k in KEYS if k in sample}
    if "ref" in values:
        values["difference"] = half(values["pos"] - values["ref"])
        values["scaled_difference"] = half(run["w"] * values["difference"])
    values["calculated"] = guided(values.get("ref"), values["pos"], run["w"])
    if values["calculated"] != values["actual"]:
        raise ValueError("复算值与记录中的采样器输入不符")
    return {"step": step, "noise_level": sample["tick"], "branches": sample["batch"], **values}


def data_for_page(record):
    width, height = record["size"]
    data = {k: record[k] for k in ("seed", "steps", "plane_shape", "size", "prompt", "negative")}
    data["maps"] = [png(unpack(frame), width, height, 1) for frame in record["maps"]]
    data["runs"] = {}
    for key, run in record["runs"].items():
        samples = []
        for sample in run["samples"]:
            item = {k: v for k, v in sample.items() if k not in KEYS}
            for field in KEYS:
                if field in sample:
                    item[field] = base64.b64encode(unpack(sample[field])).decode("ascii")
            if "ref" in sample:
                positions = struct.iter_unpack("<e", unpack(sample["pos"]))
                references = struct.iter_unpack("<e", unpack(sample["ref"]))
                delta = b"".join(struct.pack("<e", p - r) for (p,), (r,) in zip(positions, references))
                item["delta"] = base64.b64encode(delta).decode("ascii")
            samples.append(item)
        data["runs"][key] = {"w": run["w"], "scale": run["scale"], "reference": run["reference"],
                             "film": [png(unpack(f), width, height, 3) for f in run["film"]], "samples": samples}
    data["pairs"] = []
    for label, a, b in PAIRS:
        rows = [differences(unpack(x), unpack(y)) for x, y in zip(record["runs"][a]["film"], record["runs"][b]["film"])]
        data["pairs"].append({"label": label, "a": a, "b": b, "scores": rows})
    return data


def gallery(data):
    def picture(x, y, w, uri):
        return f'<image x="{x}" y="{y}" width="{w}" height="{w * .75}" href="{uri}"/>'
    parts = [text(192, 22, "文字引导 w = 2", 17, TEAL, "middle", True),
             text(381, 22, "文字引导 w = 7.5", 17, TEAL, "middle", True)]
    for scale, top in ((0, 34), (1, 167)):
        parts.extend([text(11, top + 56, "深度关闭" if scale == 0 else "深度开启", 16, INK, bold=True),
                      text(12, top + 80, f"λ = {scale}", 16, MUTED)])
        for w, left in ((2, 110), (7.5, 299)):
            parts.append(picture(left, top, 164, data["runs"][f"w{w}-d{scale}"]["film"][2]))
    parts.extend([text(564, 64, "给模型的深度图", 16, INK, "middle", True),
                  picture(493, 80, 144, data["maps"][2]), text(565, 214, "近处亮，远处暗", 15, MUTED, "middle"),
                  text(565, 239, "每列共用这张图", 15, MUTED, "middle"),
                  text(565, 264, "λ = 0 时修正归零", 14, MUTED, "middle")])
    return svg("同一噪声生成的四幅第 2 帧与实际深度条件", 298, parts)


def render(record, out, *, course=False):
    data = data_for_page(record)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg-grid.svg").write_text(gallery(data), encoding="utf-8")
    template = Path(__file__).with_name("cfg_ui.html").read_text(encoding="utf-8")
    nav = ('<nav><a href="../index.html">离线首页</a><a href="../page-index.html#p15">第 15 页的公式和代码</a>'
           '<a href="index.html">追踪一次深度修正</a></nav>') if course else ""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", r"\u003c")
    (out / "guidance.html").write_text(template.replace("__NAV__", nav).replace("__DATA__", payload), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "depth-lab/cfg.json")
    parser.add_argument("--case", default="w7.5-d1", help="如 w2-d1、w7.5-d0、w0-d1、w7.5-d1-empty")
    parser.add_argument("--step", type=int, default=0, help="从 0 起数，可选首次、中间或末次预测")
    parser.add_argument("--frame", type=int, default=2)
    parser.add_argument("--col", type=int, default=16)
    parser.add_argument("--row", type=int, default=12)
    parser.add_argument("--render", type=Path, help="在指定目录生成可离线双击的网页与图")
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if args.case not in record["runs"]:
        parser.error("可用组合为 " + ", ".join(record["runs"]))
    try:
        answer = readout(record, args.case, args.step, args.frame, args.col, args.row)
    except ValueError as problem:
        parser.error(str(problem))
    print(json.dumps(answer, ensure_ascii=False, indent=2))
    if args.render:
        render(record, args.render)
