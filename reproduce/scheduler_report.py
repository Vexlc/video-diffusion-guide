"""用 Python 标准库从真实调度器的记录复算一步，生成可离线点选的图。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from draw import INK, MUTED, RUST, TEAL, rect, svg, text
from hint_report import values
from pixel_report import bitmap, contact

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {"ddim": 522, "flow": 30}


def recompute(kind, row, x, v):
    """在同一个输入上，分别求原设置和改变一项设置后的下一格。"""
    if kind == "ddim":
        a, b, c, d = row["coef"]
        clean = a*x - b*v
        noise = b*x + a*v
        limited = min(1., max(-1., clean))
        return c*clean + d*noise, c*limited + d*noise
    width = row["following"] - row["sigma"]
    return x + width*(-v), x + width*v


def address(shape, index):
    position = []
    for size in reversed(shape):
        index, offset = divmod(index, size)
        position.append(offset)
    return list(reversed(position))


def read_cell(data, kind, step, index):
    case = data[kind]
    if not 1 <= step <= len(case["steps"]) or not 0 <= index < math.prod(case["shape"]):
        raise ValueError("次数或坐标超出范围")
    row = case["steps"][step-1]
    x, v, accepted, alternate = [values(row[key])[index] for key in ("x","v","next","alternate")]
    a, b = recompute(kind, row, x, v)
    return {"第几次更新":step, "坐标（从零数）":address(case["shape"], index), "调度器时刻":row["t"],
            "输入":x,"网络输出":v,"手算":a,"库返回":accepted,"改动后手算":b,"改动后库返回":alternate}


def pixel_plate(data):
    row = data["ddim"]["steps"][12]
    a,b,c,d = row["coef"]
    x,v,y,z = [values(row[k])[384:576] for k in ("x","v","next","alternate")]
    clean = [a*p-b*q for p,q in zip(x,v)]
    diff = [abs(p-q) for p,q in zip(y,z)]
    images = [
        ("输入 x", x, "显示范围 −2 到 2", lambda p:(p+2)/4),
        ("猜干净亮度 ŷ", clean, "显示范围 −2 到 2", lambda p:(p+2)/4),
        ("猜出界的像素", clean, "白色为 |ŷ| > 1", lambda p:float(abs(p)>1)),
        ("下一格的差", diff, "从黑 0 到白 0.089", lambda p:p/max(diff)),
    ]
    parts=[text(12,25,"第 13 次更新，只看帧 2 的同一位置",17,TEAL,bold=True)]
    for i,(caption,plane,legend,to_gray) in enumerate(images):
        left=13+i*163
        parts.append(text(left,53,caption,15,INK,bold=True))
        gray=[round(255*max(0,min(1,to_gray(p)))) for p in plane]
        parts.append(bitmap([gray[r:r+16] for r in range(0,192,16)],left,65,144))
        parts.append(f'<rect x="{left+90}" y="137" width="9" height="9" fill="none" stroke="#ff7445" stroke-width="2.5"/>')
        parts.append(text(left,195,legend,12,MUTED))
    parts += [rect(12,211,636,39,"#edf4ef",4),
              text(22,236,"橙框在第 8 行、第 10 列。框内的干净猜测为 −1.118，打开裁剪后取 −1。",14,INK)]
    return svg("猜测越界后，同一个像素怎样改变",260,parts)


def flow_plate(data):
    clips = data["flow"]["films"]
    return contact("固定雪花与首帧点 (7, 3)，读解码后最亮的 3 × 3 方块", [
        ("① 讲义欧拉", clips[0]), ("② 库接收 −u", clips[1]), ("③ 库接收 +u", clips[2])])


def render(data, out, *, course=False):
    out.mkdir(parents=True, exist_ok=True)
    (out/"record.json").write_text(json.dumps(data,separators=(",",":")),encoding="utf-8")
    for name, figure in (("ddim-clip",pixel_plate), ("flow-sign",flow_plate)):
        (out/(name+".svg")).write_text(figure(data),encoding="utf-8")
    template = Path(__file__).with_name("scheduler_ui.html").read_text(encoding="utf-8")
    template = template.replace("__COURSE_STYLE__", "" if course else "[data-course]{display:none}")
    (out/"index.html").write_text(template.replace("__TRACES__",json.dumps(data,separators=(",",":"))),
                                  encoding="utf-8")


def check_files(data, root):
    for name, digest in data["files"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"调度器记录与 {name} 不配套")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record",type=Path,default=ROOT/"scheduler-lab/record.json")
    parser.add_argument("--kind",choices=("ddim","flow"),default="ddim")
    parser.add_argument("--step",type=int,choices=range(1,25),default=13,metavar="1..24")
    parser.add_argument("--index",type=int,help="按行展平的像素或潜码地址，从 0 数")
    parser.add_argument("--out",type=Path,help="按这份记录重画网页和两张 SVG")
    args=parser.parse_args()
    data=json.loads(args.record.read_text(encoding="utf-8"))
    index=EXAMPLES[args.kind] if args.index is None else args.index
    if not 0 <= index < math.prod(data[args.kind]["shape"]):
        parser.error("地址超出当前张量的范围")
    print(json.dumps(read_cell(data,args.kind,args.step,index),ensure_ascii=False,indent=2))
    if args.out:
        render(data,args.out,course=(args.out.parent/"reproduce/scheduler_bridge.py").is_file())
