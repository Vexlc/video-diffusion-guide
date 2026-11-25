"""把 Flow Matching 作者库的两个端点交换，检查题面和速度标签。"""
import argparse
import hashlib
import inspect
import json
from math import isclose
from pathlib import Path

REVISION = "11568d37f8d5a080e12aa7b5305d9c35ae07d136"
SOURCE = f"https://github.com/facebookresearch/flow_matching/blob/{REVISION}/flow_matching/path/affine.py#L87-L92"
RECORD = Path(__file__).with_name("author_endpoints.json")
ORDERS = (("雪花 → 原片", -1.0, 0.8), ("原片 → 雪花", 0.8, -1.0))


def author_samples():
    """在 CPU 上调用作者的直线路径，同一批时间点各做两次。"""
    import torch
    from flow_matching.path import CondOTProbPath

    path = CondOTProbPath()
    times = torch.tensor([k/100 for k in range(101)], dtype=torch.float64, device="cpu")
    orders = []
    for label, start, end in ORDERS:
        x_0 = torch.full_like(times[:, None], start)
        x_1 = torch.full_like(times[:, None], end)
        answer = path.sample(x_0=x_0, x_1=x_1, t=times)
        orders.append({"label":label, "start":start, "end":end,
                       "x_t":answer.x_t[:, 0].tolist(), "dx_t":answer.dx_t[:, 0].tolist()})
    files = [Path(inspect.getfile(type(item))) for item in (path, path.scheduler)]
    return {"source_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "times":times.tolist(), "orders":orders}


def load_record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def check(record):
    if record["times"] != [k/100 for k in range(101)] or len(record["orders"]) != 2:
        raise ValueError("缺少进度或端点顺序")
    for order, (label, start, end) in zip(record["orders"], ORDERS):
        if (order["label"],order["start"],order["end"]) != (label,start,end):
            raise ValueError("端点顺序不一致")
        if len(order["x_t"]) != 101 or len(order["dx_t"]) != 101:
            raise ValueError("库输出的扫描范围不完整")
        for t,x,v in zip(record["times"],order["x_t"],order["dx_t"]):
            if not (isclose(x,(1-t)*start+t*end,abs_tol=1e-12) and isclose(v,end-start,abs_tol=1e-12)):
                raise ValueError(f"进度 {t:g} 的读数与直线插值不一致")


def figure(record, step=50):
    from draw import INK, MUTED, RUST, TEAL, circle, line, svg, text

    pieces = [text(11,20,"作者库的调用顺序",15,MUTED),text(293,20,"题面所在位置",15,MUTED,"middle"),
              text(460,20,"x_t",16,INK,bold=True),text(550,20,"dx_t",16,INK,bold=True)]
    lo,hi = 177,411
    for i,order in enumerate(record["orders"]):
        y = 47+39*i
        color = (TEAL,RUST)[i]
        x = order["x_t"][step]
        speed = order["dx_t"][step]
        px = lo+(hi-lo)*(x+1)/1.8
        direction = 1 if speed>0 else -1
        pieces.extend([text(11,y+5,order["label"],16,color,bold=True),line(lo,y,hi,y,"#a9beb8",1.5),
                       line(lo,y-4,lo,y+4,MUTED),line(hi,y-4,hi,y+4,MUTED)])
        # A dot marks the value; a short arrow identifies the direction, not the speed magnitude.
        pieces.extend([f'<g data-dot="{i}" transform="translate({px:.6f} {y})">',circle(0,0,5,color),
                       line(0,0,23*direction,0,color,2.7),line(23*direction,0,17*direction,-4,color,2),
                       line(23*direction,0,17*direction,4,color,2),'</g>',
                       f'<g data-x="{i}">'+text(460,y+6,f"{x:+.3f}",17,INK,bold=True)+'</g>',
                       text(550,y+6,f"{speed:+.3f}",17,color,bold=True)])
    pieces.extend([text(lo,111,"−1.0",13,MUTED,"middle"),text(hi,111,"+0.8",13,MUTED,"middle")])
    return svg("两次调用的中间位置和速度；点标位置，短箭头标速度方向",119,pieces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=50, choices=range(101), metavar="0..100")
    parser.add_argument("--official", action="store_true", help="本机调用已安装的 flow_matching 和 PyTorch")
    parser.add_argument("--capture", type=Path, help="调用作者库并保存全部 202 次读数")
    args = parser.parse_args()
    record = author_samples() if args.official or args.capture else load_record()
    check(record)
    if args.capture:
        args.capture.parent.mkdir(exist_ok=True,parents=True)
        args.capture.write_text(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
    print(f"t = {record['times'][args.step]:g}")
    for order in record["orders"]:
        print(f"{order['label']}   x_t = {order['x_t'][args.step]:+.3f}   dx_t = {order['dx_t'][args.step]:+.3f}")
