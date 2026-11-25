"""沿封面影片的一格潜码，复算训练题和一次真实的生成更新。默认只用标准库。"""
import argparse
import base64
import json
import hashlib
import math
import struct
from pathlib import Path

from draw import INK, MUTED, RUST, TEAL, circle, line, rect, svg, text


ROOT = Path(__file__).resolve().parents[1]
TOKEN, COMPONENT = 7, 0
STEPS = 24


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def capture(root):
    """用配套的编码器制作留出题，并在直线上的题面查询实际网络。"""
    import torch
    from dit_lab import RIGHT, patchify
    from flow_hint import SEED, hint_map, initial_states, load_trained, setup, velocity
    from pixel_lab import paint

    setup("cpu", SEED)
    vae, base, branch, saved = load_trained(root, root/"flow-hint/branch.pt", "cpu")
    archive = read(root/"flow-hint/numbers.json")
    hashes = {**archive["base_hashes"], "branch": archive["branch_sha256"]}
    for key, name in (("vae", "latent-lab/weights.pt"), ("dit", "dit-lab/weights.pt"), ("branch", "flow-hint/branch.pt")):
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != hashes[key]:
            raise ValueError("网络权重与录像使用的权重不同")
    rows = torch.tensor([3, 7])
    columns = torch.tensor([7, 7])
    hints = hint_map(torch.stack((columns, rows), 1))
    with torch.no_grad():
        clips = paint(columns, rows, torch.ones_like(rows))
        y = (vae.encode(clips)[0] - saved["mean"]) / saved["spread"]
        z = initial_states(1, "cpu").expand_as(y)
        predictions = []
        for k in range(STEPS + 1):
            t = k/STEPS
            velocity_hat = velocity(base, branch, (1-t)*z + t*y, torch.full((2,1),t),
                                    torch.full_like(rows, RIGHT), hints, 1.)
            predictions.append(patchify(velocity_hat)[:, TOKEN, COMPONENT])
        guesses = torch.stack(predictions, 1).tolist()
    return {"hashes":hashes, "seed":archive["eval_seed"], "token":TOKEN, "component":COMPONENT,
            "z":patchify(z)[0,TOKEN,COMPONENT].item(),
            "rows":{str(row):{"y":value, "guesses":series} for row,value,series in
                    zip(rows.tolist(), patchify(y)[:,TOKEN,COMPONENT].tolist(), guesses)}}


def series(tensor, case):
    cases, count, tokens, channels = tensor["shape"]
    if not 0 <= case < cases:
        raise ValueError("影片索引超出记录")
    raw = base64.b64decode(tensor["data"])
    return [struct.unpack_from("<f", raw, 4*(((case*count+k)*tokens+TOKEN)*channels+COMPONENT))[0]
            for k in range(count)]


def lesson(root):
    directory = root/"flow-hint"
    archive = read(directory/"numbers.json")
    reference = read(directory/"reading-targets.json")
    if (reference["hashes"] != {**archive["base_hashes"], "branch":archive["branch_sha256"]} or
            (archive["steps"],reference["seed"],reference["token"],reference["component"]) !=
            (STEPS,archive["eval_seed"],TOKEN,COMPONENT)):
        raise ValueError("留出题与生成记录的权重、时间或坐标不对应")
    record = archive["records"]["1.0"]
    pixels = base64.b64decode(record["pixels"])
    size = (STEPS+1)*6*12*16
    result = {}
    for case in (1,3):
        example = archive["examples"][case]
        row = str(example["point"][1])
        x = series(record["x"],case)
        if (not example["held"] or example["label"] != "向右" or
                abs(x[0]-reference["z"]) > 1e-6):
            raise ValueError("样片须对应同一起点噪声和留出的向右影片")
        result[row] = {**reference["rows"][row], "x":x, "v":series(record["v"],case),
                       "centers":record["centers"][case],
                       "film":base64.b64encode(pixels[case*size:(case+1)*size]).decode("ascii")}
    return result


def readings(case, k):
    """沿同一生成进度读取两种训练题面、对应标签和自由生成的下一步。"""
    z, y = case["x"][0], case["y"]
    t = k/STEPS
    a, sigma = math.sin(math.pi*t/2), math.cos(math.pi*t/2)
    return {"k":k, "t":t, "a":a, "sigma":sigma, "z":z, "y":y,
            "diffusion":a*y+sigma*z, "noise_label":z, "linear":(1-t)*z+t*y,
            "velocity_label":y-z, "teacher_prediction":case["guesses"][k],
            "component_squared_error":(case["guesses"][k]-(y-z))**2, "generated":case["x"][k],
            "speed":case["v"][k] if k<STEPS else None, "next":case["x"][k+1] if k<STEPS else None}


def chart(case, selected):
    points = [readings(case,k) for k in range(STEPS+1)]
    keys = (("diffusion","扩散练习",RUST,"5 3"),("linear","直线练习",INK,"2 3"),("generated","模型实走",TEAL,""))
    low = math.floor(min(p[key] for p in points for key,*_ in keys)*2)/2
    high = math.ceil(max(p[key] for p in points for key,*_ in keys)*2)/2
    left, right, top, base = 55, 490, 36, 166
    xp = lambda k: left+(right-left)*k/STEPS
    yp = lambda value: base-(base-top)*(value-low)/(high-low)
    pieces = [text(left,20,"同一格潜码的数值",15,INK,bold=True)]
    for val in (low,(low+high)/2,high):
        pieces += [line(left,yp(val),right,yp(val)),text(left-8,yp(val)+4,f"{val:g}",13,MUTED,"end")]
    for k in (0,12,24):
        pieces.append(text(xp(k),187,f"{k} 步",14,MUTED,"middle"))
    for i,(key,label,color,dash) in enumerate(keys):
        coords = " ".join(f"{xp(p['k']):.2f},{yp(p[key]):.2f}" for p in points)
        pieces += [f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.7" stroke-dasharray="{dash or "none"}"/>',
                   line(518,56+40*i,545,56+40*i,color,3,dash), text(552,61+40*i,label,15,color,bold=True),
                   '<g class="marker-value">'+text(552,78+40*i,f"{points[selected][key]:.3f}",14,color)+'</g>']
    pieces += ['<g class="slice">',line(xp(selected),top,xp(selected),base,MUTED,1,"3 3")]
    for key,_,color,_ in keys:
        pieces.append(circle(xp(selected),yp(points[selected][key]),4,color))
    pieces.append("</g>")
    return svg("扩散练习、直线流匹配练习和模型实际走过的生成轨迹，横轴为生成进度",201,pieces)


def comparison_values(point):
    def number(key, places=3):
        return f"{point[key]:.{places}f}".replace("-", "−")
    return {
        "exercise-x": number("linear"),
        "exercise-speed": number("teacher_prediction"),
        "exercise-use": "平方误差 " + number("component_squared_error", 4),
        "sample-x": number("generated"),
        "sample-speed": number("speed") if point["speed"] is not None else "已到终点",
        "sample-use": "新位置 " + number("next") if point["next"] is not None else "解码六帧",
    }


def comparison_figure(point):
    values = comparison_values(point)
    pieces = [text(139,22,"当前位置 x",16,MUTED),text(303,22,"网络回答的速度",16,MUTED),
              text(491,22,"怎样用这个答案",16,MUTED)]
    for i,(prefix,label,color,tint) in enumerate((("exercise","练习评分",RUST,"#f8efe6"),
                                                  ("sample","实际生成",TEAL,"#e6f1eb"))):
        top = 31+42*i
        pieces.extend([rect(6,top,648,36,tint,3),text(19,top+24,label,17,color,bold=True)])
        for left,suffix in ((139,"x"),(303,"speed"),(491,"use")):
            key = prefix+"-"+suffix
            pieces.append(f'<g data-readout="{key}">'+text(left,top+24,values[key],17,INK,bold=True)+'</g>')
        for left in (242,435):
            pieces.append(text(left,top+24,"→",18,color))
    return svg("在同一进度分别给练习评分和实际生成一步，从左到右读输入、回答与用途",115,pieces)


def render(root, out):
    cases = lesson(root)
    out.mkdir(parents=True,exist_ok=True)
    (out/"reading-paths.svg").write_text(chart(cases["3"],12),encoding="utf-8")
    diagram = comparison_figure(readings(cases["3"],12))
    (out/"reading-uses.svg").write_text(diagram,encoding="utf-8")
    prepared = {}
    for row,case in cases.items():
        points = [readings(case,k) for k in range(STEPS+1)]
        for point in points:
            point["comparison"] = comparison_values(point)
        prepared[row] = {"points":points, "centers":case["centers"],
                         "film":case["film"], "chart":chart(case,12)}
    template = Path(__file__).with_name("reading_ui.html").read_text(encoding="utf-8")
    (out/"reading.html").write_text(template.replace("__COMPARISON__",diagram).replace("__LESSON__",json.dumps(prepared,separators=(",",":"),
                                   ensure_ascii=False,allow_nan=False)),encoding="utf-8")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row",type=int,choices=(3,7),default=3)
    parser.add_argument("--step",type=int,choices=range(STEPS+1),default=12)
    parser.add_argument("--capture",type=Path,help="用 PyTorch 重算并写入指定文件")
    parser.add_argument("--render",action="store_true",help="在 flow-hint 目录重画本课")
    args = parser.parse_args()
    if args.capture:
        args.capture.parent.mkdir(parents=True,exist_ok=True)
        args.capture.write_text(json.dumps(capture(ROOT),ensure_ascii=False,indent=2),encoding="utf-8")
    else:
        if args.render:
            render(ROOT,ROOT/"flow-hint")
        print(json.dumps(readings(lesson(ROOT)[str(args.row)],args.step),ensure_ascii=False,indent=2))
