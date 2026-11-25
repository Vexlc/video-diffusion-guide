"""实验 J 的单步探针。用已训练权重导出网页所需的特征、算子权重与对照答案。"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import torch

from clock_report import Recording, little_floats
from control_lab import Branch, controlled
from pixel_lab import TinyVideo, FRAMES, HEIGHT, WIDTH


def tensor_record(tensor):
    return {"shape": list(tensor.shape), "data": base64.b64encode(little_floats(tensor)).decode("ascii")}


def route():
    return [[8-f, 5] for f in range(FRAMES)]


def dots(points):
    guide = torch.zeros(1, FRAMES, HEIGHT, WIDTH)
    for f, point in enumerate(points):
        if point is not None:
            x, y = point
            if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                raise ValueError("提示点不在画面内")
            guide[0, f, y, x] = 1
    return guide


@torch.no_grad()
def export(root, destination):
    torch.set_num_threads(2)
    base_path, branch_path = root/"pixel-lab/weights.pt", root/"control-lab/branch.pt"
    branch_ckpt = torch.load(branch_path, map_location="cpu", weights_only=True)
    recording = Recording(root/"pixel-lab/clock-data.js")
    fingerprint = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if not (recording.weight_sha256 == fingerprint == branch_ckpt["base_sha256"]):
        raise ValueError("输入记录、基座与支路的权重指纹不一致")
    base = TinyVideo(True).eval()
    base.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True)["joint"])
    branch = Branch(base).eval()
    branch.load_state_dict(branch_ckpt["branch"])

    n = 12
    start = recording.plane_offset(0, n, 0)
    count = FRAMES*HEIGHT*WIDTH
    state = torch.tensor(recording.states[start:start+count]).reshape(1, FRAMES, HEIGHT, WIDTH)
    tick = torch.tensor([.5])
    a, sigma, *_ = recording.coefficients[n]
    features = base.inputs(state, tick)
    skip, low = base.encode(features, FRAMES)
    enter = branch.enter(features)
    fixed = {key:tensor_record(value) for key,value in
             [("x",state),("skip",skip),("low",low),("enter",enter)]}
    layers = {}
    for owner, module, names in [("branch", branch, ("hint","down","middle","fine","coarse")),
                                 ("base", base, ("refine","exit"))]:
        for name in names:
            for param, value in getattr(module,name).state_dict().items():
                layers[f"{owner}.{name}.{param}"] = tensor_record(value)

    original = route()
    moved = [p.copy() for p in original]
    moved[2] = [10,5]
    settings = [("left", original, 1, 1), ("moved", moved, 1, 1),
                ("off", original, 0, 0), ("empty", [None]*FRAMES, 1, 1),
                ("mixed", [[0,0],None,[4,11],None,[12,0],[15,11]], .35, 1.7)]
    answers = {}
    for key, points, fine_scale, coarse_scale in settings:
        guide = dots(points)
        fine, coarse = branch(features, guide)
        prediction = controlled(base,branch,state,tick,guide,(fine_scale,coarse_scale))
        clean = a*state-sigma*prediction
        answers[key] = {"points":points,"scales":[fine_scale,coarse_scale],"fine":tensor_record(fine),
                        "coarse":tensor_record(coarse),"clean":tensor_record(clean),"v":tensor_record(prediction)}
    payload = {"base_sha256":fingerprint,"branch_sha256":hashlib.sha256(branch_path.read_bytes()).hexdigest(),
               "a":a,"sigma":sigma,"fixed":fixed,"layers":layers,"answers":answers}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"probe.json").write_text(json.dumps(payload,separators=(",",":")),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path,default=root)
    parser.add_argument("--out", type=Path,default=Path("replay/hint-probe"))
    args = parser.parse_args()
    export(args.root,args.out)
    from hint_report import render
    render(args.out)
