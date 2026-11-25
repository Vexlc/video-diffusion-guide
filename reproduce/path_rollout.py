"""从相同的八份雪花分别运行四次视频采样，比较直线和圆弧坐标。CPU + PyTorch。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from clock_report import Recording
from hint_probe import tensor_record
from path_coordinates import coefficients, line_coordinates
from pixel_lab import TinyVideo, sample


@torch.no_grad()
def line_sample(model, initial, steps, *, uniform=False, raw=False):
    """每个分支从自己的状态调用网络。与原版相配的节点按 tau=a/(a+sigma) 排列。"""
    nodes = []
    for i in range(steps+1):
        angle = (1-i/steps)*math.pi/2
        a, b = math.cos(angle), math.sin(angle)
        nodes.append(i/steps if uniform else a/(a+b))
    nodes[0], nodes[-1] = 0., 1.
    state = initial.clone()
    states = [state]
    for tau, following in zip(nodes, nodes[1:]):
        a, b = coefficients(tau)
        tick = 2/math.pi*math.atan2(b,a)
        original_units = (a+b)*state
        clock = torch.full((len(state),), tick, dtype=state.dtype, device=state.device)
        v = model(original_units, clock)
        _, _, speed = line_coordinates(original_units, v, a, b)
        state = state + (following-tau)*(v if raw else speed)
        states.append(state)
    return nodes, states


def rmse(actual, reference):
    return torch.mean((actual.double()-reference.double()).square()).sqrt().item()


def run(directory, out):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    weight = directory/"weights.pt"
    record_file = directory/"clock-data.js"
    record = Recording(record_file)
    if record.weight_sha256 != hashlib.sha256(weight.read_bytes()).hexdigest():
        raise ValueError("实验 I 的模型与随包记录不配套")
    sizes = (record.cases, record.frames, record.height, record.width)
    count = math.prod(sizes)
    initial = torch.tensor(record.states[:count].tolist()).reshape(sizes)
    model = TinyVideo(True)
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True)["joint"])
    model.eval().requires_grad_(False)
    reference = [initial]
    ref_nodes = [0.]
    def observe(x, v, following, amps):
        a, b = amps[2:]
        reference.append(following/(a+b))
        ref_nodes.append(float((a/(a+b)).flatten()[0]))
    arc, _ = sample(model, initial, observe=observe)
    if len(reference)!=record.steps+1:
        raise ValueError("采样步数与档案不一致")
    archived = torch.tensor(record.states[-count:].tolist()).reshape(sizes)
    rows = {}
    for mode in ("ddim", "matched", "raw", "uniform"):
        if mode=="ddim":
            nodes, path = ref_nodes, reference
        else:
            nodes, path = line_sample(model, initial, record.steps, uniform=mode=="uniform", raw=mode=="raw")
        end = path[-1]
        # 仅在节点匹配时逐节点比较。所有模式的最后一个节点都是干净端点。
        timeline = [rmse(x,y) for x,y in zip(path,reference)] if mode!="uniform" else None
        rows[mode] = {"nodes":nodes, "end":tensor_record(end), "rmse":rmse(end,arc),
                      "max_error":(end.double()-arc.double()).abs().max().item(), "same_node_rmse":timeline,
                      "per_case":[rmse(x,y) for x,y in zip(end,arc)]}
    data = {"shape":sizes, "steps":record.steps, "hashes":{"weights":record.weight_sha256,
            "record":hashlib.sha256(record_file.read_bytes()).hexdigest()}, "rows":rows,
            "archived_rmse":rmse(arc,archived), "torch":str(torch.__version__)}
    out.mkdir(parents=True,exist_ok=True)
    (out/"path-runs.json").write_text(json.dumps(data,separators=(",",":"),allow_nan=False),encoding="utf-8")
    for mode, row in rows.items():
        print(mode, "终点 RMSE", row["rmse"], "最大绝对差", row["max_error"])
    print("重跑原版与原有记录的终点 RMSE",data["archived_rmse"])
    return data


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1]/"pixel-lab")
    parser.add_argument("--out", type=Path, default=Path("replay/path-clock"))
    args=parser.parse_args()
    data=run(args.directory,args.out)
    from path_report import render
    render(data,args.directory/"clock-data.js",args.out,guide=(args.out/"two-clocks.html").is_file() and
           (args.out.parent/"video-diffusion-guide.pdf").is_file())
