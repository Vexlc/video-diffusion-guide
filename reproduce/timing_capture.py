"""从实验 I 的同一批初始噪声重新生成两遍，仅改变 DDIM 的完整步数。需要 CPU PyTorch。"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import torch

from clock_report import Recording
from pixel_lab import EVAL_COUNT, STEPS, TinyVideo, sample, stream

ROOT = Path(__file__).resolve().parents[1]


def capture(directory, destination):
    weight = directory / "weights.pt"
    recording_file = directory / "clock-data.js"
    rec = Recording(recording_file)
    weight_hash = hashlib.sha256(weight.read_bytes()).hexdigest()
    if weight_hash != rec.weight_sha256:
        raise ValueError("采样记录与模型权重不匹配")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    saved = torch.load(weight, map_location="cpu", weights_only=True)
    model = TinyVideo(True)
    model.load_state_dict(saved["joint"])
    model.eval()
    shape = (EVAL_COUNT, rec.frames, rec.height, rec.width)
    initial = torch.randn(shape, generator=stream(saved["seed"]+300))
    stored_size = rec.cases * rec.frames * rec.plane
    if initial[:rec.cases].flatten().tolist() != rec.states[:stored_size].tolist():
        raise ValueError("重抽的测试噪声与双时钟记录不同")
    panels = {}
    def save(key, state):
        gray = ((state[:rec.cases].clamp(-1,1)+1)*127.5).round().to(torch.uint8)
        panels[key] = base64.b64encode(bytes(gray.flatten().tolist())).decode("ascii")
    save("start", initial)
    ticks = {}
    for budget in (6, STEPS):
        nodes = []
        def observe(x, v, following, coeff):
            nodes.append([float(item.flatten()[0]) for item in coeff])
            if budget == STEPS and len(nodes) == 6:
                save("partial", following)
        end, _ = sample(model, initial, observe=observe, steps=budget)
        save(str(budget), end)
        ticks[str(budget)] = nodes
    archived = torch.tensor(rec.states[-stored_size:].tolist()).reshape_as(end[:rec.cases])
    max_error = (end[:rec.cases]-archived).abs().max().item()
    if max_error > 1e-4:
        raise ValueError(f"同一长日程未能复现随包成片，最大偏差 {max_error}")
    result = {"shape":[rec.cases,*shape[1:]], "seed":saved["seed"], "weights":weight_hash,
              "recording":hashlib.sha256(recording_file.read_bytes()).hexdigest(),
              "ticks":ticks, "panels":panels, "reference_max_error":max_error}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"timing.json").write_text(json.dumps(result,separators=(",",":")),encoding="utf-8")
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=ROOT/"pixel-lab")
    parser.add_argument("--out",type=Path,default=Path("replay/timing"))
    args = parser.parse_args()
    result = capture(args.directory,args.out)
    print("24 步重跑对原始成片的最大浮点偏差",result["reference_max_error"])
    from timing_report import render
    render(result,args.out)
