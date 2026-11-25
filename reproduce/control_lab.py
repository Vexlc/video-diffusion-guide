"""实验 J：冻结 I 的跨帧视频网络，用两个零初始化出口学习点提示。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from pixel_lab import (BATCH, FRAMES, HEIGHT, WIDTH, TinyVideo, byte_clips, exercise, make_clips,
                       mix_neighbors, paint, sample, stream)
import torch
from torch import nn
from torch.nn import functional as functional


def zeroed(module):
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


def center_dots(clips):
    batch = len(clips)
    patches = functional.avg_pool2d(clips.reshape(-1, 1, HEIGHT, WIDTH), 3, stride=1)
    peak = patches.flatten(1).argmax(1)
    row, col = peak//(WIDTH-2)+1, peak%(WIDTH-2)+1
    dots = torch.zeros_like(clips).reshape(-1, HEIGHT, WIDTH)
    dots[torch.arange(len(peak), device=clips.device), row, col] = 1
    return dots.reshape(batch, FRAMES, HEIGHT, WIDTH)


def visible(dots, modes):
    frame = torch.arange(FRAMES, device=dots.device)[None, :]
    keep = (frame==0) | ((modes[:, None]>=1) & (frame==FRAMES-1)) | (modes[:, None]==2)
    return dots * keep[:, :, None, None]


class Branch(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.enter = copy.deepcopy(base.enter)
        self.down = copy.deepcopy(base.down)
        self.middle = copy.deepcopy(base.middle)
        self.hint = zeroed(nn.Conv2d(1, 16, 3, padding=1))
        self.fine = zeroed(nn.Conv2d(16, 16, 1))
        self.coarse = zeroed(nn.Conv2d(32, 32, 1))

    def forward(self, features, guide):
        frames = guide.shape[1]
        skip = self.enter(features) + self.hint(guide.flatten(0, 1).unsqueeze(1))
        low = self.down(mix_neighbors(skip, frames, True))
        low = low + self.middle(mix_neighbors(low, frames, True))
        return self.fine(skip), self.coarse(low)


def controlled(base, branch, state, tick, guide, scales=(1., 1.)):
    features = base.inputs(state, tick)
    skip, low = base.encode(features, state.shape[1])
    fine, coarse = branch(features, guide)
    return base.decode_features(state, skip+scales[0]*fine, low+scales[1]*coarse)


def paired_cases(seed, pairs=48):
    rng = stream(seed+201)
    start = torch.randint(7, 9, (pairs,), generator=rng).repeat_interleave(2)
    row = torch.randint(2, 10, (pairs,), generator=rng).repeat_interleave(2)
    direction = torch.tensor([-1, 1]).repeat(pairs)
    truth = paint(start, row, direction)
    initial = torch.randn((pairs, FRAMES, HEIGHT, WIDTH), generator=stream(seed+202)).repeat_interleave(2, 0)
    return truth, initial


def positions(video):
    pooled = functional.avg_pool2d(video.reshape(-1, 1, HEIGHT, WIDTH), 3, stride=1)
    index = pooled.flatten(1).argmax(1).reshape(-1, FRAMES)
    return torch.stack([index%(WIDTH-2)+1, index//(WIDTH-2)+1], dim=-1)


def measurements(result, truth, opposite):
    locations = positions(result)
    distance = (locations-positions(truth)).abs().sum(2)
    other_distance = (locations-positions(opposite)).abs().sum(2)
    return {"exact": int((distance==0).all(1).sum()), "opposite": int((other_distance==0).all(1).sum()),
            "middle_l1": distance[:, 1:-1].float().mean().item(),
            "pixel_mse": (result-truth).square().mean().item(), "centers": locations.tolist()}


def grad_norm(module):
    return sum(param.grad.detach().double().square().sum().item() for param in module.parameters() if param.grad is not None)**.5


def teach(base, branch, seed, updates, device, *, observe=None):
    optimizer = torch.optim.Adam(branch.parameters(), lr=.001)
    rng = stream(seed+101)
    gradients, history = [], []
    for step in range(updates):
        if step in (1600, 2800):
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
        clean = make_clips(BATCH, rng).to(device)
        eps = torch.randn(clean.shape, generator=rng).to(device)
        tick = torch.rand(BATCH, generator=rng).to(device)
        mode = torch.randint(0, 3, (BATCH,), generator=rng).to(device)
        guide = visible(center_dots(clean), mode)
        state, target = exercise(clean, eps, tick)
        prediction = controlled(base, branch, state, tick, guide)
        optimizer.zero_grad(set_to_none=True)
        loss = (prediction-target).square().mean()
        loss.backward()
        if step < 2:
            gradients.append({key: grad_norm(getattr(branch, key)) for key in ("enter", "hint", "fine", "coarse")})
        if observe is not None:
            observe(step, state, tick, guide, target, loss)
        optimizer.step()
        if step+1 in (1, 100, 1000, 2000, updates):
            history.append((step+1, loss.item()))
            print("训练批次", step+1, loss.item(), flush=True)
    return gradients, history


@torch.no_grad()
def evaluate(base, branch, seed, device):
    truth, initial = paired_cases(seed)
    other = truth.reshape(-1, 2, FRAMES, HEIGHT, WIDTH).flip(1).reshape_as(truth)
    dots, noise = center_dots(truth).to(device), initial.to(device)
    both = visible(dots, torch.ones(len(dots), device=device))
    first = visible(dots, torch.zeros(len(dots), device=device))
    alternate = center_dots(other).to(device)
    settings = [("off", dots, (0, 0)), ("first", first, (1, 1)), ("ends", both, (1, 1)),
                ("dense", dots, (1, 1)), ("wrong", alternate, (1, 1)), ("half", dots, (.5, .5)),
                ("double", dots, (2, 2)), ("fine", dots, (1, 0)), ("coarse", dots, (0, 1))]
    scores, examples = {}, {"truth": byte_clips(truth[:8]), "opposite": byte_clips(other[:8]),
                            "guide": byte_clips(dots[:8]*2-1)}
    for key, guide, scales in settings:
        outputs, unused = sample(lambda state,tick: controlled(base, branch, state, tick, guide, scales), noise)
        scores[key] = measurements(outputs, truth, other)
        scores[key]["pair_max_delta"] = (outputs[::2]-outputs[1::2]).abs().max().item()
        examples[key] = byte_clips(outputs[:8])
    reference, unused = sample(base, noise)
    disabled, unused = sample(lambda state,tick: controlled(base, branch, state, tick, dots, (0,0)), noise)
    return scores, examples, (reference-disabled).abs().max().item()


def run(base_path, replay, output, seed, updates, device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    fingerprint = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if replay:
        state = torch.load(replay, map_location="cpu", weights_only=True)
        if state["base_sha256"] != fingerprint:
            raise ValueError("基座文件指纹与训练支路时不同，请使用同一份 weights.pt")
        seed, updates = state["seed"], state["updates"]
    torch.manual_seed(seed)
    base = TinyVideo(True)
    base.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True)["joint"])
    branch = Branch(base).to(device)
    base = base.to(device).eval().requires_grad_(False)
    original = {name: param.detach().clone() for name, param in base.named_parameters()}
    test, noise = paired_cases(seed, pairs=2)
    test, noise = test.to(device), noise.to(device)
    tick = torch.full((len(test),), .5, device=device)
    with torch.no_grad():
        zero_error = (controlled(base, branch, noise, tick, center_dots(test))-base(noise, tick)).abs().max().item()
    output.mkdir(parents=True, exist_ok=True)
    if replay:
        branch.load_state_dict(state["branch"])
        gradients, history = state["gradients"], state["history"]
    else:
        gradients, history = teach(base, branch, seed, updates, device)
        torch.save({"branch": {name: value.cpu() for name, value in branch.state_dict().items()},
                    "seed": seed, "updates": updates, "gradients": gradients, "history": history,
                    "base_sha256": fingerprint}, output/"branch.pt")
    frozen_delta = max((param.detach()-original[name]).abs().max().item() for name,param in base.named_parameters())
    scores, examples, off_error = evaluate(base, branch, seed, device)
    report = {"seed": seed, "updates": updates, "device": str(device), "torch": str(torch.__version__),
              "params": sum(param.numel() for param in branch.parameters()), "base_sha256": fingerprint,
              "frozen_delta": frozen_delta, "initial_delta": zero_error, "off_delta": off_error,
              "gradients": gradients, "history": history, "scores": scores, "examples": examples}
    (output/"numbers.json").write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")
    print({key:(value["exact"],value["opposite"],round(value["middle_l1"],3),round(value["pixel_mse"],3)) for key,value in scores.items()}, flush=True)
    from control_report import render
    render(report, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1]/"pixel-lab"/"weights.pt")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--out", type=Path, default=Path("replay/control-lab"))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--updates", type=int, default=3200)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    if args.updates<0:
        parser.error("更新次数不能为负")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    run(args.base, args.replay, args.out, args.seed, args.updates, torch.device(device))
