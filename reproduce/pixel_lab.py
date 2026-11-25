"""可选实验 I：训练两个同参数量的微型像素视频扩散网络。需要 PyTorch。"""

import argparse
import copy
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as functional


FRAMES, HEIGHT, WIDTH = 6, 12, 16
STEPS, BATCH, EVAL_COUNT = 24, 24, 96


def stream(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def paint(start, row, direction):
    center = start[:, None] + direction[:, None]*torch.arange(FRAMES)
    within_x = (torch.arange(WIDTH)[None, None, None, :]-center[:, :, None, None]).abs() <= 1
    within_y = (torch.arange(HEIGHT)[None, None, :, None]-row[:, None, None, None]).abs() <= 1
    return (within_x & within_y).float()*2-1


def make_clips(batch, rng):
    left = torch.randint(2, 9, (batch,), generator=rng)
    row = torch.randint(2, 10, (batch,), generator=rng)
    direction = torch.randint(0, 2, (batch,), generator=rng)*2-1
    return paint(torch.where(direction>0, left, WIDTH-1-left), row, direction)


def amplitudes(tick):
    angle = tick * (math.pi/2)
    return angle.cos()[:, None, None, None], angle.sin()[:, None, None, None]


def mix_neighbors(feature, frames, enabled):
    if not enabled:
        return feature
    chunks = feature.reshape(-1, frames, *feature.shape[1:])
    piece = chunks.shape[2]//4
    previous = torch.cat([chunks[:, :1, :piece], chunks[:, :-1, :piece]], dim=1)
    following = torch.cat([chunks[:, 1:, piece:2*piece], chunks[:, -1:, piece:2*piece]], dim=1)
    return torch.cat([previous, following, chunks[:, :, 2*piece:]], dim=2).reshape_as(feature)


class Block(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, stride, 1),
                                    nn.GroupNorm(4, outgoing), nn.SiLU())

    def forward(self, feature):
        return self.layers(feature)


class TinyVideo(nn.Module):
    def __init__(self, temporal):
        super().__init__()
        self.temporal = temporal
        self.enter = Block(4, 16)
        self.down = Block(16, 32, 2)
        self.middle = Block(32, 32)
        self.refine = Block(48, 16)
        self.exit = nn.Conv2d(16, 1, 3, padding=1)

    def inputs(self, state, tick):
        batch, frames, height, width = state.shape
        signal, sigma = amplitudes(tick)
        order = torch.linspace(-1, 1, frames, device=state.device)[None, :, None, None]
        features = torch.stack([state, signal.expand_as(state), sigma.expand_as(state), order.expand_as(state)], dim=2)
        return features.reshape(batch*frames, 4, height, width)

    def encode(self, features, frames):
        skip = self.enter(features)
        low = self.down(mix_neighbors(skip, frames, self.temporal))
        low = low + self.middle(mix_neighbors(low, frames, self.temporal))
        return skip, low

    def decode_features(self, state, skip, low):
        frames = state.shape[1]
        high = functional.interpolate(low, size=state.shape[-2:], mode="nearest")
        joined = self.refine(torch.cat([skip, mix_neighbors(high, frames, self.temporal)], dim=1))
        return self.exit(joined).reshape_as(state)

    def forward(self, state, tick):
        return self.decode_features(state, *self.encode(self.inputs(state, tick), state.shape[1]))


def exercise(clean, eps, tick):
    signal, sigma = amplitudes(tick)
    return signal*clean + sigma*eps, signal*eps-sigma*clean


@torch.no_grad()
def sample(model, initial, *, observe=None, steps=STEPS):
    if steps < 1:
        raise ValueError("采样至少需要一步")
    state = initial.clone()
    pictures = {"start": state[:8].cpu().clone()}
    for step in range(steps, 0, -1):
        tick = torch.full((len(state),), step/steps, device=state.device)
        signal, sigma = amplitudes(tick)
        predicted = model(state, tick)
        clean = signal*state - sigma*predicted
        eps = sigma*state + signal*predicted
        next_signal, next_sigma = amplitudes(tick-1/steps)
        following = next_signal*clean + next_sigma*eps
        if observe is not None:
            observe(state, predicted, following, (signal, sigma, next_signal, next_sigma))
        state = following
        if step in (steps, 3*steps//4, steps//2, steps//4, 1):
            pictures[str(steps-step)] = clean[:8].cpu().clone()
    return state.cpu(), pictures


def brightest_centers(videos):
    """逐帧取最亮 3×3 窗口的中心，返回（列，行）。只读像素。"""
    sums = functional.avg_pool2d(videos.reshape(-1, 1, HEIGHT, WIDTH), 3, stride=1)
    peak = sums.flatten(1).argmax(1).reshape(-1, FRAMES)
    return torch.stack((peak % (WIDTH-2) + 1, peak // (WIDTH-2) + 1), dim=2)


def template_scores(videos):
    triples = [(left if direction==1 else WIDTH-1-left, row, direction)
               for left in range(2, 9) for row in range(2, 10) for direction in (-1, 1)]
    starts, rows, directions = torch.tensor(triples).T
    templates = paint(starts, rows, directions).flatten(1)
    norm = templates.shape[1]
    values = videos.flatten(1)
    squared = values.square().mean(1)[:, None] + 1 - 2*(values @ templates.T)/norm
    best = squared.min(1)
    centers = brightest_centers(videos)
    center_x, center_y = centers.unbind(dim=2)
    allowed_x = starts[:, None]+directions[:, None]*torch.arange(FRAMES)
    hits = ((center_x[:, None, :] == allowed_x[None, :, :]) &
            (center_y[:, None, :] == rows[None, :, None])).all(2).any(1)
    frame_min = []
    for frame in range(FRAMES):
        masks = templates.reshape(-1, FRAMES, HEIGHT*WIDTH)[:, frame]
        pixels = videos[:, frame].flatten(1)
        err = pixels.square().mean(1)[:, None]+1-2*(pixels@masks.T)/(HEIGHT*WIDTH)
        frame_min.append(err.min(1).values.clamp_min(0))
    return {"route_hits": int(hits.sum()), "nearest_distinct": int(best.indices.unique().numel()), "frame_mse": float(torch.stack(frame_min).mean()),
            "clip_mse": float(best.values.clamp_min(0).mean()), "right_nearest": int((directions[best.indices]>0).sum()),
            "per_clip": best.values.clamp_min(0).tolist(), "hits": hits.tolist(),
            "centers": centers.tolist()}


def save_report(output, shape, history, examples, truth, noisy, details):
    from pixel_report import render
    output.mkdir(parents=True, exist_ok=True)
    payload = {"setup": shape, "history": history, "examples": examples, "probe": {"truth": truth, "noisy": noisy},
               "scores": details}
    (output/"numbers.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    render(payload, output)


def byte_clips(tensor):
    return ((tensor.clamp(-1, 1)+1)*127.5).round().byte().tolist()


def experiment(output, updates, seed, device, replay):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if replay:
        checkpoint = torch.load(replay, map_location="cpu", weights_only=True)
        updates, seed = checkpoint["updates"], checkpoint["seed"]
    torch.manual_seed(seed)
    local_net = TinyVideo(False)
    both_net = copy.deepcopy(local_net)
    both_net.temporal = True
    models = {"single": local_net.to(device), "joint": both_net.to(device)}
    optimizers = {key: torch.optim.Adam(net.parameters(), lr=.0015) for key, net in models.items()}
    train_rng, check_rng = stream(seed+100), stream(seed+200)
    clean_check = make_clips(EVAL_COUNT, check_rng).to(device)
    eps_check = torch.randn(clean_check.shape, generator=check_rng).to(device)
    tick_check = torch.linspace(.05, .95, EVAL_COUNT, device=device)
    noisy_check, target_check = exercise(clean_check, eps_check, tick_check)
    initial = torch.randn(clean_check.shape, generator=stream(seed+300)).to(device)
    probe_tick = torch.full((8,), .65, device=device)
    probe, unused = exercise(clean_check[:8], eps_check[:8], probe_tick)
    history = []
    early, unused = sample(both_net, initial[:8])
    @torch.no_grad()
    def evaluate(step):
        history.append({"step": step, **{key: (net(noisy_check, tick_check)-target_check).square().mean().item()
                                         for key, net in models.items()}})

    output.mkdir(parents=True, exist_ok=True)
    if replay:
        for key, net in models.items():
            net.load_state_dict(checkpoint[key])
        history = checkpoint["history"]
    else:
        evaluate(0)
        for step in range(1, updates+1):
            if step in (1201, 3001):
                for optimizer in optimizers.values():
                    for group in optimizer.param_groups:
                        group["lr"] = .0008 if step==1201 else .0004
            clean = make_clips(BATCH, train_rng).to(device)
            eps = torch.randn(clean.shape, generator=train_rng).to(device)
            tick = torch.rand(BATCH, generator=train_rng).to(device)
            state, target = exercise(clean, eps, tick)
            for key, net in models.items():
                optimizers[key].zero_grad(set_to_none=True)
                loss = (net(state, tick)-target).square().mean()
                loss.backward()
                optimizers[key].step()
            if step in (1, 10, 50, 100, 250, 500, 1000, 1200, updates) or step%1000==0:
                evaluate(step)
                print(step, history[-1], flush=True)
        checkpoint = {key: {name: value.cpu() for name, value in net.state_dict().items()} for key, net in models.items()}
        checkpoint.update({"seed": seed, "updates": updates, "history": history})
        torch.save(checkpoint, output/"weights.pt")
    from clock_report import ClockRecorder
    recorder = ClockRecorder(8)
    examples, details = {"untrained": byte_clips(early)}, {}
    for key, net in models.items():
        generated, pictures = sample(net, initial, observe=recorder.observe if key=="joint" else None)
        scores = template_scores(generated)
        with torch.no_grad():
            signal, sigma = amplitudes(probe_tick)
            inferred = signal*probe - sigma*net(probe, probe_tick)
            query_tick = torch.full_like(tick_check, .65)
            observed, unused = exercise(clean_check, eps_check, query_tick)
            swapped = observed.roll(1, dims=0)
            swapped[:, 2] = observed[:, 2]
            signal, sigma = amplitudes(query_tick)
            intact = (signal*observed - sigma*net(observed, query_tick))[:, 2]
            tampered = (signal*swapped - sigma*net(swapped, query_tick))[:, 2]
            scores["neighbor_test"] = {"mae_change": (intact-tampered).abs().mean().item(),
                                        "intact_mse": (intact-clean_check[:, 2]).square().mean().item(),
                                        "swapped_mse": (tampered-clean_check[:, 2]).square().mean().item()}
            if key == "joint":
                examples["neighbor"] = {"intact": byte_clips(intact[:8]), "swapped": byte_clips(tampered[:8]),
                                        "foreign": byte_clips(swapped[:8].cpu())}
        examples[key] = {name: byte_clips(frames) for name, frames in pictures.items()}
        examples[key]["probe"] = byte_clips(inferred.cpu())
        details[key] = scores
        print(key, {name: value for name, value in scores.items() if not isinstance(value,list)}, flush=True)
    setup = {"seed": seed, "updates": updates, "sampler_steps": STEPS, "test_clips": EVAL_COUNT,
             "parameters": sum(param.numel() for param in local_net.parameters()), "torch": str(torch.__version__),
             "device": str(device), "training_batch": BATCH}
    recorder.write(output, replay or output/"weights.pt", seed, updates)
    save_report(output, setup, history, examples, byte_clips(clean_check[:8].cpu()), byte_clips(probe.cpu()), details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--replay", type=Path, help="读取本实验保存的 state_dict，跳过训练")
    parser.add_argument("--out", type=Path, default=Path("replay/pixel-lab"))
    args = parser.parse_args()
    if args.updates < 0:
        parser.error("--updates 不能为负")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    experiment(args.out, args.updates, args.seed, torch.device(device_name), args.replay)
