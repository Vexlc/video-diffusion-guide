"""用两台随包视频网络对接 Diffusers 0.35.1 的两个调度器。仅需 CPU。"""
import argparse
import hashlib
import inspect
import json
import math
from importlib.metadata import version
from pathlib import Path

import torch
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler

from clock_report import Recording
from dit_lab import RIGHT
from flow_hint import EVAL_SEED, hint_map, initial_states, load_trained, setup, solve, velocity
from hint_probe import tensor_record
from pixel_lab import STEPS, TinyVideo, brightest_centers, byte_clips, sample

ROOT = Path(__file__).resolve().parents[1]
FILES = ("pixel-lab/weights.pt", "pixel-lab/clock-data.js", "latent-lab/weights.pt",
         "dit-lab/weights.pt", "flow-hint/branch.pt")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distance(x, y):
    delta = (x.double() - y.double()).flatten()
    return {"max": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item()}


def ddim_scheduler(clip):
    # 表的第 i 格对应连续噪声时钟 (i+1)/24。最后一格为纯噪声。
    powers = [1.] + [math.cos(i * math.pi / (2 * STEPS)) ** 2 for i in range(1, STEPS)] + [0.]
    betas = [1 - after / before for before, after in zip(powers, powers[1:])]
    scheduler = DDIMScheduler(num_train_timesteps=STEPS, trained_betas=betas, clip_sample=clip,
                              prediction_type="v_prediction", set_alpha_to_one=True,
                              timestep_spacing="leading")
    scheduler.set_timesteps(STEPS)
    return scheduler


@torch.no_grad()
def ddim_run(root):
    record = Recording(root / FILES[1])
    weights = root / FILES[0]
    if record.weight_sha256 != sha(weights):
        raise ValueError("初始噪声录像对应的网络权重不同")
    shape = (record.frames, record.height, record.width)
    start = torch.tensor(record.states[:math.prod(shape)].tolist()).reshape(1, *shape)
    net = TinyVideo(True).eval()
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["joint"])
    sampler, clipped = ddim_scheduler(False), ddim_scheduler(True)
    state, other = start.clone(), start.clone()
    rows = []
    for i, t in enumerate(sampler.timesteps):
        tick = (t.float() + 1).reshape(1) / STEPS
        v = net(state, tick)
        out = sampler.step(v, t, state, eta=0)
        local_clip = clipped.step(v, t, state, eta=0).prev_sample
        other = clipped.step(net(other, tick), t, other, eta=0).prev_sample
        power = sampler.alphas_cumprod[t]
        nxt = sampler.alphas_cumprod[t-1] if t > 0 else sampler.final_alpha_cumprod
        a, b, c, d = power.sqrt(), (1-power).sqrt(), nxt.sqrt(), (1-nxt).sqrt()
        manual = c*(a*state-b*v) + d*(b*state+a*v)
        rows.append({"i": i, "t": int(t), "coef": [float(s) for s in (a,b,c,d)],
                     "x": tensor_record(state[0]), "v": tensor_record(v[0]),
                     "next": tensor_record(out.prev_sample[0]), "alternate": tensor_record(local_clip[0]),
                     "outside": int((out.pred_original_sample.abs()>1).sum()),
                     "local": distance(manual, out.prev_sample)})
        state = out.prev_sample
    course, _ = sample(net, start)
    return {"shape": list(shape), "steps": rows, "reference": distance(state, course),
            "changed": distance(state, other),
            "films": [byte_clips(x)[0] for x in (course, state, other)]}


def flow_scheduler():
    sampler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1)
    sampler.set_timesteps(sigmas=[1-i/STEPS for i in range(STEPS)], device="cpu")
    return sampler


@torch.no_grad()
def flow_run(root):
    cpu = torch.device("cpu")
    vae, base, branch, saved = load_trained(root, root / FILES[-1], cpu)
    start = initial_states(1, cpu)
    label, point = torch.tensor([RIGHT]), torch.tensor([[7, 3]])
    hint = hint_map(point)
    sampler, local_wrong, wrong = (flow_scheduler() for _ in range(3))
    state, other = start.clone(), start.clone()
    rows = []
    for i, t in enumerate(sampler.timesteps):
        sigma, following = sampler.sigmas[i:i+2]
        tau = (1-sigma).reshape(1, 1)
        u = velocity(base, branch, state, tau, label, hint, 1.)
        accepted = sampler.step(-u, t, state).prev_sample
        reversed_step = local_wrong.step(u, t, state).prev_sample
        other_u = velocity(base, branch, other, tau, label, hint, 1.)
        other = wrong.step(other_u, t, other).prev_sample
        manual = state + (sigma-following)*u
        rows.append({"i": i, "t": float(t), "sigma": float(sigma), "following": float(following),
                     "x": tensor_record(state[0]), "v": tensor_record(u[0]),
                     "next": tensor_record(accepted[0]), "alternate": tensor_record(reversed_step[0]),
                     "local": distance(manual, accepted)})
        state = accepted
    course = solve(base, branch, start, label, point, 1.)
    clips = [vae.decode(saved["mean"] + saved["spread"] * z) for z in (course, state, other)]
    return {"shape": list(start.shape[1:]), "seed": EVAL_SEED, "steps": rows,
            "reference": distance(state, course), "changed": distance(state, other),
            "centers": [brightest_centers(x)[0].tolist() for x in clips],
            "films": [byte_clips(x)[0] for x in clips]}


def run(root, out):
    if version("diffusers") != "0.35.1":
        raise ValueError("此对照固定使用 diffusers==0.35.1")
    setup(torch.device("cpu"), EVAL_SEED)
    report = {"diffusers": version("diffusers"), "torch": str(torch.__version__),
              "files": {name: sha(root/name) for name in FILES},
              "source": {cls.__name__: sha(inspect.getfile(cls))
                         for cls in (DDIMScheduler, FlowMatchEulerDiscreteScheduler)},
              "ddim": ddim_run(root), "flow": flow_run(root)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    for key in ("ddim", "flow"):
        print(key, "与讲义独立成片的终点差", report[key]["reference"], "改变配置后的终点差", report[key]["changed"])
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=ROOT)
    cli.add_argument("--out", type=Path, default=Path("replay/schedulers.json"))
    args = cli.parse_args()
    run(args.root, args.out)
