"""用实验 P 的随包权重生成三段影片，分别固定和移动向左片的首帧点。需要 PyTorch。"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import torch

from dit_lab import LEFT, RIGHT, patchify
from flow_hint import ROOT, SEED, STEPS, load_trained, setup, solve
from latent_lab import LATENT_SHAPE, device_noise
from pixel_lab import brightest_centers, stream

POINTS = ((7,3), (7,3), (12,3))
NAMES = ("right", "left", "left_end")


@torch.no_grad()
def capture(root, seeds):
    setup("cpu", SEED)
    vae, base, branch, saved = load_trained(root, root/"flow-hint/branch.pt", "cpu")
    hashes = {**saved["base_hashes"], "branch":hashlib.sha256((root/"flow-hint/branch.pt").read_bytes()).hexdigest()}
    record = {"hashes":hashes, "steps":STEPS, "token":8, "component":1, "seeds":{}}
    for seed in seeds:
        noise = device_noise(LATENT_SHAPE,stream(seed),"cpu")
        starts = noise.unsqueeze(0).repeat(3,1,1,1,1)
        xs = [patchify(starts)[:,8,1].tolist()]
        velocities = []
        def observe(k, state, velocity, following, time, hint):
            xs.append(patchify(following)[:,8,1].tolist())
            velocities.append(patchify(velocity)[:,8,1].tolist())
        final = solve(base,branch,starts,torch.tensor([RIGHT,LEFT,LEFT]),torch.tensor(POINTS),1.,observe)
        decoded = vae.decode(saved["mean"] + saved["spread"]*final)
        centers = brightest_centers(decoded).tolist()
        gray = ((decoded.clamp(-1,1)+1)*127.5).round().to(torch.uint8)
        record["seeds"][str(seed)] = {
            name:{"film":base64.b64encode(gray[i].contiguous().numpy().tobytes()).decode("ascii"),
                  "centers":centers[i], "x":[row[i] for row in xs], "u":[row[i] for row in velocities]}
            for i,name in enumerate(NAMES)}
    return record


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,nargs="+",default=[9317,9318,9319])
    parser.add_argument("--out",type=Path,default=Path("replay/playback"))
    args = parser.parse_args()
    record = capture(ROOT,args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"playback.json").write_text(json.dumps(record,separators=(",",":")),encoding="utf-8")
    from playback_report import render
    render(record,args.out)
    print(args.out/"playback.html")
