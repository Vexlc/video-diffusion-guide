"""从实验 M 的冻结权重读取一次注意力消息，再只改一个块的 V 分量。CPU 即可。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from dit_lab import COUNT, DIM, HEADS, LEFT, TOKENS, TinyDiT, patchify
from latent_lab import LATENT_SHAPE, PairVAE, catalog, device_noise, normalization
from pixel_lab import byte_clips, stream

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def capture(directory, vae_file):
    torch.set_num_threads(2)
    weights = directory / "weights.pt"
    numbers = directory / "numbers.json"
    course = json.loads(numbers.read_text(encoding="utf-8"))
    saved = torch.load(weights, weights_only=True, map_location="cpu")
    vae_hash = hashlib.sha256(vae_file.read_bytes()).hexdigest()
    if saved["base_sha256"] != vae_hash or course["base_sha256"] != vae_hash:
        raise ValueError("这份 DiT 和 VAE 的权重不配套")

    vae = PairVAE().eval()
    vae.load_state_dict(torch.load(vae_file, weights_only=True, map_location="cpu")["light"])
    clips, hold = catalog()
    mean, scale = normalization(vae.encode(clips[~hold])[0])
    clean = (vae.encode(clips[hold])[0][:1] - mean) / scale
    noise = device_noise((COUNT, *LATENT_SHAPE), stream(course["eval_seed"] + 101), torch.device("cpu"))[:1]
    state = .5 * (clean + noise)
    torch.testing.assert_close(patchify(state)[0], torch.tensor(course["probe"]["tokens"]), atol=2e-5, rtol=2e-5)

    net = TinyDiT().eval()
    net.load_state_dict(saved["model"])
    time = torch.tensor([[.5]])
    label = torch.tensor([LEFT])
    block = net.blocks[0]
    record = {"weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "vae_sha256": vae_hash,
              "numbers_sha256": hashlib.sha256(numbers.read_bytes()).hexdigest(), "t": .5,
              "frames": byte_clips(clips[hold][:1])[0], "runs": {}}
    # 在同一帧组布局上复用完整前向，逐个修改源块。所有基线也采用 batch=1。
    for mode, local in (("full", False), ("local", True)):
        tapped = {}

        def intercept(module, args, packed):
            # Q、K、V 各有 48 个通道，V 的首通道属于第一个头。
            if source is None:
                width = DIM // HEADS
                tapped["queries"] = packed[0, :, :width].clone()
                tapped["keys"] = packed[0, :, DIM:DIM+width].clone()
            result = packed
            if source is not None:
                result = packed.clone()
                result[0, source, 2*DIM] += 1
            tapped["values"] = result[0, :, 2*DIM].clone()
            return result

        def receive(module, args):
            # 输出投影真实收到的头 0、分量 0；沿 18 个查询位置读取。
            tapped["messages"] = args[0][0, :, 0].clone()

        h1 = block.qkv.register_forward_hook(intercept)
        h2 = block.projection.register_forward_pre_hook(receive)
        try:
            source = None
            before, maps = net(state, time, label, local, trace=True)
            weights_row = maps[0][0, 0]
            dots = tapped["queries"] @ tapped["keys"].T
            logits = dots / (DIM//HEADS)**.5
            if local:
                logits = logits.masked_fill(block.other, float("-inf"))
            torch.testing.assert_close(logits.softmax(-1), weights_row, atol=2e-6, rtol=1e-5)
            values = tapped["values"]
            messages = tapped["messages"]
            torch.testing.assert_close(weights_row @ values, messages, atol=2e-6, rtol=1e-5)
            if not local:
                torch.testing.assert_close(weights_row, torch.tensor(course["probe"]["answers"]["left"]["maps"][0][0]),
                                           atol=3e-5, rtol=1e-4)
            row = {"queries": tapped["queries"].tolist(), "keys": tapped["keys"].tolist(),
                   "weights": weights_row.tolist(), "values": values.tolist(), "messages": messages.tolist(),
                   "velocity": patchify(before)[0, :, 0].tolist(), "edited_messages": [], "edited_velocity": []}
            for source in range(TOKENS):
                after, edited_maps = net(state, time, label, local, trace=True)
                torch.testing.assert_close(edited_maps[0], maps[0], atol=0, rtol=0)
                torch.testing.assert_close(tapped["messages"] - messages, weights_row[:, source], atol=2e-6, rtol=1e-5)
                row["edited_messages"].append(tapped["messages"].tolist())
                row["edited_velocity"].append(patchify(after)[0, :, 0].tolist())
            record["runs"][mode] = row
        finally:
            h1.remove()
            h2.remove()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT/"dit-lab")
    parser.add_argument("--vae", type=Path, default=ROOT/"latent-lab/weights.pt")
    parser.add_argument("--out", type=Path, default=Path("replay/attention/message.json"))
    args = parser.parse_args()
    record = capture(args.model, args.vae)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    print(args.out)
