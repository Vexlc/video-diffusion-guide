"""用 Python 标准库复现四个小实验。先看 run_noise，再看 integrate。"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from draw import save_gallery


def normal_draw(seed):
    """与离线网页相同的 Xorshift32 + Box–Muller，便于逐项核对。"""
    state = seed & 0xFFFFFFFF or 0x9E3779B9

    def uniform():
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return (state + 0.5) / 2**32

    def normal():
        return math.sqrt(-2 * math.log(uniform())) * math.cos(math.tau * uniform())

    return normal


def schedule(tick):
    angle = tick * math.acos(0.01)
    return math.cos(angle), math.sin(angle)


def clean_pixel(row, col):
    wheel = any((row-19)**2 + (col-center)**2 <= 5 for center in (17, 31))
    body = 13 <= row <= 18 and 11 <= col <= 36
    roof = 9 <= row <= 13 and 17+(13-row) <= col <= 30-(13-row)
    if wheel:
        return -0.8
    if body or roof:
        return 0.8
    if row == 23 and col % 11 < 6:
        return 0.1
    return -0.3 if row > 20 else -0.55


def run_noise(seed, bias=0.15):
    draw = normal_draw(seed + 34171)
    original = [clean_pixel(row, col) for row in range(30) for col in range(48)]
    noise = [draw() for pixel in original]
    rows = []
    for step in (20, 60, 90):
        signal, sigma = schedule(step / 100)
        mixed = [signal * clean + sigma * eps for clean, eps in zip(original, noise)]
        restored = [(pixel - sigma * (eps + bias)) / signal for pixel, eps in zip(mixed, noise)]
        rows.append((step, mixed, restored))
    return original, rows


def velocity(point, time):
    """两团高斯的精确条件平均速度；没有接收或猜测配对终点。"""
    variance = (1-time)**2 + (0.42*time)**2
    slope = (0.42**2 * time - (1-time)) / variance
    balance = math.tanh(time * (2*point[0] + 0.8*point[1]) / variance)
    return [slope*value + (1-slope*time)*center*balance
            for value, center in zip(point, (2, 0.8))]


def integrate(seed, end_time, steps):
    draw = normal_draw(seed + 811)
    points = [[draw(), draw()] for index in range(144)]
    dt = end_time / steps
    for step in range(steps):
        for point in points:
            speed = velocity(point, step * dt)
            for axis in range(2):
                point[axis] += dt * speed[axis]
    return points


def predict_clean(state, candidates, signal, sigma):
    """比较四条候选轨迹，越符合带噪观测的轨迹获得越高权重。"""
    logits = [-sum((value-signal*target)**2 for value, target in zip(state, path))
              / (2*sigma**2) for path in candidates]
    peak = max(logits)
    weights = [math.exp(logit-peak) for logit in logits]
    return [sum(weight * position for weight, position in zip(weights, column)) / sum(weights)
            for column in zip(*candidates)]


def sample_video(seed, joint):
    paths = [[direction*(2*frame/7-1) + offset for frame in range(8)]
             for direction in (-1, 1) for offset in (-0.35, 0.35)]
    draw = normal_draw(seed + 6697)
    state = [draw() for frame in range(8)]
    for step in range(24):
        signal, sigma = schedule(1-step/24)
        next_signal, next_sigma = schedule(1-(step+1)/24)
        if joint:
            clean = predict_clean(state, paths, signal, sigma)
        else:
            clean = [predict_clean([value], [[path[frame]] for path in paths], signal, sigma)[0]
                     for frame, value in enumerate(state)]
        noise = [(value-signal*target) / sigma for value, target in zip(state, clean)]
        state = [next_signal*target + next_sigma*eps for target, eps in zip(clean, noise)]
    return state


BASE_WEIGHT = 0.6 / (0.8**2 * (1+0.25**2) + 0.6**2)


def train_control(updates):
    """只训练两个初始为零的权重，老模型 BASE_WEIGHT 不动。"""
    draw = normal_draw(40323)
    examples = []
    for index in range(2048):
        guide = draw()
        clean = guide + 0.25 * draw()
        eps = draw()
        observed = 0.8*clean + 0.6*eps
        examples.append((observed, guide, eps))
    weight_input, weight_guide = 0.0, 0.0
    for step in range(updates):
        grad_input, grad_guide = 0.0, 0.0
        for observed, guide, eps in examples:
            guess = BASE_WEIGHT*observed + weight_input*observed + weight_guide*guide
            error = guess - eps
            grad_input += 2 * error * observed
            grad_guide += 2 * error * guide
        weight_input -= 0.16 * grad_input / len(examples)
        weight_guide -= 0.16 * grad_guide / len(examples)
    return weight_input, weight_guide


def use_control(seed, weights):
    draw = normal_draw(seed + 941)
    guide = [-1, -0.7, -0.15, 0.65, 1, 0.55, -0.25, -0.8]
    truth = [value + 0.25 * draw() for value in guide]
    observed = [0.8 * value + 0.6 * draw() for value in truth]
    base = [(value - 0.6*BASE_WEIGHT*value)/0.8 for value in observed]
    weight_input, weight_guide = weights

    def correct(condition):
        return [start - 0.6/0.8 * (weight_input*value + weight_guide*mark)
                for start, value, mark in zip(base, observed, condition)]

    return truth, guide, base, correct(guide), correct([guide[0]] + guide[:-1])


def run(seed, flow_steps, updates, output, bias=0.15):
    original, noise_rows = run_noise(seed, bias)
    clouds = [integrate(seed, end, steps) for end, steps in
              [(0, 1), (0.5, 32), (1, 1), (1, flow_steps)]]
    reference = integrate(seed, 1, 256)
    tracks = [sample_video(seed, joint) for joint in (False, True)]
    weights = train_control(updates)
    controls = use_control(seed, weights)
    truth = controls[0]
    metrics = {
        "seed": seed, "flow_steps": flow_steps, "updates": updates, "bias": bias,
        "A_pixel_error_k90": abs(noise_rows[-1][2][15*48+24] - original[15*48+24]),
        "B_mean_distance_to_reference": [mean(math.dist(point, ref) for point, ref in zip(cloud, reference))
                                         for cloud in clouds[2:]],
        "C_mean_absolute_second_difference": [mean(abs(after-2*now+before)
            for before, now, after in zip(track, track[1:], track[2:])) for track in tracks],
        "D_position_MAE": [mean(abs(guess-target) for guess, target in zip(estimate, truth))
                           for estimate in controls[2:]], "D_weights": weights,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_gallery(output, metrics, original, noise_rows, clouds, tracks, controls)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="复现像素加噪、速度场、视频采样和条件支路")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--flow-steps", type=int, default=32, help="B 的第二种走法，1 至 128")
    parser.add_argument("--updates", type=int, default=80, help="D 的训练步数，0 至 120")
    parser.add_argument("--bias", type=float, default=0.15, help="A 的噪声预测偏差，0 至 0.5")
    parser.add_argument("--out", type=Path, help="默认 replay/seed-种子-f步数-u训练次数")
    args = parser.parse_args()
    if not 0 <= args.seed <= 999999 or not 1 <= args.flow_steps <= 128 or not 0 <= args.updates <= 120 or not 0 <= args.bias <= 0.5:
        parser.error("seed 需在 0..999999，flow-steps 在 1..128，updates 在 0..120，bias 在 0..0.5")
    suffix = "" if args.bias == 0.15 else f"-b{args.bias}"
    output = args.out or Path("replay") / f"seed-{args.seed}-f{args.flow_steps}-u{args.updates}{suffix}"
    metrics = run(args.seed, args.flow_steps, args.updates, output, args.bias)
    print(json.dumps(metrics, indent=2))
    print("双击查看图解：", (output / "index.html").resolve())


if __name__ == "__main__":
    main()
