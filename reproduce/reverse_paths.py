"""在同一组八帧潜变量上，对照 DDPM 的随机反向步与 eta=0 的 DDIM。"""

import math
import random

from reproduce import predict_clean, schedule


PATHS = [[direction*(2*frame/7-1)+offset for frame in range(8)]
         for direction in (-1, 1) for offset in (-0.35, 0.35)]
ROUTES = ["向左・低偏移", "向左・高偏移", "向右・低偏移", "向右・高偏移"]


def reverse_step(state, clean, tick, total, sampler, rng):
    signal, sigma = schedule(tick/total)
    prev_signal, prev_sigma = schedule((tick-1)/total)
    if sampler == "DDIM":
        noise = [(value-signal*estimate)/sigma for value, estimate in zip(state, clean)]
        return [prev_signal*estimate + prev_sigma*eps for estimate, eps in zip(clean, noise)]
    if sampler != "DDPM":
        raise ValueError("unknown sampler")
    alpha = (signal/prev_signal)**2
    beta = 1-alpha
    step_sigma = math.sqrt(beta * prev_sigma**2 / sigma**2)
    means = [(prev_signal*beta*estimate + math.sqrt(alpha)*prev_sigma**2*value)/sigma**2
             for value, estimate in zip(state, clean)]
    return [center + step_sigma*rng.gauss(0, 1) for center in means] if tick > 1 else means


def sample(initial, sampler, reverse_seed):
    state = list(initial)
    rng = random.Random(reverse_seed)
    for tick in range(24, 0, -1):
        signal, sigma = schedule(tick/24)
        clean = predict_clean(state, PATHS, signal, sigma)
        state = reverse_step(state, clean, tick, 24, sampler, rng)
    distances = [math.dist(state, route) / math.sqrt(len(state)) for route in PATHS]
    nearest = min(range(len(PATHS)), key=distances.__getitem__)
    return {"route": nearest, "distance": distances[nearest], "track": state}


def compare(seed, count=96):
    initial_rng = random.Random(seed+803)
    starts = [[initial_rng.gauss(0, 1) for frame in range(8)] for repeat in range(count)]
    records = {}
    for sampler in ("DDIM", "DDPM"):
        for start_mode in ("fixed", "fresh"):
            cases = [sample(starts[0 if start_mode == "fixed" else repeat], sampler, seed+200003+repeat)
                     for repeat in range(count)]
            records[sampler+"_"+start_mode] = {
                "counts": [sum(case["route"] == route for case in cases) for route in range(4)],
                "max_rms": max(case["distance"] for case in cases), "cases": cases}
    return records
