"""三个可独立阅读的小实验：学速度、学空间条件、算注意力。仅用标准库。"""

import math
import random
from statistics import mean


KNOTS = 33
TIMES = 13
LIMIT = 3.2
SIGNAL = 0.55
SIGMA = math.sqrt(1 - SIGNAL**2)
BASE = SIGMA / (SIGNAL**2 * (0.7**2 + 0.12**2) + SIGMA**2)


def position_pair(rng):
    return [0.24*rng.gauss(0, 1), rng.choice((-1, 1))*1.05 + 0.14*rng.gauss(0, 1)]


def training_pair(rng):
    clean = position_pair(rng)
    noise = [rng.gauss(0, 1) for axis in range(2)]
    time = rng.random()
    middle = [(1-time)*start + time*end for start, end in zip(noise, clean)]
    return middle, time, [end-start for start, end in zip(noise, clean)]


def neighbors(value, time):
    column = max(0, min(KNOTS-1, (value + LIMIT)*(KNOTS-1)/(2*LIMIT)))
    row = time*(TIMES-1)
    left, top = min(KNOTS-2, int(column)), min(TIMES-2, int(row))
    for time_step in (0, 1):
        for space_step in (0, 1):
            fraction = (1-abs(row-top-time_step)) * (1-abs(column-left-space_step))
            yield (top+time_step)*KNOTS + left+space_step, fraction


def train_tables(seed, samples):
    rng = random.Random(seed)
    totals = [[0.0]*(TIMES*KNOTS) for axis in range(2)]
    counts = [[0.0]*(TIMES*KNOTS) for axis in range(2)]
    for example in range(samples):
        middle, time, speed = training_pair(rng)
        for axis in range(2):
            for cell, weight in neighbors(middle[axis], time):
                totals[axis][cell] += weight*speed[axis]
                counts[axis][cell] += weight
    return [[total/(count+0.1) for total, count in zip(axis_totals, axis_counts)]
            for axis_totals, axis_counts in zip(totals, counts)]


def predict(table, value, time):
    return sum(weight*table[cell] for cell, weight in neighbors(value, time))


def advance(tables, point, steps=32):
    result = list(point)
    for step in range(steps):
        for axis, table in enumerate(tables):
            result[axis] += predict(table, result[axis], step/steps) / steps
    return result


def decode(point):
    center, motion = point
    return [center + motion*(2*frame/7-1) for frame in range(8)]


def fm_workshop(seed, samples):
    starts_rng, target_rng = random.Random(seed+101), random.Random(seed+202)
    starts = [[starts_rng.gauss(0, 1), starts_rng.gauss(0, 1)] for clip in range(256)]
    target = [position_pair(target_rng) for clip in range(256)]
    check_rng = random.Random(seed+303)
    heldout = [training_pair(check_rng) for example in range(1536)]
    stages = []
    for count in sorted({0, min(512, samples), samples}):
        tables = train_tables(seed, count)
        points = [advance(tables, point) for point in starts]
        loss = mean((predict(tables[axis], state[axis], time)-speed[axis])**2
                    for state, time, speed in heldout for axis in range(2))
        sorted_motion = sorted(point[1] for point in points)
        target_motion = sorted(point[1] for point in target)
        stages.append({
            "count": count, "loss": loss, "points": points,
            "w1": mean(abs(first-second) for first, second in zip(sorted_motion, target_motion)),
            "gap": sum(abs(point[1]) < 0.5 for point in points),
            "profiles": [[predict(tables[1], value/20, tick) for value in range(-42, 43)]
                         for tick in (0.25, 0.5, 0.75)],
        })
    return {"target": target, "stages": stages}


def train_spatial(seed, updates):
    rng = random.Random(seed+401)
    examples = []
    for index in range(2048):
        mask = rng.choice((-1, 1))
        clean = 0.7*mask + 0.12*rng.gauss(0, 1)
        noise = rng.gauss(0, 1)
        examples.append((SIGNAL*clean + SIGMA*noise, mask, noise))
    weight_image = weight_mask = 0.0
    for step in range(updates):
        grad_image = grad_mask = 0.0
        for observed, mask, noise in examples:
            error = BASE*observed + weight_image*observed + weight_mask*mask - noise
            grad_image += error*observed
            grad_mask += error*mask
        weight_image -= 0.3*grad_image / len(examples)
        weight_mask -= 0.3*grad_mask / len(examples)
    return weight_image, weight_mask


def car_mask(row, col, frame):
    local = col - (6 + 4*frame)
    in_car = (-5 <= local <= 6 and 13 <= row <= 17) or (-2 <= local <= 3 and 10 <= row <= 12)
    return 1 if in_car else -1


def spatial_workshop(seed, updates):
    weights = train_spatial(seed, updates)
    rng = random.Random(seed+509)
    frames = []
    for frame in range(8):
        arrays = {name: [] for name in ("truth", "observed", "mask", "late_mask", "base", "aligned", "late")}
        for row in range(30):
            for col in range(48):
                mask = car_mask(row, col, frame)
                late_mask = car_mask(row, col, max(frame-1, 0))
                truth = 0.7*mask + 0.12*rng.gauss(0, 1)
                observed = SIGNAL*truth + SIGMA*rng.gauss(0, 1)
                base = (observed-SIGMA*BASE*observed) / SIGNAL

                def corrected(condition):
                    return base - SIGMA/SIGNAL * (weights[0]*observed + weights[1]*condition)

                for key, value in zip(arrays, (truth, observed, mask, late_mask, base,
                                               corrected(mask), corrected(late_mask))):
                    arrays[key].append(value)
        frames.append(arrays)
    names = ("base", "aligned", "late")
    metrics = {}
    for region in ("all", "changed"):
        metrics[region] = {
            key: mean(abs(value-target) for arrays in frames
                      for value, target, mask, late in zip(arrays[key], arrays["truth"], arrays["mask"], arrays["late_mask"])
                      if region == "all" or mask != late) for key in names}
    metrics["n_changed"] = sum(mask != late for arrays in frames for mask, late in zip(arrays["mask"], arrays["late_mask"]))
    return {"frames": frames, "weights": weights, "metrics": metrics}


def attention(ambiguous=False):
    queries = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    keys = [queries[2], queries[0], queries[1]]
    if ambiguous:
        keys = [queries[0], queries[0], queries[1]]
    weights = []
    for query in queries:
        scores = [sum(first*second for first, second in zip(query, key))/math.sqrt(3) for key in keys]
        mass = [math.exp(score-max(scores)) for score in scores]
        weights.append([value/sum(mass) for value in mass])
    return weights
