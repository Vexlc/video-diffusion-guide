"""运行 E–H 并导出能独立打开的交互报告、插图和机器可读指标。"""

import argparse
import json
from pathlib import Path

from workshop_figures import (attention_map, changed_plate, control_contact, motion_distribution,
                              route_counts, sampler_film, ticket, velocity_profile)
from workshop_models import attention, decode, fm_workshop, spatial_workshop
from reverse_paths import compare


def run(seed, samples, updates, output):
    output.mkdir(parents=True, exist_ok=True)
    fm = fm_workshop(seed, samples)
    spatial = spatial_workshop(seed, updates)
    checkpoints = fm["stages"]
    stages = [{**{key: stage[key] for key in ("count", "gap", "loss", "w1")},
               "track": decode(stage["points"][0]), "figure": motion_distribution(fm, stage)}
              for stage in checkpoints]
    attention_cases = [attention(False), attention(True)]
    reverse = compare(seed)
    payload = {
        "seed": seed, "updates": updates, "stages": stages,
        "profiles": [velocity_profile(fm, index) for index in range(3)],
        "attention": [attention_map(case, bool(index)) for index, case in enumerate(attention_cases)],
        "votes": [case[0] for case in attention_cases],
        "frames": [control_contact(arrays) for arrays in spatial["frames"]],
        "edges": [changed_plate(arrays, spatial["metrics"]) for arrays in spatial["frames"]],
        "spatial": spatial["metrics"], "reverse": reverse,
    }
    figures = {"training-ticket": ticket(), "velocity": payload["profiles"][1],
               "motion": stages[-1]["figure"], "attention-unique": payload["attention"][0],
               "attention-ambiguous": payload["attention"][1], "spatial": payload["frames"][3],
               "spatial-edge": payload["edges"][3], "reverse-film": sampler_film(reverse),
               "reverse-counts": route_counts(reverse)}
    for name, figure in figures.items():
        (output / (name + ".svg")).write_text(figure, encoding="utf-8")
    metrics = {"seed": seed, "samples": samples, "updates": updates,
               "fm": [{key: stage[key] for key in ("count", "loss", "w1", "gap")} for stage in checkpoints],
               "spatial": spatial["metrics"], "weights": list(spatial["weights"]),
               "attention": attention_cases, "reverse": {key: {field: record[field] for field in ("counts", "max_rms")}
                                                        for key, record in reverse.items()}}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    template = Path(__file__).with_name("workshop_ui.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template.replace("__WORKSHOP_DATA__", json.dumps(payload,
        separators=(",", ":"), ensure_ascii=False).replace("</script", "<\\/script")), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=32768, help="流匹配训练对数（0..262144）")
    parser.add_argument("--updates", type=int, default=80, help="空间条件支路更新次数（0..240）")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 0 <= args.samples <= 262144 or not 0 <= args.updates <= 240:
        parser.error("请检查 --samples 和 --updates 的范围")
    output = args.out or Path("replay") / f"workshop-{args.seed}-n{args.samples}-u{args.updates}"
    metrics = run(args.seed, args.samples, args.updates, output)
    print("FM 的留出速度损失：", [round(stage["loss"], 4) for stage in metrics["fm"]])
    print("对齐 / 迟一帧的变化区域 MAE：", *(round(metrics["spatial"]["changed"][key], 4) for key in ("aligned", "late")))
    print("双击打开：", (output / "index.html").resolve())


if __name__ == "__main__":
    main()
