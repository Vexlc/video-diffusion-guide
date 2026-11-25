"""在同一组输入上比较扩散与直线混合。用四张等可能的题卡手算方差，再用高斯抽样复核。"""
import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import fmean

from draw import INK, MUTED, RUST, TEAL, circle, line, svg, text
from reproduce import normal_draw

CASES = ("independent", "fixed", "shared")
COUNT = 4096


def inputs(mode, y, noise):
    if mode == "independent":
        return y, noise
    if mode == "fixed":
        return .8, noise
    if mode == "shared":
        return y, y
    raise ValueError("未知输入方式")


def variance(values):
    """总体方差：每个结果减平均数，差的平方再取平均。"""
    center = fmean(values)
    return fmean((x - center) ** 2 for x in values)


def sign_cards(mode):
    if mode == "independent":
        return [(y, e) for y in (-1., 1.) for e in (-1., 1.)]
    return [inputs(mode, sign, sign) for sign in (-1., 1.)]


def coefficients(share):
    if not 0 <= share <= 1:
        raise ValueError("配比应在 0 到 1 之间")
    return ((sqrt(share), sqrt(1 - share)), (share, 1 - share))


def gaussian_pairs(seed):
    draw = normal_draw(seed)
    return [(draw(), draw()) for _ in range(COUNT)]


def compare(share, mode, pairs):
    cards = sign_cards(mode)
    trials = [inputs(mode, y, e) for y, e in pairs]
    results = []
    for a, b in coefficients(share):
        card_x = [a * y + b * e for y, e in cards]
        gaussian_x = [a * y + b * e for y, e in trials]
        results.append({"coefficients": [a, b], "inputs": cards, "x": card_x, "mean": fmean(card_x),
                        "variance": variance(card_x), "gaussian_mean": fmean(gaussian_x),
                        "gaussian_variance": variance(gaussian_x)})
    return results


def number_line_figure():
    modes = ("开根号", "直接作系数")
    colors = (TEAL, RUST)
    parts = [text(10, 21, "两枚独立硬币，每种组合占四分之一", 16, INK, bold=True),
             text(600, 21, "总体方差", 15, MUTED, "middle")]
    # 只用四种等可能的输入，把两行结果放在同一把数轴上。
    for k, (a, b) in enumerate(coefficients(.64)):
        results = [a * y + b * e for y, e in sign_cards("independent")]
        top = 48 + 59 * k
        parts.extend([text(10, top + 3, modes[k], 15, colors[k], bold=True),
                      text(10, top + 24, f"{a:.2f} 与 {b:.2f}", 14, MUTED),
                      line(140, top, 536, top), line(338, top - 6, 338, top + 6, MUTED),
                      text(600, top + 9, f"{variance(results):.4f}".rstrip("0").rstrip("."), 19, colors[k], "middle", True)])
        for value in sorted(results):
            x = 338 + value * 130
            parts.extend([circle(x, top, 5, colors[k]),
                          text(x, top + 21, f"{value:+.2f}".replace("-", "−"), 14, INK, "middle")])
    return svg("四种等可能的混合结果。开根号后一行方差为 1，直接混合一行方差为 0.5392", 141, parts)


def render(out, seed=17, share=.64, mode="independent", *, course=False):
    coefficients(share)
    sign_cards(mode)
    out.mkdir(parents=True, exist_ok=True)
    (out / "variance-coins.svg").write_text(number_line_figure(), encoding="utf-8")
    payload = {"pairs": gaussian_pairs(seed), "seed": seed, "share": share, "mode": mode}
    template = Path(__file__).with_name("variance_ui.html").read_text(encoding="utf-8")
    nav = ('<nav><a href="../index.html">离线首页</a><a href="../page-index.html#p7">第 7 页的公式与源码</a>'
           '<a href="index.html">模型怎样读这个像素</a></nav>') if course else ""
    (out / "variance.html").write_text(template.replace("__NAV__", nav).replace(
        "__DATA__", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", type=float, default=.64, help="0 到 1 之间的干净端配比")
    parser.add_argument("--mode", choices=CASES, default="independent")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, help="另生成带当前设置的离线网页和书中的示意图")
    args = parser.parse_args()
    if not 0 <= args.share <= 1:
        parser.error("--share 应在 0 到 1 之间")
    rows = compare(args.share, args.mode, gaussian_pairs(args.seed))
    print(json.dumps(dict(zip(("开根号", "直接作系数"), rows)), ensure_ascii=False, indent=2))
    if args.out:
        render(args.out, args.seed, args.share, args.mode)
