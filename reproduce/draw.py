"""绘图与计算分开：只把实际算出的数组画成 SVG，不需要安装绘图库。"""

import base64
import struct
import zlib
from html import escape


INK, TEAL, RUST = "#193540", "#00695f", "#ad492c"
MUTED, LIGHT, GRID = "#495f66", "#f1f6f2", "#c8d7d0"
WIDTH = 660
NAMES = ["a-noise", "b-flow", "c-video", "d-control"]
TITLES = ["雪花盖住车以后，还认得出它吗？", "给一群点指路，一步够不够？",
          "每一帧都是车，为什么播起来会跳？", "路标放错一帧，小车会去哪儿？"]


def text(left, top, content, size=15, color=INK, anchor="start", bold=False):
    return (f'<text x="{left}" y="{top}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{700 if bold else 400}">{escape(str(content))}</text>')


def line(left, top, right, bottom, color=GRID, width=1, dash=""):
    return (f'<line x1="{left}" y1="{top}" x2="{right}" y2="{bottom}" stroke="{color}" '
            f'stroke-width="{width}" stroke-dasharray="{dash or "none"}"/>')


def rect(left, top, width, height, fill, radius=0):
    return f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="{radius}" fill="{fill}"/>'


def circle(left, top, radius, color):
    return f'<circle cx="{left}" cy="{top}" r="{radius}" fill="{color}"/>'


def svg(title, height, pieces):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" role="img" '
            f'aria-label="{escape(title)}" font-family="MiSans, Noto Sans CJK SC, Microsoft YaHei, sans-serif">'
            f'<title>{escape(title)}</title>' + rect(0, 0, WIDTH, height, "#fbfcf8")
            + "".join(pieces) + '</svg>')


def grayscale_uri(pixels):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind+payload))

    gray = bytes(max(0, min(255, int((pixel+1)*127.5+0.5))) for pixel in pixels)
    raw = b"".join(b"\x00" + gray[start:start+48] for start in range(0, len(gray), 48))
    header = struct.pack(">IIBBBBB", 48, 30, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def noise_plate(original, rows, bias):
    pieces = [text(20, 32, "同一辆车　·　同一片雪花", 19, TEAL, bold=True),
              text(20, 58, "从左往右：车的信号越来越弱。上下两排看的是同一批像素。", 14)]
    for index, (step, mixed, restored) in enumerate(rows):
        left = 32 + 210*index
        pieces.append(text(left+88, 92, f"噪声步 {step}", 16, INK, "middle", True))
        for top, pixels in [(107, mixed), (254, restored)]:
            pieces.append(f'<image x="{left}" y="{top}" width="176" height="110" '
                          f'style="image-rendering:pixelated" href="{grayscale_uri(pixels)}"/>')
        pieces.append(text(left+88, 388, f"像素偏差 {abs(restored[744]-original[744]):.3f}", 15, RUST, "middle"))
    caption = f"↓ 都用“多猜了 {bias:g}”的噪声反推一次" if bias else "↓ 都用准确的噪声反推一次"
    pieces += [text(32, 241, caption, 15, MUTED)]
    return svg(TITLES[0], 407, pieces)


def flow_plate(clouds, steps):
    pieces = []
    extent = max(3.7, max(abs(value) for cloud in clouds for point in cloud for value in point)*1.04)
    captions = ["① 出发：一团噪声", "② 走到半程：开始分流", "③ 只问一次路：一步到底", f"④ 每走一段再问：{steps} 步到底"]
    for index, (cloud, caption) in enumerate(zip(clouds, captions)):
        left, top = 9+(index%2)*324, 6+(index//2)*184
        pieces += [rect(left, top, 314, 176, LIGHT, 7), text(left+12, top+26, caption, 16, TEAL, bold=True)]
        center_x, center_y, scale = left+155, top+101, 68/extent
        pieces += [line(left+12, center_y, left+302, center_y), line(center_x, top+34, center_x, top+167)]
        for sign in (-1, 1):
            pos_x, pos_y = center_x+sign*2*scale, center_y-sign*0.8*scale
            pieces += [line(pos_x-6, pos_y, pos_x+6, pos_y, RUST, 2),
                       line(pos_x, pos_y-6, pos_x, pos_y+6, RUST, 2)]
        pieces.extend(circle(center_x+point[0]*scale, center_y-point[1]*scale, 2, TEAL) for point in cloud)
        if len({tuple(point) for point in cloud}) == 1:
            pieces += [line(center_x+3, center_y+3, center_x+22, center_y+26, INK),
                       text(center_x+25, center_y+34, "144 点重合", 13)]
    pieces += [text(18, 397, "青点 = 生成样本　　橙色十字 = 最终两团的中心　　四格使用同一坐标尺度", 14)]
    return svg(TITLES[1], 410, pieces)


def tiny_car(center, baseline, color, scale=1):
    return (f'<g transform="translate({center} {baseline}) scale({scale})">' +
            rect(-10, -6, 20, 6, color, 1.6) + rect(-5, -10, 10, 6, color, 1.4) +
            circle(-6, 0, 2.3, INK) + circle(6, 0, 2.3, INK) + '</g>')


def video_plate(tracks):
    pieces = [text(16, 28, "先看胶片：每格是同一辆车的一张照片", 18, TEAL, bold=True)]
    for row, (track, caption, color) in enumerate(zip(tracks, ("分开猜", "一起猜"), (RUST, TEAL))):
        top = 49 + 57*row
        pieces.append(text(12, top+26, caption, 15, color, bold=True))
        for frame, position in enumerate(track):
            left = 79+72*frame
            pieces += [rect(left, top, 67, 42, LIGHT, 3), line(left+2, top+29, left+65, top+29)]
            pieces.append(tiny_car(left+33+17*position, top+28, color, 0.82))
    pieces += [text(16, 191, "再连车心：线往上表示车在画面里更靠右", 18, TEAL, bold=True)]
    for value in (-1, 0, 1):
        top = 284-44*value
        pieces += [line(82, top, 615, top), text(70, top+5, str(value), 14, MUTED, "end")]
    for frame in range(8):
        left = 85+75*frame
        pieces.append(text(left, 373, f"帧 {frame}", 13, MUTED, "middle"))
    for track, color, dash in zip(tracks, (RUST, TEAL), ("5 3", "")):
        points = [(85+75*frame, 284-44*position) for frame, position in enumerate(track)]
        for start, end in zip(points, points[1:]):
            pieces.append(line(*start, *end, color, 2.5, dash))
        pieces.extend(circle(*point, 3, color) for point in points)
    pieces += [text(20, 402, "橙色虚线：各猜各的　　青色实线：合起来猜一条轨迹", 14)]
    return svg(TITLES[2], 416, pieces)


def control_plate(controls):
    truth, guide, base, aligned, delayed = controls
    frame = 3
    values = [truth[frame], guide[frame], guide[frame-1], base[frame], aligned[frame], delayed[frame]]
    extent = max(1.55, max(map(abs, values))+0.12)
    def project(value):
        return 417 + 208*value/extent

    pieces = [text(18, 31, "把镜头停在第 3 帧：× 是真实车心", 19, TEAL, bold=True),
              text(18, 57, "三条路用同一条位置刻度；只改变模型看到的路标。", 14)]
    for row, (caption, position, mark) in enumerate([
            ("① 不看路标", base[frame], None), ("② 路标对齐", aligned[frame], guide[frame]),
            ("③ 路标迟一帧", delayed[frame], guide[frame-1])]):
        top = 76+row*96
        pieces += [rect(10, top, 640, 86, LIGHT, 6), text(21, top+31, caption, 16, TEAL, bold=True),
                   text(21, top+58, f"本帧误差 {abs(position-truth[frame]):.3f}", 13, MUTED)]
        baseline = top+52
        pieces += [line(188, baseline, 641, baseline, GRID, 2),
                   line(project(0), baseline+5, project(0), baseline+12)]
        if mark is not None:
            left = project(mark)
            pieces += [line(left, top+8, left, baseline, RUST, 1.5, "3 2"),
                       f'<path d="M{left},{top+8} h14 l-4,5 4,5 h-14z" fill="{RUST}"/>']
        left = project(truth[frame])
        pieces += [line(left-4, baseline+4, left+4, baseline+12, INK, 2),
                   line(left+4, baseline+4, left-4, baseline+12, INK, 2),
                   tiny_car(project(position), baseline-1, TEAL, 1.35)]
    pieces += [text(20, 385, "小车 = 模型估计　　路旗 = 输入条件　　× = 真实位置（只用来核对）", 14)]
    return svg(TITLES[3], 400, pieces)


def save_gallery(output, metrics, original, rows, clouds, tracks, controls):
    plates = [noise_plate(original, rows, metrics["bias"]), flow_plate(clouds, metrics["flow_steps"]),
              video_plate(tracks), control_plate(controls)]
    descriptions = [
        f'上排加噪，下排用偏差为 {metrics["bias"]:g} 的噪声估计倒推。偏差为零时可还原原车；不为零时，高噪声会放大误差。',
        "看左下和右下：从同一批点出发，改变问路次数。把 --flow-steps 改为 1，两格应重合。",
        "沿虚线读八帧的位置，再沿实线读。数字的二阶差分只适合本例匀速轨迹，停着不动也会得零。",
        "看车心与 × 的距离，再看旗子。将 --updates 改为 0，三行小车应在同一位置；路旗本身不会移动车。",
    ]
    cards = []
    for name, title, plate, description in zip(NAMES, TITLES, plates, descriptions):
        (output / f"{name}.svg").write_text(plate, encoding="utf-8")
        cards.append(f'<section><h2>{title}</h2><img src="{name}.svg" alt="{title}"><p>{description}</p></section>')
    meta = f'种子 {metrics["seed"]}　|　流匹配终点对照 {metrics["flow_steps"]} 步　|　条件支路更新 {metrics["updates"]} 次'
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>我的生成模型实验图册</title><style>body{background:#edf3ee;color:#193540;font:17px/1.65 sans-serif;'
            'max-width:800px;margin:auto;padding:20px}section{background:white;padding:16px 22px;margin:20px 0}'
            'img{display:block;width:100%}h2{font-size:23px}a{color:#00695f}</style><h1>我的生成模型实验图册</h1>'
            f'<p>{meta}　·　<a href="metrics.json">查看原始指标</a></p>'+''.join(cards)+'</html>')
    (output / "index.html").write_text(html, encoding="utf-8")
