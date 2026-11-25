"""读取实验 P 的一段可控影片，分别量生成速度、片中位移与两帧的局部匹配代价。"""
import argparse
import base64
import json
import struct
from pathlib import Path

from draw import INK, MUTED, RUST, TEAL, line, rect, svg, text
from pixel_report import bitmap

W, H, FRAMES, STEPS, TOKEN = 16, 12, 6, 24, 8
WINDOWS = {"point":(0,0), "strip":(1,0), "square":(1,1)}
TITLES = {"point":"一个像素", "strip":"横排三格", "square":"整块九格"}


def probe_record(data):
    """抽取种子 9317、强度 1 的两条路线，保留全部解码字节和第 8 号块。"""
    rec = data["records"]["1.0"]
    px = base64.b64decode(rec["pixels"])
    size = (STEPS+1)*FRAMES*H*W
    def token_values(key, case, steps):
        raw = base64.b64decode(rec[key]["data"])
        return [struct.unpack_from("<4f", raw, 4*((case*steps+n)*18+TOKEN)*4) for n in range(steps)]
    routes = {}
    for case, key in enumerate(("left","right")):
        routes[key] = {"pixels":base64.b64encode(px[case*size:(case+1)*size]).decode(),
                       "centers":rec["centers"][case], "target":data["examples"][case]["target"],
                       "x":token_values("x",case,STEPS+1), "u":token_values("v",case,STEPS)}
    return routes


def picture(route, step, frame):
    start = (step*FRAMES+frame)*W*H
    raw = base64.b64decode(route["pixels"])
    return [list(raw[start+y*W:start+(y+1)*W]) for y in range(H)]


def two_times(route, step, frame, component, fps, pixel):
    """同一潜分量沿生成进度取差；成片中同一物体沿相邻视频帧取差。"""
    col,row = pixel
    x = route["x"][step][component]
    u = route["u"][step][component]
    current, following = route["centers"][frame:frame+2]
    dx,dy = (b-a for a,b in zip(current,following))
    grid = [[picture(route,n,f)[row][col] for f in (frame,frame+1)] for n in (step,step+1,STEPS)]
    return {"x":x, "u":u, "next_x":route["x"][step+1][component], "delta_x":u/STEPS,
            "centers":[current,following], "pixels_per_frame":[dx,dy],
            "pixels_per_second":[dx*fps,dy*fps], "brightness":grid}


def reference_frame(center):
    col,row = center
    return [[255 if abs(x-col)<=1 and abs(y-row)<=1 else 0 for x in range(W)] for y in range(H)]


def match_window(a,b,at,window):
    """枚举整数位移。每个候选只比较两帧之间相同形状的窗口，保留全部并列最优解。"""
    x,y = at
    rx,ry = WINDOWS[window]
    def fits(col,row):
        return rx<=col<W-rx and ry<=row<H-ry
    if not fits(x,y):
        raise ValueError("源窗口超出图像")
    costs = []
    for dy in range(-2,3):
        row = []
        for dx in range(-2,3):
            if fits(x+dx,y+dy):
                cost = sum(abs(a[y+j][x+i]-b[y+dy+j][x+dx+i])
                           for j in range(-ry,ry+1) for i in range(-rx,rx+1))
            else:
                cost = None
            row.append(cost)
        costs.append(row)
    lowest = min(val for row in costs for val in row if val is not None)
    best = [[i-2,j-2] for j,row in enumerate(costs) for i,val in enumerate(row) if val==lowest]
    return {"costs":costs, "lowest":lowest, "best":best}


def pair(route, frame, source):
    if source=="ideal":
        return [reference_frame(c) for c in route["target"][frame:frame+2]]
    return [picture(route,STEPS,f) for f in (frame,frame+1)]


def outline(x,y,w,h,color):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{color}" stroke-width="1.7"/>'


def grid_figure(route):
    col,row = 9,3
    sample = two_times(route,12,2,1,6,(col,row))
    parts=[text(12,25,"固定屏幕上的 (9,3)，读出框内的灰度",18,TEAL,bold=True),
           text(167,52,"帧 2",15,INK,"middle",True),text(319,52,"帧 3",15,INK,"middle",True),
           text(420,52,"竖读 · 第 2 帧",16,TEAL,bold=True)]
    for r,n in enumerate((12,13,24)):
        top=64+81*r
        parts.append(text(12,top+32,f"已走 {n} 步",15,MUTED))
        for c,f in enumerate((2,3)):
            left=116+c*151
            parts.extend((bitmap(picture(route,n,f),left,top,88),outline(left+col*5.5,top+row*5.5,5.5,5.5,"#ffae50"),
                          text(left+95,top+37,sample["brightness"][r][c],15,INK,bold=True)))
    a,b = (row[0] for row in sample["brightness"][:2])
    start,end=sample["centers"]
    track=f"({start[0]},{start[1]}) → ({end[0]},{end[1]})"
    parts.extend((text(420,89,f"{a} → {b}",22,TEAL,bold=True),text(420,117,"同一像素，隔一生成步。",15,MUTED),
                  line(420,147,640,147,"#d2ded4"),text(420,178,"横读 · 第 24 步",16,RUST,bold=True),
                  text(420,212,track,19,INK,bold=True),text(420,240,"亮块中心，隔一视频帧。",15,MUTED),
                  text(12,317,"橙框始终钉在同一屏幕坐标。框外的数字为解码后的显示灰度，0 黑、255 白。",14,MUTED)))
    return svg("两只时钟的三行两列真实解码图和同一像素的灰度",332,parts)


def patch_figure(route):
    frames=pair(route,2,"ideal")
    parts=[text(12,25,"从帧 2 的中心出发，去帧 3 找同样的局部图案",17,TEAL,bold=True)]
    for f,left in enumerate((24,211)):
        parts.extend((bitmap(frames[f],left,40,112),text(left+121,68,f"帧 {f+2}",14,INK,bold=True)))
        col,row=route["target"][2+f]
        parts.append(outline(left+7*col,40+7*row,7,7,"#ffad48"))
    parts.extend((text(406,64,"每张表有 25 个候选位移",15,INK,bold=True),
                  text(406,91,"横轴 Δ列、纵轴 Δ行，单位为格。",14,MUTED),
                  text(406,116,"深色 = 并列最低误差。",14,MUTED)))
    for panel,key in enumerate(WINDOWS):
        left=24+218*panel
        result=match_window(*frames,route["target"][2],key)
        parts.append(text(left,157,TITLES[key],16,TEAL,bold=True))
        for i in range(5):
            parts.extend((text(left+45+25*i,182,i-2,12,MUTED,"middle"),
                          text(left+23,204+24*i,i-2,12,MUTED,"middle")))
        for j,row in enumerate(result["costs"]):
            for i,val in enumerate(row):
                hit=val==result["lowest"]
                parts.append(rect(left+33+25*i,188+24*j,23,22,TEAL if hit else "#e4ede6",2))
                if hit:
                    parts.append(text(left+45+25*i,204+24*j,"0",12,"white","middle",True))
        parts.append(text(left,336,f'{len(result["best"])} 个位移同样吻合',15,INK,bold=True))
    return svg("同一中心从单像素、三像素到九像素得到九、三、一种零误差位移",349,parts)


def render(data,out):
    routes=probe_record(data)
    route=routes["right"]
    for name,figure in (("film-generating-time",grid_figure(route)),("local-motion",patch_figure(route))):
        (out/(name+".svg")).write_text(figure,encoding="utf-8")
    template=Path(__file__).with_name("motion_ui.html").read_text(encoding="utf-8")
    payload=json.dumps(routes,separators=(",",":"),allow_nan=False)
    (out/"motion.html").write_text(template.replace("__ROUTES__",payload),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path,help="实验 P 的 numbers.json 所在目录")
    parser.add_argument("--direction",choices=("left","right"),default="right")
    parser.add_argument("--step",type=int,choices=range(24),default=12)
    parser.add_argument("--frame",type=int,choices=range(5),default=2)
    parser.add_argument("--component",type=int,choices=range(4),default=1)
    parser.add_argument("--fps",type=int,choices=(3,6,12),default=6)
    parser.add_argument("--pixel",type=int,nargs=2,metavar=("COL","ROW"),default=(9,3))
    parser.add_argument("--window",choices=WINDOWS,default="square")
    parser.add_argument("--source",choices=("ideal","decoded"),default="ideal")
    parser.add_argument("--anchor",choices=("center","background"),default="center")
    args=parser.parse_args()
    if not (0<=args.pixel[0]<W and 0<=args.pixel[1]<H):
        parser.error("读数点的列为 0–15，行为 0–11")
    route=probe_record(json.loads((args.directory/"numbers.json").read_text()))[args.direction]
    print(json.dumps(two_times(route,args.step,args.frame,args.component,args.fps,args.pixel),indent=2))
    anchor=route["target"][args.frame] if args.anchor=="center" else (5,8)
    result=match_window(*pair(route,args.frame,args.source),anchor,args.window)
    print(json.dumps(result,indent=2))
