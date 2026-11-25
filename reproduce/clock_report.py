"""实验 I 的双时钟放映台。读取随包的浮点采样记录，无需 PyTorch。"""

import argparse
import base64
import hashlib
import json
import shutil
import sys
from array import array
from pathlib import Path

from draw import INK, MUTED, TEAL, svg, text
from pixel_report import bitmap


PREFIX = "const clockRecording = "
CASE, STEP, FRAME, ROW, COL = 0, 12, 2, 5, 7


def little_floats(tensor):
    values = array("f", tensor.detach().to("cpu").contiguous().view(-1).tolist())
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


class ClockRecorder:
    """在原采样器完成每步时收集张量；保持完整批次的前向计算。"""

    def __init__(self, count):
        self.count = count
        self.states = []
        self.velocities = []
        self.coefficients = []

    def observe(self, state, predicted, following, coefficients):
        if not self.states:
            self.shape = tuple(state[:self.count].shape)
            self.states.append(little_floats(state[:self.count]))
        self.states.append(little_floats(following[:self.count]))
        self.velocities.append(little_floats(predicted[:self.count]))
        self.coefficients.append([value.flatten()[0].item() for value in coefficients])

    def write(self, output, checkpoint, seed, updates):
        def encode(parts):
            return base64.b64encode(b"".join(parts)).decode("ascii")

        data = {"shape":self.shape, "steps":len(self.velocities), "seed":seed,
                "updates":updates, "coefficients":self.coefficients,
                "weight_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "states":encode(self.states), "velocities":encode(self.velocities)}
        (output/"clock-data.js").write_text(PREFIX+json.dumps(data,separators=(",",":"))+";\n",encoding="utf-8")


class Recording:
    def __init__(self, file):
        script = file.read_text(encoding="utf-8").strip()
        if not script.startswith(PREFIX) or not script.endswith(";"):
            raise ValueError("无法识别浮点采样记录")
        data = json.loads(script[len(PREFIX):-1])
        self.cases,self.frames,self.height,self.width = data["shape"]
        self.steps = data["steps"]
        self.coefficients = data["coefficients"]
        self.weight_sha256 = data["weight_sha256"]
        self.plane = self.height*self.width
        stride = self.cases*self.frames*self.plane

        def decode(key, blocks):
            values = array("f")
            values.frombytes(base64.b64decode(data[key],validate=True))
            if sys.byteorder != "little":
                values.byteswap()
            if len(values) != blocks*stride:
                raise ValueError("记录包含的像素数与形状不符")
            return values

        self.states = decode("states",self.steps+1)
        self.velocities = decode("velocities",self.steps)
        if len(self.coefficients)!=self.steps:
            raise ValueError("每次调用都应有一组混合系数")

    def plane_offset(self, case, step, frame):
        return ((step*self.cases+case)*self.frames+frame)*self.plane

    def pixel(self, case, step, frame, row, col):
        coords = (case,step,frame,row,col)
        limits = (self.cases,self.steps+1,self.frames,self.height,self.width)
        if any(not 0 <= n < bound for n,bound in zip(coords,limits)):
            raise ValueError("起点、生成次数、影片帧或像素坐标超出记录范围")
        i = self.plane_offset(case,step,frame)+row*self.width+col
        x = self.states[i]
        if step == self.steps:
            return {"state":x}
        v = self.velocities[i]
        a,sigma,na,ns = self.coefficients[step]
        clean,noise = a*x-sigma*v, sigma*x+a*v
        return {"state":x,"v":v,"clean":clean,"noise":noise,"a":a,"sigma":sigma,
                "next_a":na,"next_sigma":ns,"computed_next":na*clean+ns*noise,
                "recorded_next":self.states[i+self.cases*self.frames*self.plane]}

    def image(self, case, step, frame, kind="state"):
        start = self.plane_offset(case,step,frame)
        values = self.states[start:start+self.plane]
        if kind == "clean":
            a,sigma,*_ = self.coefficients[step]
            values = [a*x-sigma*v for x,v in zip(values,self.velocities[start:start+self.plane])]
        return [[round(max(0,min(255,127.5*(v+1)))) for v in values[r:r+self.width]]
                for r in range(0,self.plane,self.width)]


def film_grid(recording, rows, title, *, mark_pixel=False):
    pieces = [text(12,24,title,17,TEAL,bold=True)]
    for frame in range(recording.frames):
        pieces.append(text(199+80*frame,48,f"帧 {frame}",14,TEAL if frame==FRAME else MUTED,"middle",frame==FRAME))
    for i,(label,step,kind) in enumerate(rows):
        top = 56+60*i
        pieces.append(text(12,top+29,label,15,INK,bold=True))
        for frame in range(recording.frames):
            left = 165+80*frame
            pixels = recording.image(CASE,step,frame,kind)
            pieces.append(bitmap(pixels,left,top,68))
            if frame==FRAME:
                pieces.append(f'<rect x="{left-2}" y="{top-2}" width="72" height="55" fill="none" stroke="{TEAL}" stroke-width="2"/>')
                if mark_pixel:
                    px,py = left+COL*68/recording.width,top+ROW*68/recording.width
                    pieces.append(f'<rect x="{px-1.5}" y="{py-1.5}" width="7.25" height="7.25" fill="none" stroke="#ffab3f" stroke-width="1.5"/>')
    return svg(title,57+60*len(rows),pieces)


def render(file, output):
    rec = Recording(file)
    output.mkdir(parents=True,exist_ok=True)
    if file.resolve() != (output/"clock-data.js").resolve():
        shutil.copyfile(file,output/"clock-data.js")
    rows = [("起点 · 0 次",0,"state"),("已更新 12 次",STEP,"state"),("成片 · 24 次",rec.steps,"state")]
    (output/"two-clocks.svg").write_text(film_grid(rec,rows,"横着读影片帧，竖着读生成次数"),encoding="utf-8")
    rows = [("当前输入 x",STEP,"state"),("这次猜的成片 ŷ",STEP,"clean"),
            ("步后送回的 x′",STEP+1,"state"),("最终影片",rec.steps,"state")]
    (output/"one-update.svg").write_text(film_grid(rec,rows,"同一次采样，第 13 次调用前后",mark_pixel=True),encoding="utf-8")
    timing_link = '<p><a href="timing.html">少算几步，影片会变短吗？对照两次实际成片</a></p>' if (output/"timing.html").is_file() else ""
    template = Path(__file__).with_name("clock_ui.html").read_text(encoding="utf-8")
    (output/"two-clocks.html").write_text(template.replace("__TIMING_LINK__",timing_link),encoding="utf-8")
    return rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording",type=Path)
    parser.add_argument("--out",type=Path,help="可选，将双时钟网页、图和记录重建至该目录")
    parser.add_argument("--case",type=int,default=1,help="起点编号从 1 开始")
    parser.add_argument("--step",type=int,default=STEP,help="已经完成的更新次数")
    parser.add_argument("--frame",type=int,default=FRAME)
    parser.add_argument("--row",type=int,default=ROW)
    parser.add_argument("--col",type=int,default=COL)
    args = parser.parse_args()
    recording = render(args.recording,args.out) if args.out else Recording(args.recording)
    try:
        print(json.dumps(recording.pixel(args.case-1,args.step,args.frame,args.row,args.col),indent=2))
    except ValueError as error:
        parser.error(str(error))
