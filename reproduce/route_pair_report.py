"""用随包权重生成学习地图、四格图和可离线推理的实验页。"""
import argparse
import json
from pathlib import Path

from draw import INK,MUTED,RUST,TEAL,circle,line,rect,svg,text
from route_pair import COUNT,FIRST_SEED,ROOT,TASKS,cells,cohort,four,load_models,start


def matrix(traces,scores,row):
    parts=[text(12,25,"同一份噪声　·　同一条目标行　·　四次从头生成",17,TEAL,bold=True)]
    for j,header in enumerate(("关闭位置支路　λ = 0","开启位置支路　λ = 1")):
        parts.append(text(232+280*j,54,header,16,INK,"middle",True))
    for i,(task,caption) in enumerate(zip(TASKS,("视频扩散","流匹配"))):
        top=65+135*i
        parts += [text(9,top+36,caption,17,TEAL,bold=True),
                  text(9,top+62,"猜 v" if i==0 else "猜速度 u",14,MUTED)]
        for j,trace in enumerate(traces[task]):
            left=98+j*280
            unit=9
            def px(x):return left+unit*(x+.5)
            def py(y):return top+unit*(y+.5)
            parts.extend([rect(left,top,16*unit,12*unit,"#e9efe9",2),
                          line(left,py(row),left+16*unit,py(row),RUST,1,"4 3")])
            track=cells(trace[-1])
            for a,b in zip(track,track[1:]):
                parts.append(line(px(a[0]),py(a[1]),px(b[0]),py(b[1]),TEAL,2))
            for x,y in track:
                parts.append(circle(px(x),py(y),2.5,TEAL))
            x,y=track[0]
            parts.append(f'<circle cx="{px(x)}" cy="{py(y)}" r="4.7" fill="#fff" stroke="{TEAL}" stroke-width="2"/>')
            x,y=track[-1];parts.append(rect(px(x)-3.8,py(y)-3.8,7.6,7.6,TEAL))
            labelx=left+154
            parts.extend((text(labelx,top+29,"首帧 → 末帧",13,MUTED),
                          text(labelx,top+51,f"行 {track[0][1]} → {y}",15,INK,bold=True),
                          text(labelx,top+80,"整段命中",13,MUTED),
                          text(labelx,top+99,f"{scores[task][j]} / {COUNT*2}",14,TEAL,bold=True)))
    parts.append(text(12,336,"空心圆为首帧，实心方块为末帧。六个位置叠在同一画幅；橙色虚线是指定的行。",14,MUTED))
    return svg("两种训练任务分别关闭和开启位置支路，同一噪声生成的四段轨迹",349,parts)


def render(weights,destination,*,course=False):
    raw=json.loads(weights.read_text(encoding="utf-8"))
    models=load_models(weights)
    scores=cohort(models)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"objective-control.svg").write_text(matrix(four(models,FIRST_SEED,3),scores,3),encoding="utf-8")
    samples=[start(seed) for seed in range(FIRST_SEED,FIRST_SEED+COUNT)]
    payload={"models":raw["models"],"starts":samples,"firstSeed":FIRST_SEED}
    nav=('<nav><a href="../index.html">离线首页</a><a href="../flow-hint/reading.html">跟一段方块学三件事</a>'
         '<a href="../page-index.html#p87">本页公式与源码</a></nav>') if course else ""
    source=Path(__file__).with_name("route_pair_ui.html").read_text(encoding="utf-8")
    (destination/"index.html").write_text(source.replace("__NAV__",nav).replace("__DATA__",
                                         json.dumps(payload,separators=(",",":"))),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",type=Path,default=ROOT/"pair-lab/weights.json")
    p.add_argument("--out",type=Path,default=Path("replay/route-pair"))
    args=p.parse_args()
    render(args.weights,args.out)
