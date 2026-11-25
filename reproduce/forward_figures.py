"""从同一组噪声画逐格对照、成对散点和条件直方图。"""
import json
from pathlib import Path

from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, circle, grayscale_uri, line, rect, svg, text
from forward_noise import COUNT, HALF_WINDOW, MODES, NAMES, PIXEL, bins, image_inputs, noise_pairs, pictures, scatter, two_steps


def frame_strip(seed, keep, e1, e2):
    planes=pictures(seed,keep,e1,e2)
    parts=[text(12,25,"同一辆车，沿橙框跟踪车身的一格",17,TEAL,bold=True)]
    keys=("clean","first","chain","direct","reuse")
    names=("最初的车","第一层","第二层 · 逐步","第二层 · 合并","第二层 · 沿用 ε₁")
    for i,(key,caption) in enumerate(zip(keys,names)):
        x=10+130*i
        parts += [text(x,54,caption,14,INK,bold=True),
                  f'<image x="{x}" y="68" width="120" height="75" style="image-rendering:pixelated" href="{grayscale_uri(planes[key])}"/>',
                  f'<rect x="{x+57.5}" y="103" width="8" height="8" fill="none" stroke="#ff6a35" stroke-width="1.8"/>',
                  text(x+60,165,f'{planes[key][PIXEL]:+.4f}',15,TEAL if i in (2,3) else INK,"middle",True)]
    trial=two_steps(.8,e1,e2,keep)
    parts.extend([rect(10,182,640,76,LIGHT,4),
                  text(21,204,f'第一次的雪花 ε₁ = {e1:+.3f}，走到第二层时还留下 {keep*.6*e1:+.4f}',15),
                  text(21,227,f'第二次的雪花 ε₂ = {e2:+.3f}，这一步新加 {trial["new"]*e2:+.4f}',15),
                  text(21,250,f'两项相加再除以总噪声幅度 {trial["sigma"]:.4f}，得到合并雪花 {trial["effective"]:+.4f}',15,TEAL,bold=True)])
    return svg("两次加噪与用合并雪花一次直达的逐像素比较",267,parts)


def pair_plate(points, rows, at):
    parts=[text(11,24,"每个点都是同一像素先后两次的读数",17,TEAL,bold=True)]
    curves={key:(bins(points[key]),bins(points[key],at=at)) for key in MODES}
    peaks=[max(max(pair[j]) for pair in curves.values()) or 1. for j in (0,1)]
    lo,hi=-1.6,2.8
    for i,key in enumerate(MODES):
        x=13+216*i
        parts.append(text(x+96,55,NAMES[key],15,INK,"middle",True))
        x0,y0,w,h=x+22,69,165,111
        xx=lambda a:x0+w*(a-lo)/(hi-lo)
        yy=lambda b:y0+h*(hi-b)/(hi-lo)
        left=max(lo,at-HALF_WINDOW)
        right=min(hi,at+HALF_WINDOW)
        parts.extend([rect(x0,y0,w,h,"#eef4ef"),line(x0,yy(0),x0+w,yy(0),GRID),
                      line(xx(0),y0,xx(0),y0+h,GRID),text(x0-4,yy(0)+4,"0",11,MUTED,"end")])
        if left<right:
            parts.append(rect(xx(left),y0,xx(right)-xx(left),h,"#f4d4aa"))
        for a,b in points[key][:320]:
            if lo<=a<=hi and lo<=b<=hi:
                parts.append(circle(round(xx(a),2),round(yy(b),2),1.45,RUST if left<=a<right else TEAL))
        parts += [text(x0,y0+h+16,"−1.6",11,MUTED),text(x0+w,y0+h+16,"2.8",11,MUTED,"end"),
                  text(x0+w/2,y0+h+17,"第一层 →",12,MUTED,"middle")]
        for j,(hist,color) in enumerate(zip(curves[key],(TEAL,RUST))):
            bottom=241+51*j
            coords=" ".join(f"{x0+(k+.5)*w/28:.1f},{bottom-hh*32/peaks[j]:.1f}" for k,hh in enumerate(hist))
            parts += [line(x0,bottom,x0+w,bottom,GRID),
                      text(x0,bottom-23,"全体" if j==0 else "入窗",11,color),
                      f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>']
        parts.extend([text(x0,309,"−2.3",11,MUTED),text(x0+w,309,"3.3",11,MUTED,"end"),
                      text(x0+w/2,309,"第二层亮度 →",12,MUTED,"middle"),
                      text(x0+w/2,332,f'跨层相关 {rows[key]["corr"]:.3f}',14,INK,"middle",True)])
    parts.extend([text(12,356,"散点横轴读第一层，纵轴读第二层。橙色竖带圈出第一层相近的样本。",14),
                  text(12,378,"青线和棕线分别画全体与入窗样本的第二层分布。横轴相同，每行内三栏同一纵标。",14)])
    return svg("三种配方的联合散点与第二层分布",390,parts)


def render(out, *, seed=17, keep=.75, e1=-1., e2=1., at=.04, course=False):
    out.mkdir(parents=True,exist_ok=True)
    points,rows=scatter(seed,keep,at)
    (out/"forward-steps.svg").write_text(frame_strip(seed,keep,e1,e2),encoding="utf-8")
    (out/"forward-pairs.svg").write_text(pair_plate(points,rows,at),encoding="utf-8")
    package={"seed":seed,"keep":keep,"e1":e1,"e2":e2,"at":at,"image":image_inputs(seed),
             "pairs":noise_pairs(seed,COUNT),"half":HALF_WINDOW}
    template=Path(__file__).with_name("forward_ui.html").read_text(encoding="utf-8")
    nav='<a href="../index.html">返回离线首页</a>' if course else ''
    (out/"forward.html").write_text(template.replace("__NAV__",nav).replace("__DATA__",json.dumps(package,separators=(",",":"))),encoding="utf-8")
