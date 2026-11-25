"""从同一套 Python 对照实验画第 19 和 78 页。"""
from draw import GRID, INK, LIGHT, MUTED, RUST, TEAL, circle, line, rect, svg, text
from first_walk import cohort, lesson
from reproduce import integrate, run_noise
from pixel_report import bitmap


def worksheet():
    parts=[]
    for i,(title,hint) in enumerate((("A　多减一点雪花","车身第 15 行、第 24 列"),("B　把一步分成 32 步","数横坐标绝对值大于 1 的点"),("C　让八帧共用一条路线","数小车何时改变左右方向"),("D　把路标推迟一帧","量预测位置离真实位置有多远"))):
        x,y=8+327*(i%2),7+151*(i//2)
        parts += [rect(x,y,317,143,LIGHT,5),text(x+10,y+23,title,16,TEAL,bold=True),
                  text(x+10,y+46,hint,14,MUTED)]
    for column,bias in enumerate((0,.15)):
        _,rows=run_noise(17,bias)
        pixels=rows[-1][2]
        grey=[round(127.5*(min(1,max(-1,v))+1)) for v in pixels]
        parts.append(bitmap([grey[i:i+48] for i in range(0,len(grey),48)],22+154*column,61,83))
        parts.append(f'<rect x="{22+154*column+24*83/48}" y="{61+15*83/48}" width="2.2" height="2.2" fill="none" stroke="#efa322" stroke-width=".9"/>')
        parts.append(text(22+154*column,137,f'亮度 {pixels[744]:.3f}',14,INK))
    for column,n in enumerate((1,32)):
        cx=410+144*column
        cloud=integrate(17,1,n)
        parts.extend((line(cx-52,88,cx+52,88,GRID),line(cx,61,cx,115,GRID)))
        for threshold in (-1,1):
            parts.append(line(cx+14*threshold,61,cx+14*threshold,115,RUST,.8,"3 3"))
        parts.extend(circle(cx+14*x,88-14*y,1.3,TEAL) for x,y in cloud)
        count=sum(abs(x)>1 for x,_ in cloud)
        parts.append(text(cx,137,f'{n} 步：{count} 个',14,INK,"middle"))
    c=lesson("C",17)
    for k,(name,track) in enumerate((("逐帧",c["single"]),("联合",c["joint"]))):
        left=22+155*k
        points=[(left+i*17,244-v*14) for i,v in enumerate(track)]
        parts.append(line(left,244,left+119,244,GRID))
        for a,b in zip(points,points[1:]):
            parts.append(line(*a,*b,RUST if not k else TEAL,2))
        parts.append(text(left,287,f'{name}：{c["before" if not k else "after"]} 次折返',14,INK))
    d=lesson("D",17)
    for i,(name,v) in enumerate((("准时",d["before"]),("迟到",d["after"]))):
        y=222+29*i
        parts += [text(348,y+11,name,14,INK),rect(394,y,290*v,12,TEAL if not i else RUST),
                  text(402+290*v,y+11,f'{v:.3f}',14,INK)]
    parts.append(text(348,287,"两条都与同一段真值比较",13,MUTED))
    return svg("四道单因素对照，种子17的读数和形状",310,parts)


def seed_figure(trials, summary):
    parts=[text(12,25,"每格一段八帧片，按种子从左到右排列",17,TEAL,bold=True)]
    for index,r in enumerate(trials):
        left,top=53+(index%16)*34,40+(index//16)*29
        gap=r["delayed_mae"]-r["aligned_mae"]
        fill=RUST if gap>0 else TEAL if gap<0 else MUTED
        parts += [rect(left,top,31,25,fill,3),text(left+15.5,top+18,str(index),13,"white","middle",True)]
    parts.extend((text(12,176,f'棕色：迟到误差更大 {summary["late_worse"]} 段',14,RUST,bold=True),
                  text(325,176,f'青色：迟到误差更小 {len(summary["late_better_seeds"])} 段',14,TEAL,bold=True),
                  text(12,215,"打开两格，量一帧的误差怎样改变",17,TEAL,bold=True)))
    scale=67
    for i,seed in enumerate((17,21)):
        left=28+327*i
        row=trials[seed]
        parts.extend((text(left,240,f"种子 {seed}",15,INK,bold=True),line(left,295,left+263,295,MUTED)))
        for value in (-.5,.5):
            y=295-value*scale
            parts.extend((line(left,y,left+263,y,GRID,.8,"3 3"),
                          text(left+282,y+4,f"{value:+.1f}",11,MUTED,"middle")))
        for f,diff in enumerate(row["difference"]):
            x=left+8+32*f
            if abs(diff)>1e-12:
                parts.append(rect(x,295-max(0,diff)*scale,16,abs(diff)*scale,RUST if diff>0 else TEAL))
            else:
                parts.append(line(x,295,x+16,295,INK,2))
            parts.append(text(x+8,342,str(f),12,MUTED,"middle"))
        parts.append(text(left,371,f'八帧平均：{row["delayed_mae"]-row["aligned_mae"]:+.3f}',15,INK,bold=True))
    parts.append(text(12,403,"横轴为帧号。上柱表示迟到后误差增加，下柱表示减少。两幅图共用同一高度刻度。",14,MUTED))
    return svg("连续64个配对试验的结果与种子17、21的逐帧误差差值",416,parts)


def render(directory):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"walk-sheet.svg").write_text(worksheet(),encoding="utf-8")
    (directory/"walk-seeds.svg").write_text(seed_figure(*cohort()),encoding="utf-8")
