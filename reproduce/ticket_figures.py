"""实验 K 延伸的三幅印图，数据与离线网页共用端点和计数口径。"""
import math
from draw import INK, TEAL, RUST, MUTED, LIGHT, GRID, text, line, rect, circle, svg


def arrow(x,y,end_x,end_y,color,width=2):
    angle=math.atan2(end_y-y,end_x-x)
    parts=[line(x,y,end_x,end_y,color,width)]
    for offset in (-.52,.52):
        parts.append(line(end_x,end_y,end_x-8*math.cos(angle+offset),end_y-8*math.sin(angle+offset),color,width))
    return parts


def intersect():
    parts=[text(20,24,"两道题怎样造出来",16,TEAL,bold=True),
           text(356,24,"两条速度都从交点画起",16,TEAL,bold=True),
           line(330,12,330,167,GRID),text(28,53,"z",14,MUTED),text(253,53,"y",14,MUTED)]
    for a,b,color in [(133,66,TEAL),(66,133,RUST)]:
        parts+=arrow(54,a,260,b,color)
        parts.append(circle(54,a,4,color))
    parts += [circle(157,99.5,4.5,INK),text(156,133,"(0, 1)",13,INK,"middle"),
              text(30,154,"(−2, 0)",13,TEAL),text(22,66,"B",14,RUST),
              text(21,132,"A",14,TEAL),text(263,68,"(2, 2)",13,TEAL),
              text(263,136,"(2, 0)",13,RUST),text(54,54,"(−2, 2)",13,RUST)]
    for end_y,color,label in [(54,TEAL,"A (4, 2)"),(146,RUST,"B (4, −2)"),(100,INK,"平均 (4, 0)")]:
        parts+=arrow(381,100,510,end_y,color,2.7)
        parts.append(text(523,end_y+4,label,15,color,bold=True))
    parts += [circle(381,100,4,INK),text(20,181,"左图画位置。右图画单位生成时间的位移，假设两道题等可能。",14,MUTED)]
    return svg("两条直线在半程相交，各自速度与等权平均速度",193,parts)


def ticket_page(rows,stats):
    parts=[text(15,26,"速度标签落在哪一格？",17,TEAL,bold=True),
           text(345,26,"统一报同一速度，损失多大？",17,TEAL,bold=True)]
    if not rows:
        return svg("窗口中没有样本",120,parts+[text(18,75,"这批样本在默认窗口内没有训练题。",15)])
    bins=stats["bins"]
    peak=max(sum(pair) for pair in bins)
    for i,(left,right) in enumerate(bins):
        x=34+i*17
        h1,h2=104*left/peak,104*right/peak
        parts.extend([rect(x,151-h1,14,h1,RUST),rect(x,151-h1-h2,14,h2,TEAL)])
    parts+=[line(33,152,307,152,INK),text(35,172,"−8",13),text(105,172,"−4",13),
            text(171,172,"0",13),text(238,172,"4",13),text(299,172,"8",13),
            text(32,55,f"最高一格 {peak} 道",13,MUTED),text(35,196,"橙：左团　青：右团　每格宽 1",13,MUTED)]
    avg,var=stats["mean"],stats["variance"]
    low,high=min(-4,math.floor(avg-1)),max(5,math.ceil(avg+1))
    limit=math.ceil(max(var+(low-avg)**2,var+(high-avg)**2)/5)*5
    xx=lambda v: 368+250*(v-low)/(high-low)
    yy=lambda v: 152-112*v/limit
    poly=" ".join(f"{xx(low+i*(high-low)/60):.2f},{yy(var+(low+i*(high-low)/60-avg)**2):.2f}" for i in range(61))
    parts+=[line(368,40,368,152,GRID),line(368,152,618,152,INK),
            text(346,49,limit,12,MUTED),text(349,156,"0",12,MUTED),
            f'<polyline points="{poly}" fill="none" stroke="{TEAL}" stroke-width="3"/>']
    for tick in (low,0,2,high):
        parts.append(text(xx(tick),172,tick,13,MUTED,"middle"))
    for guess,color in [(0,RUST),(avg,TEAL)]:
        parts.append(circle(xx(guess),yy(var+(guess-avg)**2),4.8,color))
    parts.append(text(357,196,f"橙点猜 0，青点猜样本均值 {avg:+.3f}",13,MUTED))
    return svg("同一批样本的速度直方图，以及常数预测的均方损失曲线",213,parts)


def gate_page(info,total):
    parts=[text(14,25,"把门固定在 x = 0.25，两次清点同一批人",17,TEAL,bold=True)]
    width=413
    for row,(t,count) in enumerate([(info["early"],info["before"]),(info["late"],info["after"])]):
        top=50+row*46
        parts.extend([text(16,top+18,f"t = {t:.3f}",15,INK,bold=True),rect(111,top,width,23,"#d5e5e2"),
                      rect(111,top,width*count/total,23,TEAL),text(534,top+17,f"左侧 {count:,}",14,INK)])
    parts+=[text(111,139,"深色占门左边，浅色占门右边；条长表示人数占比。",13,MUTED),
            line(15,154,642,154,GRID),text(16,179,"过门方向",15,TEAL,bold=True)]
    scale=170/max(1,info["left"],info["right"])
    for i,(sign,label,count,color) in enumerate([(1,"向右",info["right"],TEAL),(-1,"向左",info["left"],RUST)]):
        y=204+33*i
        parts += [text(18,y+4,label,15,color),text(81,y+4,f"{count:,} 人",15,INK,bold=True)]
        if count:
            parts+=arrow(369,y,369+sign*count*scale,y,color,3)
        else:
            parts.append(circle(369,y,2,color))
    parts += [line(369,186,369,249,MUTED,1,"3 3"),
              text(17,271,f"净向右 {info['net']:,} 人　=　门左边减少的人数",16,INK,bold=True)]
    return svg("门两侧人数占比与往返穿过人数的比较",286,parts)
