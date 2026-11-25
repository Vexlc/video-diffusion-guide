"""从当前讲义标题、页序与实际 Python 函数生成离线页码索引。"""
import argparse
import ast
from html import escape
from html.parser import HTMLParser
from pathlib import Path

from page_routes import ROUTES


BOOK = "video-diffusion-lab.html"
PDF = "video-diffusion-guide.pdf"
TARGETS = {
    "message":("dit-lab/message.html","跟一条跨帧消息"),
    "variance":("prediction-lab/variance.html","四张题卡算出加噪方差"),
    "starter":("starter-lab/index.html","四道可编辑的笔记本练习"),
    "timing":("pixel-lab/timing.html","生成步数与播放速度"),
    "methodPair":("pair-lab/index.html","两台生成器的位置控制对照"),
    "cfg":("depth-lab/guidance.html","文字引导与深度控制对照"),
    "depth":("depth-lab/index.html","预训练深度控制影片"),
    **{k:(f"{BOOK}#lab{k}",f"实验 {k}") for k in "ABCD"},
    **{k:(f"workshop/index.html#{tag}",f"工坊 {k}") for k,tag in zip("EFGH",("fm","spatial","att","reverse"))},
    "reading":("flow-hint/reading.html","跟一段方块学三件事"),
    "walk":(BOOK+"#walkthrough","四道入门练习"),
    "cohort":(BOOK+"#walk-cohort","64 段路标对照"),
    "step":("prediction-lab/reverse-step.html","反向一步实验"),
    "bayes":("prediction-lab/posterior.html","两种来源的去噪实验"),
    "forward":("prediction-lab/forward.html","两次加噪实验"),
    "wire":("pixel-lab/temporal-shift.html","跨帧通道实验"),
    "I":("pixel-lab/index.html","像素放映室"), "clock":("pixel-lab/two-clocks.html","六帧放映台"),
    "J":("control-lab/index.html","条件控制"), "hint":("control-lab/edit-points.html","提示点实验"),
    "K":("flow-lab/index.html","流匹配实验"), "tickets":("flow-lab/training-tickets.html","训练题实验"),
    "L":("latent-lab/index.html","潜空间实验"), "M":("dit-lab/index.html","条件 DiT 实验"),
    "N":("prediction-lab/index.html","公式练习"), "O":("hunyuan-lab/index.html","源码练习"),
    "path":("pixel-lab/path-clock.html","扩散与直线速度换算"),
    "gate":("control-lab/gate-training.html","零出口训练实验"),
    "solver":("flow-hint/first-film.html","同预算求解器实验"),
    "scheduler":("scheduler-lab/index.html","开源调度器对照"),
    "windows":("flow-hint/first-film.html","位置提示时段实验"),
    "windowScores":("flow-hint/window-scores.html","早晚时段逐题成绩"),
    "solverScore":("flow-hint/solver-scores.html","求解器配对成绩"),
    "first":("flow-hint/first-film.html","浏览器生成器"),
    "playback":("flow-hint/playback.html","倒放与重新生成"),
    "motion":("flow-hint/motion.html","生成速度与片中运动实验"),
    "P":("flow-hint/index.html","首帧点与方向联合实验"),
    "reader":(BOOK+"#chapter-20","原论文导读"), "sources":("prediction-lab/index.html","作者库的端点与速度标签"),
}
QUICK = {"I":"pixel", "J":"control", "L":"latent", "M":"dit"}
COMMANDS = {
    "message":"attention_report.py --query 8 --source 13 --mode full",
    "variance":"variance_lab.py --share 0.64 --mode independent",
    "cfg":"cfg_report.py --case w7.5-d1 --step 0 --frame 2 --col 16 --row 12",
    "starter":"lane_lessons.py --start 0.4 --steps 12",
    "timing":"timing_report.py --case 1 --frame 2 --row 5 --col 10 --fps 3",
    "methodPair":"route_pair.py --seed 1700 --row 3",
    "playback":"playback_report.py --seed 9317 --frame 1 --step 12 --fps 6",
    "reading":"reading_route.py --row 3 --step 12",
    "walk":"first_walk.py --seed 17", "cohort":"first_walk.py --cohort --seed 21",
    "path":"path_coordinates.py",
    "gate":"gate_lesson.py ../control-lab --batch 1",
    "solver":"solver_lab.py --sample --budget 4 --out replay/solver-sample",
    "solverScore":"solver_report.py ../flow-hint --budget 4",
    "first":"flow_hint.py --device cpu --replay ../flow-hint/branch.pt --point 7 3 --direction right --strength 1 --noise-seed 9317 --out replay/my-first-film",
    "motion":"motion_readout.py ../flow-hint --step 12 --frame 2 --component 1 --fps 6",
    "P":"flow_hint.py --device cpu --replay ../flow-hint/branch.pt --out replay/flow-hint-cpu",
    "step":"reverse_pixel.py --x 0.04 --target 0.96",
    "bayes":"posterior_lab.py --x 0.04 --neighbor 0.64 --mode shared",
    "wire":"temporal_report.py ../pixel-lab --switches 111 --swap 0",
    "clock":"clock_report.py ../pixel-lab/clock-data.js --step 12 --frame 2 --row 5 --col 7",
    "tickets":"flow_tickets.py","hint":"hint_probe.py --out replay/edit-points","K":"flow_coupling.py",
    "N":"prediction_lab.py","O":"hunyuan_trace.py","reader":"flow_tickets.py","sources":"author_endpoints.py --step 25",
}
ALIASES = {
    "message":"视频骨架 自注意力 attention Transformer Q K V softmax 值向量 跨帧 权重 贡献 潜块",
    "variance":"方差 均值 高斯 正态 标准差 根号 概率 系数 题卡 条件统计 加噪 variance DDPM Flow Matching",
    "cfg":"CFG ControlNet ControlVideo 负面提示 空文字 深度 引导 guidance scale 条件 橙球",
    "starter":"目录 阅读地图 主题 笔记本 notebook Jupyter 速度场 入门 概率 两条目标行",
    "methodPair":"入门 四格 视频扩散 流匹配 flow matching ControlNet 位置控制 DDIM v 欧拉",
    "depth":"深度 depth ControlVideo ControlNet 原版 预训练 球 RGB 跳跃连接",
    "playback":"倒放 正放 重新生成 帧率 生成速度 光流 flow matching 运动 片中速度",
    "reading":"留出练习 半程 原片 速度 实走 flow matching 扩散 ControlNet",
    "walk":"初学 入门 自测 扩散 flow matching 视频 ControlNet",
    "cohort":"入门 配对 误差 MAE 路标 错位 ControlNet 种子 评估",
    "path":"v_prediction v 参数化 流匹配 flow matching DDIM 欧拉 速度 坐标 时间表 直线 圆弧",
    "gate":"ControlNet 零卷积 零初始化 Adam 残差 梯度 反向传播 共享权重 卷积",
    "solver":"采样器 欧拉 Euler 显式中点 midpoint Flow Matching NFE 步数 查询",
    "forward":"DDPM noise schedule beta alpha_bar alphas_cumprod add_noise 合并噪声 正向加噪 边缘分布 联合分布 相关系数 方差",
    "windows":"ControlNet control_guidance_start control_guidance_end 时段 早开 晚开 控制窗口",
    "windowScores":"ControlNet 半程 提示 时段 配对 命中 早开 晚开",
    "scheduler":"Diffusers DDIMScheduler FlowMatchEulerDiscreteScheduler flow matching 裁剪 符号 调度器",
    "solverScore":"积分误差 步数 欧拉 Euler 显式中点法 midpoint RMSE 查询预算",
    "first":"入门 第一次生成 浏览器 视频 流匹配 flow matching ControlNet",
    "motion":"光流 optical flow 帧率 生成速度 flow matching 位移 视频帧 RAFT",
    "P":"综合实验 ControlNet flow matching 视频扩散 联合控制",
    "A":"加噪 噪声预测 diffusion 扩散", "B":"流匹配 flow matching 欧拉", "C":"视频扩散 DDIM 时序",
    "D":"ControlNet 控制 条件支路", "E":"flow matching 流匹配", "F":"ControlNet 遮罩",
    "G":"注意力 attention", "H":"DDIM DDPM 随机 确定性", "I":"video diffusion 视频扩散 时序",
    "step":"DDPM DDIM 扩散 反向采样 条件分布 方差 均值",
    "bayes":"score 得分函数 后验 条件平均 贝叶斯 扩散 去噪 噪声估计",
    "wire":"跨帧通道 时间移位 TSM 感受野 视频扩散",
    "timing":"生成步 影片帧 采样步数 帧率 帧数 FPS 时钟 片长 动画",
    "clock":"DDIM 视频扩散 采样器", "J":"ControlNet 条件控制", "hint":"ControlNet 控制",
    "K":"flow matching 流匹配", "tickets":"flow matching 流匹配", "L":"VAE 潜空间",
    "M":"DiT Transformer", "N":"参数化 ε 噪声预测", "O":"HunyuanVideo 混元",
}


# 页码由讲义节点解析，符号入口与对应页共用可搜索的别名。
SYMBOLS = (
    ("ε、ε̂", "epsilon eps noise 噪声 真值 预测 合并", "没有帽子的 ε 是这道题抽到的噪声。ε̂ 是网络看过带噪输入后给出的猜测。", ("chapter-8","noiseMerge")),
    ("β、A、σ", "beta alpha_bar alphas_cumprod sigma 方差 幅度 累积", "βᵢ 记一次新加噪声的方差。Aₖ 记截至该层的信号幅度平方，σₖ = √(1−Aₖ) 是累计噪声幅度。", ("chapter-7","noiseMerge")),
    ("f、k、τ", "frame timestep tau 影片帧 生成进度 调度器", "f 数播放的帧，k 数扩散噪声级，τ 数噪声到数据的生成进度。读 σ 时还要看它所在的路径。", ("chapter-6","libraryFlow")),
    ("v、u", "v_prediction velocity flow matching 扩散速度 流速度", "圆弧扩散中的 v = aε−σy，直线流匹配的训练标签 u = y−ε。两者各自对应一种路径与时钟。", ("predictionPaths","pathPair")),
    ("z", "latent noise 潜码 噪声起点", "直线训练题的 z 是随机起点。VAE 章节的 z 是编码器给一段视频取的潜码；读它从哪一步产生。", ("chapter-10","latentBook")),
    ("w、λ", "cfg guidance scale lambda controlnet 控制强度", "本书 CFG 的 w 放大两次预测的差。ControlNet 的 λ 调整加进基座的特征修正。", ("chapter-15","chapter-16")),
)


class Pages(HTMLParser):
    """只读取印刷页的第一个标题，忽略网页实验与图中的文字。"""
    def __init__(self):
        super().__init__()
        self.pages = []
        self.section_depth = 0
        self.heading = None

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == "section":
            if self.section_depth:
                self.section_depth += 1
            elif "sheet" in attr.get("class","").split():
                self.section_depth = 1
                self.pages.append({"part":attr["data-part"],"id":attr.get("id",f"chapter-{len(self.pages)+1}"),
                                   "title":""})
        if self.section_depth and self.heading is None and tag in ("h1","h2") and not self.pages[-1]["title"]:
            self.heading = tag

    def handle_data(self, value):
        if self.heading:
            self.pages[-1]["title"] += value

    def handle_endtag(self, tag):
        if tag == self.heading:
            self.heading = None
        if tag == "section" and self.section_depth:
            self.section_depth -= 1


def module_source(file):
    body=file.read_text(encoding="utf-8")
    return body.splitlines(), ast.parse(body)


def source_excerpt(directory, entry):
    module, *names = entry.split(".")
    file = directory/(module+".py")
    lines, node = module_source(file)
    for name in names:
        node = next((item for item in node.body if isinstance(item,(ast.FunctionDef,ast.ClassDef))
                     and item.name==name),None)
        if node is None:
            raise ValueError(f"缺少源码入口 {entry}")
    start=min([node.lineno]+[d.lineno for d in node.decorator_list])
    return file.name, start, "\n".join(lines[start-1:node.end_lineno])


def terminal(lab, entry):
    if entry=="depth_report.differences":
        return "depth_report.py --frame 3 --pixel 152 102",False
    if entry=="depth_report.readout":
        return "depth_report.py --frame 2 --feature 16 12",False
    if entry=="control_window.compare":
        return "control_window.py --sample --width 12 --out replay/my-window",True
    if entry=="window_report.judge":
        return "window_report.py --case 7",False
    if entry=="forward_noise.two_steps":
        return "forward_noise.py --noise -1 1",False
    if entry=="forward_noise.scatter":
        return "forward_noise.py --group --at 0.04",False
    if entry=="scheduler_report.recompute":
        return "scheduler_report.py --kind ddim --step 13 --index 522",False
    if entry=="scheduler_bridge.flow_run":
        return "scheduler_bridge.py --out replay/schedulers.json",True
    if entry=="path_coordinates.line_coordinates":
        return "path_coordinates.py --pair 0.8 -1 --signal 0.8",False
    if entry=="path_rollout.line_sample":
        return "path_rollout.py --out replay/path-clock",True
    if entry=="gate_training.fixed_response":
        return "gate_training.py --out replay/gates",True
    if entry=="gate_lesson.return_sum":
        return "gate_lesson.py ../control-lab --batch 2",False
    if entry=="posterior_lab.estimate":
        return "posterior_lab.py --mode hidden",False
    if entry.startswith("first_steps."):
        return "first_steps.py", False
    if entry=="motion_readout.match_window":
        return "motion_readout.py ../flow-hint --frame 2 --source decoded --window square",False
    if entry=="flow_hint_report.cell":
        return "flow_hint_report.py ../flow-hint --step 12 --case 1 --token 8",False
    if lab in QUICK:
        key=QUICK[lab]
        weight="branch" if lab=="J" else "weights"
        return f"{key}_lab.py --device cpu --replay ../{key}-lab/{weight}.pt --out replay/{key}-cpu",True
    if entry=="flow_tickets.gate":
        return "flow_tickets.py --x 0",False
    if lab in {"A","B","C","D"}:
        return "reproduce.py",False
    if lab in {"E","F","G","H"}:
        return "workshop.py",False
    return COMMANDS[lab],lab in {"hint","P","first","solver"}


def cards(root):
    parser=Pages()
    parser.feed((root/BOOK).read_text(encoding="utf-8"))
    if set(p["part"] for p in parser.pages)!=set(ROUTES) or len(parser.pages)!=len(ROUTES):
        raise ValueError("讲义的问题与索引未一一对应，请检查章节标记")
    for number,p in enumerate(parser.pages,1):
        route,entry,formula,action=ROUTES[p["part"]]
        lab,_,anchor=route.partition("#")
        target,caption=TARGETS[lab]
        if anchor:
            target=target.partition("#")[0]+"#"+anchor
        filename,first,line_text=source_excerpt(root/"reproduce",entry)
        command,torch=terminal(lab,entry)
        terms=[symbols+" "+aliases for symbols,aliases,_,ids in SYMBOLS if p["id"] in ids]
        yield {**p,"number":number,"target":target,"caption":caption,"formula":formula,"action":action,
               "file":filename,"first":first,"source":line_text,"entry":entry,"torch":torch,
               "command":"python3 "+command,"aliases":" ".join([ALIASES.get(lab,""),*terms])}


def card_html(card):
    c={key:escape(str(value),quote=True) for key,value in card.items()}
    n=c["number"]
    env="复跑网络需 PyTorch，CPU 可用。" if card["torch"] else "此命令只用 Python 标准库。"
    if card["file"]=="scheduler_bridge.py":
        env="复跑使用 PyTorch 和 diffusers==0.35.1，CPU 可用。安装命令见实验页。"
    return f'''<details class="card" id="p{n}" data-search="{c["aliases"]} {c["entry"]}">
<summary><span class="page">{int(n):02}</span> <span>{c["title"]}</span></summary>
<div class="body"><p class="part">{c["part"]}</p><p class="formula">{c["formula"]}</p>
<p class="action">{c["action"]}</p><nav aria-label="第 {n} 页的阅读与实验入口"><a class="primary" href="{c["target"]}">打开{c["caption"]}</a>
<a href="{BOOK}#{c["id"]}">读图文版</a> <a href="{PDF}#page={n}">翻到 PDF 第 {n} 页</a></nav>
<details class="code"><summary>看这一步的代码与运行命令</summary>
<p>在解压包的 <code>reproduce</code> 目录运行。{env} Windows 可将 <code>python3</code> 换成 <code>py -3</code>。</p>
<pre><code>{c["command"]}</code></pre><p>实际源码 <a href="reproduce/{c["file"]}">{c["file"]}</a>，从第 {c["first"]} 行开始。</p>
<pre tabindex="0"><code>{c["source"]}</code></pre></details></div></details>'''


def render(root, destination):
    entries=list(cards(root))
    template=Path(__file__).with_name("page_index_ui.html").read_text(encoding="utf-8")
    page=template.replace("__CARDS__","\n".join(card_html(c) for c in entries)).replace("__COUNT__",str(len(entries)))
    numbers={c["id"]:c["number"] for c in entries}
    glossary=[]
    for symbols,_,definition,ids in SYMBOLS:
        links=" · ".join(f'<a href="#p{numbers[id_]}">第 {numbers[id_]} 页</a>' for id_ in ids)
        glossary.append(f"<dt>{escape(symbols)}</dt><dd>{escape(definition)} {links}</dd>")
    destination.write_text(page.replace("__SYMBOLS__","".join(glossary)),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument("--page",type=int,help="在终端查指定页的实验、公式和源码位置")
    parser.add_argument("--out",type=Path,help="重建 HTML；与其他讲义文件同目录时可直接跳转")
    args=parser.parse_args()
    if args.page is not None:
        record=next((c for c in cards(root) if c["number"]==args.page),None)
        if record is None:
            parser.error("页码超出当前讲义范围")
        for key in ("title","formula","action","target","command"):
            print(record[key])
        print(f'{record["file"]}:{record["first"]}  {record["entry"]}')
    if args.out or args.page is None:
        render(root,args.out or root/"page-index.html")
