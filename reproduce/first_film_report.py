"""把权重和起始噪声嵌进离线页面。浏览器自己运行生成器，包内只保留一份网页权重。"""
import json
import re
from pathlib import Path


def render(spec,out,*,with_guide=False):
    here=Path(__file__).parent
    template=(here/"first_film_ui.html").read_text(encoding="utf-8")
    for tag,name in (("__DECODER__","pair_decoder.js"),("__NETWORK__","first_film.js")):
        template=template.replace(tag,(here/name).read_text(encoding="utf-8"))
    links={
        "__WINDOW_SCORES__":'<p><a href="window-scores.html#paired">在 224 段留出影片中逐题比较</a></p>',
        "__SCORES_LINK__":'<p><a href="solver-scores.html">用 224 段留出影片比较两种走法</a></p>',
        "__GUIDE_NAV__":'<nav><a href="../index.html">离线首页</a><a href="../video-diffusion-guide.pdf#page=18">PDF 第 18 页的训练原理</a></nav>',
        "__SOURCE_LINK__":'<a href="../page-index.html#p18">打开第 18 页对应源码</a>。',
        "__MORE_LINKS__":'<a href="index.html#step">追踪速度怎样受点图影响</a><a href="motion.html#times">分清生成速度与片中速度</a>',
    }
    for tag,markup in links.items():
        template=template.replace(tag,markup if with_guide else "")
    html=template.replace("__MODEL__",json.dumps(spec,separators=(",",":"),allow_nan=False))
    out.mkdir(parents=True,exist_ok=True)
    (out/"first-film.html").write_text(html,encoding="utf-8")


def read_model(path):
    markup=path.read_text(encoding="utf-8")
    match=re.search(r'<script id="model" type="application/json">(.*?)</script>',markup,re.S)
    if not match:
        raise ValueError("浏览器生成器缺少模型参数")
    return json.loads(match.group(1))


def repack(path,data,out):
    spec=read_model(path)
    expected={**data["base_hashes"],"branch":data["branch_sha256"]}
    if spec["hashes"]!=expected:
        raise ValueError("网页权重与实验 P 不配套。请用 first_film_export.py 从当前权重重新导出。")
    render(spec,out,with_guide=True)
