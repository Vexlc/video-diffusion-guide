# 从雪花到运动

一份面向初学者的中文视频生成讲义。通过一段六帧方块影片，动手理解扩散去噪、Flow Matching 的速度场、跨帧注意力，以及 ControlNet 怎样把位置提示送入网络。

[阅读 89 页 PDF](video-diffusion-guide.pdf) · [复现实验说明](reproduce/README.md) · [查看 Jupyter 笔记本](starter-lab/first-lab.ipynb)

## 第一次使用

1. [下载完整离线实验包（ZIP）](https://github.com/Vexlc/video-diffusion-guide/releases/download/v1.0.0/video-diffusion-kit.zip) 并解压，或克隆本仓库，在本机浏览器打开根目录的 `index.html`。
2. 先读讲义第 2–5 页，再选「生成我的第一段影片」。保持噪声相同，把首帧提示从第 3 行移到第 7 行，比较两段影片。
3. 要亲手画图，在仓库根目录运行下面的命令，再打开生成的 `my-first-lab.html`。将步数改为 12，就能比较同一份初始噪声的两条轨迹。

```sh
python3 starter-lab/first_lab.py --steps 2
```

Windows 上可将 `python3` 换成 `py -3`。这个脚本只使用 Python 标准库。离线网页可直接打开，训练神经网络的进阶实验请参照复现说明中的 PyTorch 环境要求。

## 按问题找页码

| 想弄懂的事 | 阅读入口 |
| --- | --- |
| 怎样把雪花当作去噪训练的答案？ | 第 7–9 页 |
| Flow Matching 学什么速度，怎样走下一步？ | 第 10–12、18、89 页 |
| 为什么相邻视频帧要交换信息？ | 第 13–14、48 页 |
| 空间提示怎样改变生成结果？ | 第 15–17、85–87 页 |
| 到哪里找原论文和作者代码？ | 第 20、26 页 |
