# 从雪花到运动：把图自己算出来

第一次打开解压包里的 `index.html`，点“跟一段方块学三件事”即可看图跟算。若想直接用权重生成影片，点“生成我的第一段影片”。这个页面内嵌权重，在本机浏览器完成 24 步采样和解码。先按默认值生成，再沿用上一段设置移到第 7 行。第二段中心应从 (7,7) 向右走到 (12,7)。可在 PDF 第 2–5 页读入门图解，在第 18 页看同一段影片的训练原理。所有交互图都可离线使用。重算 A–H、K、N、O 用 Python 3.8 或更高版本的标准库。用 Python 训练或运行 I、J、L、M、P 的网络需 PyTorch，CPU 即可，权重随包。

### 第一次用自己的参数画图

在解压包的 `starter-lab` 文件夹打开终端，逐行运行下面两条命令。也可以只把 `first_lab.py` 复制到任意空文件夹，在那里运行。Windows 可把开头的 `python3` 换成 `py -3`，也可以使用本机启动 Python 3 的 `python` 命令。这个脚本只用标准库。

```sh
python3 first_lab.py --out before.html
python3 first_lab.py --noise-error 0 --out after.html
```

双击两份网页，先只看第一张图最右的橙框。亮度从 0.6875 回到 0.8000，恰好多了 0.1125。同一份带噪输入和另三张图可逐项对照。再运行 `python3 first_lab.py --steps 2 --out two.html`，看第二张图的橙线为何只走到 0.3799。对应 PDF 第 88–89 页。用 `--help` 可查看全部参数。

## 第 26 页的作者库端点实验

在离线首页查第 26 页，图里的两次调用都取自作者库的 CPU 输出。固定雪花 −1 和原片 +0.8，交换参数 `x_0`、`x_1`。半程两次都返回题面 −0.1，速度标签分别为 +1.8、−1.8。到四分之一处，两排题面变成 −0.55 和 +0.35。第一条命令只需标准库，会用直线公式核对随包的 202 个样本。

```sh
python3 author_endpoints.py --step 25
```

要直接调用作者函数，可在自己的 Python 虚拟环境安装实验使用的提交（需要网络、Git 和作者库的依赖）。在本目录运行下列命令。最后一行保存你本机的读数，以及实际载入的路径和调度器文件的 SHA-256。随包的 `author_endpoints.json` 也包含这两项校验值。[作者实现](https://github.com/facebookresearch/flow_matching/blob/11568d37f8d5a080e12aa7b5305d9c35ae07d136/flow_matching/path/affine.py#L87-L92)

```sh
python3 -m pip install "flow_matching @ git+https://github.com/facebookresearch/flow_matching.git@11568d37f8d5a080e12aa7b5305d9c35ae07d136"
python3 author_endpoints.py --official --step 25
python3 author_endpoints.py --capture replay/author-endpoints.json
```

## 跨帧消息从哪里来（PDF 第 13、48 页）

双击 `../dit-lab/message.html`。保持查询块 8、源块 13。展开“从 Q 和 K 算出这个百分数”，可以逐个读 16 个乘积，用 0.7351 ÷ 4.0640 得到 18.09%。再选“加权贡献”，观察三组各自送来多少。打开“给当前源块的 V 加 1”。同一行的 18 个注意力权重保持相同，块 8 在第一层第一个头汇总的分量从 −0.2162 变为 −0.0353。再只允许两层读取本组，跨组改动对这里的消息和最终速度均无影响。

第一条命令用 Python 标准库复算全部 18 项加权和。第二条用 CPU 版 PyTorch（与实验 M 相同的环境）重新加载随包的 VAE 和 DiT，取第一段留出原片，用原种子混合成进度 0.5 的训练题，检查潜块与保存记录吻合。程序用钩子取第一层 QKV 的 V 和注意力输出投影真正读入的消息，再逐一给 18 个源块的 V 分量 0 加 1，每次完整重跑网络。两种连接方式各有一份基线和 18 次干预。第三条读取重算后的结果并画出自己的离线图。数字来自这里训练的六帧小模型。[注意力原文](https://arxiv.org/abs/1706.03762) · [Video Diffusion Models](https://arxiv.org/abs/2204.03458) · [DiT 作者代码](https://github.com/facebookresearch/DiT/blob/main/models.py)

```sh
python3 attention_report.py --query 8 --source 13 --mode full
python3 attention_capture.py --out replay/attention/message.json
python3 attention_report.py --record replay/attention/message.json --mode local --out replay/attention
```

## 第 7 页的根号与方差

打开 `../prediction-lab/variance.html`，干净端配比取 64%。独立抽两枚硬币时，四张卡等可能，开根号的一栏给出 −1.4、−0.2、+0.2、+1.4。均值是 0，方差是四个结果平方的平均值 1。改选“固定像素 0.8，只抽雪花”。两个混合结果为 0.04 和 1.24，均值 0.64，方差 0.36。选“同一枚硬币”会让两项相关，可以看到独立性在公式的哪一步起作用。

以下命令只用 Python 3.8 及以上版本的标准库。第一条枚举题卡并计算 4,096 个高斯样本的均值和总体方差。第二条用新种子重做直方图，打开输出目录的 `variance.html` 可以继续改配比。取 50% 且独立抽两端时，直线混合的总体方差是 0.5。课本硬币用来手算随机量的方差；DDPM 正向过程采用高斯噪声。网页的两张直方图使用同一批抽数。[DDPM 式（4）](https://arxiv.org/pdf/2006.11239#page=2) · [Flow Matching 论文](https://arxiv.org/abs/2210.02747)

```sh
python3 variance_lab.py --share 0.64 --mode independent
python3 variance_lab.py --share 0.5 --mode fixed --seed 29 --out replay/variance
```

## 可独立运行的第一份笔记本

PDF 第 88 页的阅读地图把每个主题的看图、手算和代码入口放在同一行。第 89 页印有可复制的 Python 小程序，可从噪声 0.4 分别走 1 步、12 步，观察两行数据怎样分流。

打开 `../starter-lab/index.html` 可离线看四道练习的默认代码和图。用 JupyterLab 或支持笔记本的编辑器打开旁边的 `first-lab.ipynb`，选择 Python 3 内核，执行“Restart Kernel and Run All Cells”。可单独复制这个笔记本到空目录运行。计算与 SVG 绘图用 Python 标准库，无需权重、训练数据或 GPU。若要安装 Jupyter，可参照[官方本地运行说明](https://docs.jupyter.org/en/latest/running.html)。

用普通 Python 也能检查同一组数字。在本目录运行：

```sh
python3 lane_lessons.py --start 0.4 --steps 12
python3 lane_lessons.py --start 0 --steps 2 --noise-error 0 --updates 0
python3 notebook_report.py --out replay/first-notebook
```

第一条的去噪还原亮度为 0.6875；一步流匹配停在 0，多步约为 +1。控制出口经过 8 次更新后权重约为 0.89989。第二条把输入放在对称中线，检查两种停在零的原因。第三条用相同实现重建预览、笔记本和图。四个小实验分别隔离已知噪声的还原、可精确求解的速度、帧间共用信息和可训练标量出口。学习包中的像素视频网络和原版 ControlVideo 实验另有权重或采样记录。

## 用同一个方块比较扩散、流匹配和位置提示

打开 `../pair-lab/index.html`，把第 3 行改成第 7 行。同一份噪声会分别经过两台训练过的模型，每台各生成“支路关闭”和“支路开启”两段影片。上半页可逐帧看轨迹；下半页可沿生成进度检查首帧纵坐标的一次预测。随包已内嵌权重，浏览器可断网运行。PDF 第 87 页采用种子 1700。

在本目录用 Python 3.8 或以上复算。第一行列出四段影片的六帧中心，第二行让 64 份测试噪声各做两道位置题。每一帧都落在指定行才给这段计分。

```bash
python3 route_pair.py --seed 1700 --row 3
python3 route_pair.py --cohort
```

从零训练需要 CPU 版 PyTorch 和 NumPy。随包权重使用 PyTorch 2.1.2。两个基座各训练 3,000 步，再分别冻结并训练位置支路 3,000 步。每批 96 段，种子固定为 617。新页面也可直接双击打开。

```bash
python3 route_pair_train.py --steps 3000 --seed 617 --out replay/route-pair
python3 route_pair_report.py --weights replay/route-pair/weights.json --out replay/route-pair
python3 route_pair.py --weights replay/route-pair/weights.json --cohort
```

四条训练路线来自两个行号与两个方向的组合，六帧中心占 12 个浮点数，画面由中心渲染成方块。测试换新噪声。要观察真实像素生成和深度条件，可打开实验 I 或第 85–86 页的 ControlVideo 实验。

## 文字引导和深度控制各乘在哪里（PDF 第 15 页）

解压后双击 `depth-lab/guidance.html`。先保持“固定深度为 1，提高文字引导”，选帧 2 和采样步 0。读 A、B 两组，参照和正面预测分别相同，采样器收到的值却从 −1.18359375 变到 −1.263671875。“文字差 Δ”以正面减参照，换用放大 25 倍的色阶。黑框这一格为 −0.0146484375，可以看出参照和正面虽然很像，合成结果仍会改变。再选“固定文字引导为 7.5，打开深度”。深度支路在同一格送出的修正由 0 变到 −0.00873566，后面的两份噪声预测也随之改变。这个修正属于隐藏特征，正面与参照属于噪声预测；只有相应层中对齐的量才参与同一次加法。

在 `reproduce` 目录运行下面两条命令。第一条复算书中的一格，第二条选择作者界面的 w = 0，并在指定目录重建动画和图。三个可查采样步为 0、10、19，与五个播放帧分别编号。显示通道的绝对差在 8 位 RGB 上求平均，每个像素只要有一个通道改变就计入变动数。

```sh
python3 cfg_report.py --case w7.5-d1 --step 0 --frame 2 --col 16 --row 12
python3 cfg_report.py --case w0-d1 --step 10 --frame 4 --render replay/cfg
```

浏览器和以上命令读取七次真实采样的记录。若希望更换种子并从噪声重新生成，按下一节准备作者环境、显卡、权重，然后运行下面的采集命令。每次调用在首、中、末三个采样步核对送入 DDIM 的完整张量，保存四通道噪声预测中的通道 0 和第一处跳跃修正的通道 0。复算使用半精度逐次舍入。所有运行固定正面文字、初始噪声和步数，六种配对各只改变一个设置。本页的具体颜色和数字来自种子 1947。[CFG 原论文](https://arxiv.org/abs/2207.12598) · [作者版 CFG 开关](https://github.com/YBYBZhang/ControlVideo/blob/4eac95f722bad014bbd42932795a33212ed417a1/models/pipeline_controlvideo.py#L808-L822) · [作者版 CFG 合成](https://github.com/YBYBZhang/ControlVideo/blob/4eac95f722bad014bbd42932795a33212ed417a1/models/pipeline_controlvideo.py#L895-L925)

```sh
python cfg_video.py --repo ControlVideo --weights ControlVideo/checkpoints --seed 1947 --steps 20 --out replay/guidance
```

## 开源 ControlVideo 的深度条件（PDF 第 16、85–86 页）

在离线首页输入 85，保持默认帧 3 和坐标 (152,102)。条件 A 和 B 在这帧的深度图完全相同。控制强度为 1 时，A 在这里的 RGB 是 (255,118,0)，B 是 (0,96,183)。调到 0 时，两组五帧成片的显示字节完全相同。浏览器的数值来自四次完整的预训练模型采样。热图则来自各次采样的第一次预测，噪声级为 951。实测机器为 GeForce RTX 4070 Ti SUPER，使用 FP16、PyTorch 1.13.1+cu116、Diffusers 0.14.0、Transformers 4.26.1。

在 `reproduce` 目录执行下面两条命令，只需 Python 标准库。第一条从随包的无损 RGB 数组复算一帧的平均绝对差和变动像素数，第二条取真实主干、控制支路和解码端的同一格，另用 FP16 核对加法。加上 `--scale 0` 可复算关闭控制的结果。

```sh
python3 depth_report.py --frame 3 --pixel 152 102
python3 depth_report.py --frame 2 --feature 16 12
```

### 用公开权重重跑五帧生成

在具备 CUDA 的机器上准备作者的代码和配套环境。下面的命令在解压包的 `reproduce` 目录执行，克隆到自己可写的位置。若已准备过作者环境，可直接使用现有的 Python。

```sh
git clone https://github.com/YBYBZhang/ControlVideo.git ControlVideo
git -C ControlVideo checkout 4eac95f722bad014bbd42932795a33212ed417a1
```

按作者说明配置依赖，将 Stable Diffusion 1.5 的 Diffusers 目录放在 `ControlVideo/checkpoints/stable-diffusion-v1-5`，将 ControlNet 1.0 深度模型的 `config.json` 和 `diffusion_pytorch_model.bin` 放在同级的 `sd-controlnet-depth` 目录，RIFE 的 `flownet.pkl` 放在 `checkpoints` 内。SD 目录需含 `unet`、`vae`、`tokenizer`、`text_encoder`、`scheduler`；VAE 和文本编码器使用 safetensors。脚本在内存中将 U-Net 配置适配成作者的视频结构，严格匹配 SD 权重，支持 U-Net 的 bin 或 safetensors 文件。五帧全程关闭 RIFE 插帧。公开大模型权重需另行准备，离线包包含全部对照的成片与数值。[作者安装与权重清单](https://github.com/YBYBZhang/ControlVideo/tree/4eac95f722bad014bbd42932795a33212ed417a1#setup) · [深度 ControlNet 权重](https://huggingface.co/lllyasviel/sd-controlnet-depth) · [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

```sh
python depth_video.py --repo ControlVideo --weights ControlVideo/checkpoints --seed 1947 --steps 20 --out replay/my-depth
python3 depth_report.py --record replay/my-depth/record.json --frame 3 --pixel 152 102
```

打开新目录的 `index.html` 可回放自己的结果。种子和步数可改，GPU 型号、计算库和精度也可能改变生成读数。该脚本自己画相对深度灰度，256 × 192 的五张图以 RGB 三通道形式送给网络。A 的球心按列 80、104、128、152、176 移动，B 仅把帧 2 的列从 128 改到 176，行号始终是 102。四次运行分别是 A 与 B 的控制强度 0 与 1，按原项目将同一份初始空间噪声复制到每一帧，使用相同文字、DDIM、η = 0、文字引导 7.5。记录包含生成模块与关键权重的 SHA-256、张量形状、完整 8 位 RGB、首次预测中截取的通道 0 和初始带噪输入的指纹。

`take` 的三个钩子分别读取支路送出的修正、主干首层的输出和解码器最后一个上采样块中 ResNet 实际读入的跳跃特征。前两项做 FP16 加法后，在记录的全部五张特征图上逐格等于第三项。出口共有 12 份跳跃修正及 1 份中间修正，形状可在网页展开查看。[ControlNet 论文](https://arxiv.org/abs/2302.05543) · [ControlVideo 论文](https://arxiv.org/abs/2305.13077) · [作者支路出口](https://github.com/YBYBZhang/ControlVideo/blob/4eac95f722bad014bbd42932795a33212ed417a1/models/controlnet.py#L549-L562) · [作者的联合采样循环](https://github.com/YBYBZhang/ControlVideo/blob/4eac95f722bad014bbd42932795a33212ed417a1/models/pipeline_controlvideo.py#L895-L925)

## 倒放成片和换条件重新生成（PDF 第 6、68–69 页）

在离线首页输入 68。先沿第一排的列号量屏幕速度，再把第二排倒着追溯到原片。展开“查看生成这格潜码时做的加法”，两排取自同一次模型调用。第三排把标签改为向左，第四排再把首帧点改成 (12,3)。帧率只改变播放时钟。播放格 5 是末帧，界面会停在这里，避免把回放跳回开头的位移算进去。

默认种子 9317，倒放片和第四排成片的中心全程重合。两片各有 6 × 12 × 16 个显示像素，逐位置对齐得到 36 个亮度不同的位置。播放格 1 点“定位本帧最大差异”，可读到 (10,4) 的 21 与 255。中心取自解码器全精度输出的最亮 3 × 3 窗口，亮度对照取量化为 8 位后的显示图。

```sh
python3 playback_report.py --seed 9317 --frame 1 --step 12 --fps 6
python3 playback_lab.py --seed 9317 9318 9319 --out replay/playback
python3 playback_report.py --record replay/playback/playback.json --seed 9318
```

第一、三条只需 Python 标准库。第二条用 PyTorch 从相同噪声独立生成三个条件的成片，另将第一段直接倒放。输出目录中的 `playback.html` 可以直接双击。支持用 `--seed` 传入新的噪声种子，结果依赖这个方块小模型和所给条件。

## 跟一段方块学三件事（封面与第 18 页）

双击首页的“跟一段方块学三件事”。默认选第 3 行、第 12 步，黑色点线标出的练习题是 0.795，绿色实线标出的生成状态是 0.680。点“下一步”，模型走到 0.773。接着点“看成片”，只把首帧点移到第 7 行，两排六帧可直接比较。练习用的干净值由 VAE 编码器给出，生成曲线来自随包模型 24 步的实际记录。两条目标路线都属于留出集，种子统一为 9317。

在本目录运行第一条命令，使用 Python 标准库复算同一格。把步数换为 0 到 24 中的任意整数，或把行换为 7。第二条命令用 PyTorch 从配套权重重新编码原片、计算直线题面上的模型预测，输出文件可与 `../flow-hint/reading-targets.json` 作数值比对。完整自由采样可用下方的实验 P 命令复跑。

```sh
python3 reading_route.py --row 3 --step 12
python3 reading_route.py --capture replay/reading-targets.json
```

## 第一课：四个小实验（PDF 第 2–5、19、78 页）

在离线首页选择“第一课：四个小实验”。从左图读出种子 17 的起始值，预测右图的变化，再查看同一套实验函数算出的图和数字。揭晓后展开公式，可跟算车身的一格亮度、第一个点的小步、前三帧的两次位移或一帧路标的两次距离。想继续操作，可以把当前种子和改动后的参数带进完整实验。原书的四幅读图卡仍可在每题末尾展开。

在本目录运行下面两条命令，只用 Python 标准库。第一条计算全部四题；在末尾加上 `--card A` 可只看第一题。第二条复算连续的 64 段，并列出种子 21 的八帧真值、预测和误差。

```bash
python3 first_walk.py --seed 17
python3 first_walk.py --cohort --seed 21
```

种子 17 中，车身亮度从 0.800 到 −0.095；分 32 小步后有 142/144 个点走到两侧；小车折返从 5 次降到 0 次。路标迟一帧，八帧平均位置误差从 0.285 升到 0.463。打开 64 段的对照，选种子 21：迟到后有五帧误差变小，两帧变大，首帧相同。全体的平均误差从 0.211 升到 0.470。这个对照的支路只训练两个权重。

## 控制点图在哪些生成步生效（PDF 第 16、83–84 页）

离线首页输入 83，沿用种子 9317、点 (7, 3)、向右、强度 1 与窗口宽度 12，选择“生成四排影片”。每排独立运行同一套随包网络 24 次。全程关闭时方向标签仍在。把宽度改为 0，两条半程方案都等于全程关闭；改为 24，两条都等于全程开启。拖动第几次更新可查看块 8 的分量 0。

下面第一条仅使用 Python 标准库，读取随包记录重算同题的早晚成绩。第二条用 PyTorch 在 CPU 上重新生成默认影片，第三条重新评测八份噪声与 28 条留出路线。可以修改第二条的 `--width`（0 到 24）以及起点、方向、种子和强度。若指定的路线将走出画面，生成仍按点图和方向进行，越界的目标无法由画内中心命中。

```sh
python3 window_report.py --case 7
python3 control_window.py --sample --width 12 --point 7 3 --direction right --noise-seed 9317 --strength 1 --out replay/my-window
python3 control_window.py --width 12 --out replay/window-cohort
```

打开对应目录的 `window-films.svg` 看四排成片，`window-scores.html` 可筛选早开独中、晚开独中等题目。将新评测与已有记录比对时，可执行 `python3 window_report.py --record replay/window-cohort/windows.json --case 7`。标准版早开与晚开各有 64 与 158 段六帧全中。两者共同答对 52 段，另有 12 段只在早开时全中。这里的网络是在方块视频上训练的 DiT 位置支路，扩展了 ControlNet 的残差注入思路。Diffusers 的 Stable Diffusion ControlNet 流水线按步判断开始、结束比例，将掩码乘进条件强度；本实验位置提示关闭时直接跳过支路计算。[开源的窗口掩码](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/controlnet/pipeline_controlnet.py#L1234-L1241) · [注入条件强度的调用](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/controlnet/pipeline_controlnet.py#L1271-L1286)

## 一步直达同一层（PDF 第 7、81–82 页）

打开离线首页的“按符号找公式”，或输入 81。逐步图和合并图的车身橙框都是 0.6914。在第一行命令末尾把第二个数改为 -1，两个结果会一起降到约 -0.6314，沿用第一次雪花的结果仍为 -0.3200。把窗口中心移到 1.24，会看到按第一层亮度筛选能区分三种配对方式。

```sh
python3 forward_noise.py --noise -1 1
python3 forward_noise.py --group --seed 17 --at 0.04
python3 forward_noise.py --group --seed 18 --at 1.24 --out replay/forward
```

只用 Python 标准库。打开输出目录的 `forward.html` 可以调整本次种子的实验，两个 SVG 可单独使用。Web 与 Python 使用同一组高斯抽数，图像限定在显示范围内，统计使用原始浮点亮度。第 81 页采用两步教学噪声表，保留幅度分别为 0.8 与 0.75。第 82 页把干净值固定为 0.8，计算跨层相关，种子 17 的 185 对落在 [−0.02, 0.10)。[DDPM 论文式 (2)、(4)](https://arxiv.org/pdf/2006.11239#page=2) · [Diffusers 0.35.1 的 add_noise](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddpm.py#L501-L525)

安装了 PyTorch 和 Diffusers 0.35.1 时，可把合并雪花直接交给该版本的 `DDPMScheduler.add_noise`。在本目录启动 Python 运行下面几行。调度器从 0 数，两次加噪后的层对应索引 1。输出约为 0.691438。

```python
import torch
from diffusers import DDPMScheduler

sampler = DDPMScheduler(num_train_timesteps=2, trained_betas=[0.36, 0.4375])
clean = torch.tensor([0.8])
eps1, eps2 = torch.tensor([-1.0]), torch.tensor([1.0])
merged = (0.45 * eps1 + 0.4375**0.5 * eps2) / 0.8
print(sampler.add_noise(clean, merged, torch.tensor([1])).item())
```

## 用开源调度器校对一次更新（PDF 第 79–80 页）

打开离线首页，输入 79 或 80。调度器实验记录了一整段视频每步的网络输出和库返回值。点一格会用浏览器重新算公式。展开成片能比较各自连续运行 24 步的结果。在本目录用 Python 标准库复算书上的两个位置。

```sh
python3 scheduler_report.py --kind ddim --step 13 --index 522
python3 scheduler_report.py --kind flow --step 13 --index 30
```

要重新问网络并运行官方调度器，在已安装 PyTorch 的环境中执行下面三行。实测环境为 Python 3.10、PyTorch 2.1.2、Diffusers 0.35.1，运算设备为 CPU。所有模型读取包内权重；依赖安装完成后可以断网重跑。生成的交互页是 `replay/scheduler-lab/index.html`。各实验采用 float32，不同环境的末位数值可能稍有差异。

```sh
python3 -m pip install diffusers==0.35.1
python3 scheduler_bridge.py --out replay/schedulers.json
python3 scheduler_report.py --record replay/schedulers.json --out replay/scheduler-lab
```

DDIM 网络在连续时间上训练。实验取 24 个余弦系数构造离散库网格，库的第 23 格、11 格分别对应网络的 1 和 0.5。开启裁剪时保留原 ε̂（`use_clipped_model_output=False`）。流匹配实验给库 24 个递减的非零 σ，库自动补上终点 0。它的 step 接收实际时刻张量，网络收到 1−σ。两种实验都保存了同一步的局部对照，以及模型每步重新计算之后的独立终点对照。[DDIM 版本源码](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddim.py#L411-L448) · [Flow Euler 版本源码](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L432-L467)

## 第一次用 Python 重新生成封面影片

浏览器入口是包内的 `flow-hint/first-film.html`。它支持选择首帧的列与行、方向、三份固定的起始噪声以及支路强度。网页存储权重和三份 72 维高斯噪声，运行时计算神经网络的速度、24 次欧拉更新和解码图像。展开“在 Python 中复现这次设置”可取得与当前成片一致的命令。在已安装 PyTorch 的环境中，从本目录运行：

```sh
python3 flow_hint.py --device cpu --replay ../flow-hint/branch.pt --point 7 3 --direction right --strength 1 --noise-seed 9317 --out replay/my-first-film
```

打开 `replay/my-first-film/sample.svg` 即可看六帧；终端中的中心从 (7,3) 走到 (12,3)。把 `--point 7 3` 改为 `--point 7 7` 可复算第二排。浏览器的同一模式包含浮点数近似，不同引擎的末位数值可能稍有差别。浏览器前向计算在 `first_film.js`，与实验 L 共用的解码器在 `pair_decoder.js`。若重训了支路，可运行 `python3 first_film_export.py --checkpoint replay/flow-hint-training/branch.pt --out replay/browser` 导出新的独立生成页面。

## 对着 PDF 找实验

解压后打开首页 `index.html`，输入页码或术语。也可直接打开 `page-index.html`。例如输入 31，进入六帧放映台，读出指定像素送入下一步的状态；回到结果卡展开“看这一步的代码与运行命令”，能读到本包实际使用的函数及行号。图文版每页页脚也可跳到对应卡片。

在本目录运行下面两行，只需要 Python 标准库。第一行复算第 8、10、16 页的三个入门例子，第二行在终端查第 31 页。

```sh
python3 first_steps.py
python3 page_index.py --page 31
```

第 8 页默认得到 0.575。把噪声猜测改为真实值，再把右上箭头的权重改到 75%，看看干净像素和最优箭头各变成多少。运行 `python3 first_steps.py --noise-guess -1 --upper 0.75` 应得到 0.800 和 (4, 1)。零出口实验固定基座与支路输入，只练一个出口的 W、b。它示范梯度如何开始传回支路；第 34 页另列出实际两尺度视频支路的训练读数。Windows 可将 `python3` 换为 `py -3`。

## 第 8、12 页 · 同一格雪花的两种来历

双击 `prediction-lab/posterior.html`（相对于压缩包顶层）。把当前带噪读数留在 0.04，逐一选择“遮住邻帧”“跟踪同一部位”“两帧独立抽部位”。两帧噪声独立；只有共享部位身份时，邻帧的亮暗才提供本帧身份的线索。页面同时给出两类来源权重、最佳噪声猜测、真实出题记录的近邻平均，以及条件对数密度的局部斜率。

在本目录用标准库重算下面的对照。第一条使固定种子 539 的 65,536 道训练题中有 411 道同时落入两个窗口，其中 336 道来自车身。第二条移除两帧之间的身份关联。加 `--out replay/posterior` 可重画离线网页和两张 PDF 插图。

```sh
python3 posterior_lab.py --x 0.04 --neighbor 0.64 --mode shared
python3 posterior_lab.py --x 0.04 --neighbor 0.64 --mode independent
```

## 第 9、65 页 · 反向一步为什么会抽到不同亮度

打开 `prediction-lab/reverse-step.html`（相对于包顶层）。保留当前读数 0.04 和下一层信号系数 0.96，把新高斯数从 −1 换到 +1，再打开完整分布，点击“直达干净层”。绿色参照使用两种已知原值的条件混合；DDPM 一栏使用各组共同的固定方差，DDIM 一栏取 η = 0。正向出题的 65,536 条记录和按正向加噪积分的有限窗理论是另一条核对路径。

在本目录重算；不需要第三方依赖。JSON 中 `step` 是系数，`bins` 是精确读数处的分组概率，`forward` 是窗口记录与理论，`one_draw` 是当次抽样的三个结果。分组顺序为左尾、16 个 0.15 宽区间、右尾。每个内部区间含左端点，不含右端点。

```sh
python3 reverse_pixel.py --x 0.04 --target 0.96 --fresh 1
python3 reverse_pixel.py --x 0.04 --target 1 --uniform 0.75 --out replay/reverse-step
```

第二条会生成含书中预设与命令所选读数和目标层的独立网页。绘入 PDF 的两张插图始终使用讲义固定例题。

## 第 63–64 页 · 追踪一个特征，或只换一帧

打开 `../pixel-lab/temporal-shift.html`。入口有 16 个通道。固定帧 2、第 5 行、第 7 列，通道 2 读帧 1，通道 6 读帧 3，通道 12 读自己。页面支持八种连接组合，能在浏览器里用随包权重直接计算任一单格改动。整帧对照表则读取独立 PyTorch 前向记录；点中格子后可以查看 A、B 的输入与预测。两种对照都固定一次网络调用。

在 `reproduce` 目录读出一个通道来源和整帧替换后的六个数，仅需 Python 标准库。

```sh
python3 temporal_report.py ../pixel-lab --switches 111 --layer 0 --frame 2 --channel 2 --row 5 --col 7 --swap 0
```

加上 `--render` 可从记录重画两幅 SVG 和整个交互页。用 PyTorch 重新运行同一组权重、重做全部对照，运行下面第一条命令。若有 Node 18 或更新版本，第二条可逐像素核对八种开关和八组单格改动。

```sh
python3 temporal_probe.py --out replay/temporal-shift
node verify_temporal.js replay/temporal-shift/temporal.json
```

## 第一次运行

解压整个包，在含有本说明和 `reproduce.py` 的目录打开终端，执行：

```sh
python3 reproduce.py
```

Windows 可用 `py -3 reproduce.py`。双击新生成的 `replay/seed-17-f32-u80/index.html`，从上到下读四张图。这个图册由 Python 新算出的四个同目录 SVG 组成（请勿只移动 HTML）；数字保存在同目录的 `metrics.json`。只想对答案，可以先打开包中已经算好的 `example-seed17/index.html`。

## 一次只改一件事

| 想追问 | 在本目录运行 | 看哪里 |
| --- | --- | --- |
| B：为什么不能一步到终点？ | `python3 reproduce.py --flow-steps 1` | B 图下排原本不同的两格应完全重合，坐标都为零。 |
| D：没有训练，路旗会生效吗？ | `python3 reproduce.py --updates 0` | D 图里的旗子还在，三辆小车却停在同一处。 |
| C：换一段噪声，轨迹还一样吗？ | `python3 reproduce.py --seed 18` | 对比 C 图折线形状。联合结果仍来自四条可能的匀速路线；两个种子也可能生成同一路线。 |
| A：若准确知道雪花？ | `python3 reproduce.py --bias 0` | A 图下排三辆车都恢复原亮度，误差应为零（忽略浮点舍入）。 |

默认输出目录包含种子、步数和更新次数。相同参数重复运行会覆盖这个目录的同名结果；想保留不同代码版本的输出，用 `--out 新目录`。

## 对照第 19 页的数字

默认设置：种子 17，B 的终点对照为 1 / 32 步，C 为 24 个 DDIM 步，D 为 80 次全批量梯度下降。下列值为三位小数，网页与 Python 用同一随机数方案和同一噪声日程。

| 实验 | 预期读数 | 实现入口（都在 reproduce.py） |
| --- | --- | --- |
| A：噪声步 90 | 指定像素的还原绝对误差 0.895 | `run_noise`：混合，再带偏差地倒推 |
| B：走到终点 | 到 256 步参考的平均距离 2.178 / 0.037 | `velocity` 给当前位置的方向，`integrate` 迈小步 |
| C：八帧小车 | 逐帧 / 联合的平均绝对二阶差分 1.910 / 0.000 | `sample_video`：只切换 `joint` 观察证据范围 |
| D：三种路标 | 无控制 / 对齐 / 错一帧的八帧位置 MAE 0.650 / 0.285 / 0.463 | `train_control` 训练两个参数，`use_control` 只做测试 |

数值自检：`python3 -m unittest -v test_reproduce`。没有软件包需要安装。若修改了实验，回归测试针对的是**原版**设置；预期改变的断言可以用来定位修改影响的环节。

## 哪些是真的计算，哪些是教学简化？

- A 直接知道自己加的噪声，不是假装训练好的图像生成器。图像显示有裁剪，报告的误差用未裁剪的浮点数。
- B 的任务是沿精确速度场做数值积分，不在这个例子中拟合神经网络。一步失败是这个对称分布的反例，不是说任何 FM 都不能一步采样。
- C 真的运行 24 步确定性 DDIM，忽略外观，只生成车的位置。四条匀速轨迹的先验已知，最优去噪器能通过加权平均精确计算。
- D 从零执行梯度下降，不读取训练好的参数；训练与测试使用不同的样本。它演示“冻结基座 + 学残差”，不是原版 ControlNet 权重或完整视频模型。

想只看算法，读 `reproduce.py`；`draw.py` 只负责排图。除最后的文件导出外，计算和画图彼此独立。

## 第二条路线：训练速度、检查像素、重跑采样

不用安装任何额外依赖，在同一目录运行 `python3 workshop.py`。打开打印出的 HTML 路径，或免运行直接打开包内的 `workshop/index.html`。新报告无需服务器，单独一个 HTML 就能玩 E–H；同目录的 SVG 和 `metrics.json` 供做笔记与核对。默认种子为 17。

| 实验 | 读哪里 / 改什么 | 可验证的预测 |
| --- | --- | --- |
| E 从样本学流匹配（21–22 页） | `workshop_models.py`：`training_pair` → `train_tables` → `advance`。运行 `python3 workshop.py --samples 0` | 空表的速度全是零，采样保持起点，但解码后的小车不一定静止。默认训练 32,768 对，留出速度 MSE 约 0.814；参考不是模型输入。 |
| F 遮罩与零初始化（24–25 页） | `train_spatial`、`spatial_workshop`；运行 `python3 workshop.py --updates 0` | 零更新时三种还原相同。默认训练后，把遮罩迟一帧，全图 MAE 约 0.146；只统计 448 个变化像素，约 1.394。 |
| G 跨帧注意力（23 页） | `attention`；在交互页切换“两块都有车的外观” | 车查询从一个位置约 83.4% 变成两个位置各约 47.6%；查看树的查询，缺少匹配仍会分完一票。 |
| H DDPM 对 DDIM（27–28 页） | `reverse_paths.py`：`reverse_step` → `sample` → `compare`；改 `--seed 18` | 固定起始噪声重跑 96 次，确定性 DDIM 只能得到一个结果；每次换起点就不受该限制。DDPM 的反向随机数流单独控制。 |

四组进阶实验同样是教学小模型。E 用表格回归，不是视频大网络；F 真正训练共享的 1×1 线性支路，不是复现完整 ControlNet；G 是人工特征；H 是四路线显式先验的 24 步反向采样。外部真实模型只提供源码阅读，不宣称在本机运行过它们。

完整自检：`python3 -m unittest -v`。使用相同参数会覆盖同名结果；自定义输出用 `--out my-workshop`。

## 第三条路线：从六帧雪花训练真正的像素网络（实验 I）

直接双击包中的 `pixel-lab/index.html`，看 8 个事先固定的样本、生成快照和逐帧播放器，完全离线。若想重算，只需另备 PyTorch；参照 [官方安装选择器](https://pytorch.org/get-started/locally/) 选当前操作系统和 CPU 或 CUDA。无需 torchvision，不下载任何数据集或预训练大模型。参考环境：Python 3.10、PyTorch 2.1.2+cu118。

### 看两只钟，再验算一步（PDF 第 6、9、31 页）

双击 `pixel-lab/two-clocks.html`。固定更新次数播放六帧，再固定影片帧播放生成步。展开接触表可以点选 25 × 6 个格子。每个起点包含 25 份实际状态、24 份网络输出 v̂ 和相应混合系数。黑白图将亮度截到 [−1, 1]，下方数值与曲线保留原浮点值。

在 `reproduce` 目录内用标准库核对第 31 页的橙框，或换一个像素。起点编号从 1 数，帧、行、列从 0 数；step 是已经完成的更新次数。step=24 只剩最终状态。

```bash
python3 clock_report.py ../pixel-lab/clock-data.js --case 1 --step 12 --frame 2 --row 5 --col 7
python3 clock_report.py ../pixel-lab/clock-data.js --out replay/two-clocks
```

第一条应读到当前值 0.617646、干净估计 −0.173202、下一次接到的值 0.559907。第二条在新目录重建网页与两幅讲义插图。每次运行下面的 PyTorch 命令也会从采样器重新录制；记录选取同批 96 个独立起点中的前八个，保留 float32。网络的训练与采样代码在 `pixel_lab.py`，读数和印图在 `clock_report.py`。

第 6 页把 24 步日程的第 6 步、24 步成片、另从同一份噪声生成的 6 步成片放在一起。打开 `../pixel-lab/timing.html`，固定起点 1，帧 2、第 5 行、第 10 列，依次可读到显示灰度 238、49、101。当播放帧率从 3 改为 6，同一卷六帧的片长从 2 秒变为 1 秒。这一页能导出实际动画。也可以在本目录运行：

```sh
python3 timing_report.py --case 1 --frame 2 --row 5 --col 10 --fps 3
python3 timing_report.py --case 1 --steps 6 --fps 6 --animation replay/six-step.png
```

两行只用标准库。动态 PNG 以 192 × 144 显示原始 16 × 12 灰度图，每个像素最近邻放大 12 倍。要从权重重新生成，可用 CPU 版 PyTorch 运行 `python3 timing_capture.py --out replay/timing`。它调用同一采样函数，以原实验的 96 份噪声分别走完两个日程，保留前八组；重建页面为 `replay/timing/timing.html`。采样原理可对照 [DDIM 作者论文](https://arxiv.org/abs/2010.02502)，动画的每帧时长采用 [PNG 规范的分数秒](https://www.w3.org/TR/png-3/#11fcTL)。

在本目录运行，先从随包权重重演，再按需训练：

```sh
python pixel_lab.py --device cpu --replay ../pixel-lab/weights.pt --out replay/weights-cpu
python pixel_lab.py --device cpu --updates 100 --out replay/quick
python pixel_lab.py --updates 4000 --seed 17 --out replay/full
```

最后一条自动选择 CUDA 或 CPU。两个模型使用相同初值和每批相同噪声，每批 24 段，每段 6 帧 × 12 × 16 像素；只有跨帧版在三个位置移动特征通道。单机完整参考使用 CUDA；CPU 也可运行，但从头训练更慢。100 步只用于入门演练，不能拿来直接比对 4,000 步的结果。只加载本包或自己信任的权重。

| 目的 | 源码入口 | 默认输出与核对项 |
| --- | --- | --- |
| 看网络真的学什么 | `pixel_lab.py` 的 `exercise`、`TinyVideo`、`mix_neighbors` | 两台各 21,745 个参数，输入无干净目标；4,000 步留出 v 预测 MSE 约 0.051 / 0.035。 |
| 看生成时有没有偷看路线 | `sample` 从同一批高斯噪声做 24 步 DDIM，没有轨迹参数 | `models.svg` 的三排是同一个起点的真实像素输出。 |
| 分开评画质与运动 | `template_scores` 只做评估 | 96 段整片对最近模板的像素 MSE：逐帧约 0.253，跨帧约 0.123；两台严格车心命中都为 0/96。 |
| 检查时序通信 | `experiment` 中的 `swapped` 只保留原第 2 帧 | 跨帧网络目标帧的 MSE 从约 0.008 升到 0.233；逐帧网络输出完全不变。 |

所有样例均由本包训练，不是对实拍视频或大型模型的测试。不同设备/库版本不保证逐位一致。训练后生成 `weights.pt`、`numbers.json` 和可视化；使用 `--replay` 时不重新训练，也不向新目录复制权重。纯标准库命令 `python pixel_report.py ../pixel-lab/numbers.json` 可以重新排出训练记录对应的图册，但它不会重新采样。

## 给视频网络加上空间条件（实验 J，对应第 34–37 页）

双击 `control-lab/index.html`。固定一个样本对，选择“仅首帧”，按“同一份雪花，反转任务”。然后改为“首尾两帧”再试。拖动帧号可以在同一时刻放大查看原任务、提示和生成结果。网页内含预先选定的四对样片及全部 48 对的成绩。

在 `reproduce` 目录，用与 I 相同的 PyTorch 环境运行：

```sh
python control_lab.py --device cpu --replay ../control-lab/branch.pt --out replay/control-cpu
python control_lab.py --seed 31 --updates 3200 --out replay/control-full
python control_report.py ../control-lab/numbers.json
```

第一行会检查随包基座 `../pixel-lab/weights.pt` 的 SHA-256，再用随包支路对九种设置各跑 24 步。第二行只从头训练条件支路，默认自动选择设备，从第 1 行独立运行。第三行只读保存的像素与评分，用标准库重画图。若移动了基座，请用 `--base 路径` 指定；在不同设备和库版本上可能有数值差异。

| 想查什么 | 在 control_lab.py 中读哪里 | 默认参考 |
| --- | --- | --- |
| 路标怎样加进网络 | `Branch` → `controlled` | 两个 1×1 出口初始化为零，控制前后初始预测相同。 |
| 冻结、训练与反传 | `teach` | 每批 24 段，在三种可见范围中逐段随机挑一种；只更新副本和连接层，3,200 次。 |
| 如何构造成对反事实 | `paired_cases` → `evaluate` | 48 份测试噪声各配两条反向任务，共 96 道。只给首帧时每对输出完全相同；给两端时 47 道六帧全中。 |
| 准确到底是什么意思 | `positions` → `measurements` | 最亮 3×3 窗口的中心作定位。六帧给正确点命中 95 道；反方向点仍对原任务评分，命中 0 道，对相反任务评分则为 95 道。 |

实验 I 的无条件评分允许选最接近的合法模板；实验 J 预先指定唯一目标路线。两组实验的噪声、评分题目也不同，请在各组内部做对照。

### 移动一个提示点，沿着中间特征查到像素（第 59–60 页）

双击 `control-lab/edit-points.html`。A 的六点路线固定；B 的每一帧都可移动或清除提示。任意编辑后，浏览器用训练好的两个控制出口、支路卷积及冻结解码器即时计算。输入是实验 I 起点 1 在更新 12 次后的同一份雪花。与提示无关的三组编码特征已由 PyTorch 算好，保存在 `probe.json`。两组共用相同尺度系数。网页将点图、两次的干净估计、差图及选定通道的加法并排显示。

1. 恢复讲义对照。在帧 2、第 5 行、第 7 列，A/B 的干净估计约为 +0.6243 / −0.4489。帧 1 和 3 的整帧平均绝对变化均约 0.021。
2. 只把粗出口的系数调为 0.5。粗尺度的通道 0、第 2 行、第 3 列现在得到 0.1344 + 0.5 × 0.5031 ≈ 0.3859。通过解码器以后，像素变化通常不再与强度成正比。
3. 清空 B 的所有点。把两出口先开到 1，再一起归零，对照页面显示的 B 与基座最大差。也可以把一个点放到画面边界，查看与直线运动训练题不同的条件如何影响这次预测。

在包内 `reproduce` 目录运行下面的命令。第一行用标准库重画两幅图和离线网页；第二行使用 PyTorch 从随包权重和浮点输入重新提取特征；第三行用 Node.js 18 或更高版本将网页算子与第二行生成的 PyTorch 结果逐元素比较。

```sh
python3 hint_report.py ../control-lab
python hint_probe.py --out replay/edit-points
node verify_hint.js replay/edit-points/probe.json
```

校验覆盖原路线、挪点、两个零出口、全空提示，以及含缺帧和边界点的自定义路线（强度 0.35 / 1.7）。比较两个出口的全部通道、六帧 v̂ 和干净估计，合计 149,760 个数。手算结果使用四舍五入的读数。逐元素复算使用未裁剪的浮点值。

## 谁给噪声指路（实验 K，第 38–41 页）

打开 `flow-lab/index.html`，将生成进度固定为 0.5，把位置从 0 拖到 0.25；右团概率应从 50% 升到约 83.2%。在下一栏跟随第 75 百分位，把老师改为 1 步和 64 步。在末栏保留或打乱老师的训练配对，对照单个起点和全部起点的成绩。

在 `reproduce` 目录运行，无需安装依赖。默认种子 41，每位学生看 16,384 道题。第二条命令同时换种子和训练量；若要测量单一因素，请只改一个参数。

```sh
python3 flow_coupling.py
python3 flow_coupling.py --examples 4096 --seed 42 --out replay/k42
python3 -m unittest -v test_flow_coupling
```

### 手算两支箭头

先想象已知终点来自哪一团。把该团的中心写作 m（−2 或 +2）。起点 Z 是标准高斯，终点 Y 的均值为 m、方差为 ¼。两者独立。于是 X=(1−t)Z+tY 和标签 U=Y−Z 都是高斯变量的线性组合。

1. X 的均值为 tm，方差 V=(1−t)²+t²/4。右团占比 p 可用两团在 x 处的密度相除得到。因两团先验各占一半，p = 1/[1+exp(−4tx/V)]。在 t=½、x=0.25 时，p≈0.8320。
2. 同团内 U 与 X 的协方差是 C=t/4−(1−t)。给定 X=x 后，用高斯条件均值公式得 uₘ=m+(C/V)(x−tm)。在这个路口 C/V=−1.2，代入 m=−2 得 −3.5，代入 m=+2 得 +2.9。
3. v=(1−p)u₋₂+p u₊₂≈1.8249。直接抽样可读 `local_check`：固定 t=½，从 12 万对独立样本中挑处于小窗口内的样本算平均值，与单点公式核对。

### 怎样得到独立的评分参考

记 Φ 为标准高斯的累积概率，Fₜ(x)=[Φ((x+2t)/√V)+Φ((x−2t)/√V)]/2。已知一个分位 q，`quantile` 用二分法求 Fₜ(x)=q，无需调用 `velocity`。对一维光滑流，轨迹上的累积概率不变。也可对 Fₜ(xₜ)=q 求导，得到 dxₜ/dt=−(∂Fₜ/∂t)/(∂Fₜ/∂x)。这里的混合高斯在实轴处处密度为正。

### 训练配对与复核

- `euler` 查询解析速度场得到老师终点。生成学生训练集时只保存老师给出的终点。评分用的理想分位终点从未交给学生。
- `fit_displacements` 将起点分到 96 个区间，在每个非空区间用样本的平均起点和平均位移建节点。`student` 线性插值并把估计位移加回起点，边界外沿最近两节点外推。打乱组只置换终点次序，与保留组使用相同训练样本、区间和容量。
- 默认 64 步老师的全组误差约 0.021865；保留配对学生相对老师约 0.013963、相对参考约 0.035828；打乱组相对参考约 2.024074。这里按原起点的分位逐一对齐，指标对配错人也会计错。不同种子或样本量会改变学生结果。
- 代码拟合的是 t=0 的一步学生。一般 reflow 还会在中间时刻训练新速度场。对照 [Flow Matching 原论文](https://arxiv.org/abs/2210.02747)、[Rectified Flow 原论文](https://arxiv.org/abs/2209.03003) 与 [作者的损失实现](https://github.com/gnobitab/RectifiedFlow/blob/main/ImageGeneration/losses.py) 中的时间选择和目标。

### 从单题的标签到过门人数（第 10、38、61–62 页）

打开 `flow-lab/training-tickets.html`。先保留默认半程和位置 0.25，翻看第 1 张入窗题（原题 #61），再把猜测从 0 换成样本均值。半径 0.025 内共 1,065 道题，损失从约 9.598 降到 6.658。切换“额外告知来自右团”后，只对这 876 道题打分。恢复默认，再到门口观察 t=0.475 到 0.525 的过门人数。点“把门移到正中央”，训练直线分别有 1,388 与 1,361 人换边，精确 ODE 组两个方向都为零。ODE 组从同一批 z 出发，用 q = Φ(z) 和两次的 F(x,t) 比较时段首尾落在哪边；任意时段的首尾换边统计只描述这两次快照。

在包内 `reproduce` 目录运行，均只需 Python 标准库。

```sh
python3 flow_tickets.py
python3 flow_tickets.py --x 0.25 --time 0.5 --radius 0.025 --group 1
python3 flow_tickets.py --total 20000 --half-span 0.1 --out replay/flow-tickets
```

第一条输出样本个数、速度直方图、均值、均方损失的最低值以及门两侧人数。group 为 0 时包含全部分量，−1 和 1 分别表示已知左团、右团。过门统计始终使用同批全部路径。时间范围会裁在 [0,1] 内，以实际时长归一化。最后一条可重建使用指定种子与题数的网页和 SVG；网页起始视图保留与讲义相同的路口和时间。前缀题数共享完全相同的端点，较少样本的窄窗可能没有题，此时局部平均没有定义。

程序在 `flow_coupling.training_draws` 中抽取独立端点；`flow_tickets.py` 和本地网页共用这一批双精度端点。理论时段净流率可由两次分布函数读数算出：令 F 表示门左边的概率，时段平均为 [F(x,t₀)−F(x,t₁)]/(t₁−t₀)。理论瞬时读数 ρv 用混合高斯的密度及条件平均速度独立计算。数值例题的分布及种子见上述代码。概率路径与边缘速度场的原理见 [Lipman 等的 Flow Matching 论文](https://arxiv.org/abs/2210.02747)。

## 从视频 VAE 到潜空间 Flow Matching（实验 L，第 42–46 页）

双击 `latent-lab/index.html`，选择样片 1、中间帧组、通道 0，选第 1 行第 1 列（从 0 数）。把改变量设为 +1.5，再归零。网页在本机用训练过的四层卷积/转置卷积权重重算两张图。色格显示原均值，改变量单独显示。后两栏使用固定的采样结果供对照。

在 `reproduce` 目录运行。与实验 I 使用同一 PyTorch 环境，参考环境为 Python 3.10 和 PyTorch 2.1.2。可分别重演、完整训练或只重画：

```sh
python latent_lab.py --device cpu --replay ../latent-lab/weights.pt --out replay/latent-cpu
python latent_lab.py --seed 53 --vae-steps 2200 --flow-steps 3200 --out replay/latent-full
python latent_report.py ../latent-lab/numbers.json
```

- **数据分开在哪里？** `catalog` 生成 7 个起点参数 × 8 行 × 2 个方向。训练 84 条完整路线、留出 28 条。数据小且规则相同，不同路线可能共享某些两帧局部图样。可视化从留出集中按索引等距取八段；生成示例展示 256 份固定噪声中的前八份。
- **训练 VAE。** `PairVAE.encode` 每两帧编码一组。编码器分开输出 μ、ℓ=log(σ²)，解码器使用 z=μ+exp(ℓ/2)η。原片亮度为 −1/+1，训练重建时映射到 0/1，用逐像素的二元交叉熵。KL 用每个潜格的平均值。网络各有 44,742 个参数，批大小 32、Adam 学习率 0.002。训练两台模型时，用相同初始化、样本序列和后验噪声，β 分别为 0.01 和 0.2。
- **训练生成器。** 选择轻正则 VAE，冻结全部权重，仅编码 84 条训练片，取均值码，按通道计算均值和总体标准差。`train_flow` 对标准化后的 72 维码做线性插值速度回归。全连接网络共 65,544 个参数，批大小 96、Adam 学习率 0.001。它可同时读三个时间组。模型在噪声空间走完后先还原通道单位，再解码。
- **怎样算分？** 重建与各自原片比对；无条件生成另取固定种子的 256 份噪声，与任一合法完整路线相比。位置由最亮 3×3 窗口的中心读出。轻正则均值重建对 28 条原题全中，抽一次为 27 条；24 步无条件 FM 对任一合法模板全中 62/256。显示的灰度量化到 8 位；评分用浮点亮度 −1 到 +1。
- **只改变训练量。** 同时保留同种子和 VAE 步数，将 `--flow-steps` 改成 0，可观察未经训练的随机速度网络。在生成栏中，“直接解码 FM 起点”相当于完全跳过速度网络。两种对照的生成操作不同。
- **对照源码。** [VAE 原论文](https://arxiv.org/abs/1312.6114)解释重参数化；[潜扩散论文](https://arxiv.org/abs/2112.10752)给出先学视觉压缩再学生成器的设计；在 CompVis 的 [AutoencoderKL](https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py) 和[对角高斯类](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/distributions/distributions.py) 中追踪 `sample` 与 `mode` 的不同输入。[HunyuanVideo 报告 §4.1](https://arxiv.org/html/2412.03603v6)可核对真实视频模型的时间分组与通道数。

## 条件视频 Transformer 与 CFG（实验 M，第 47–51 页）

双击 `dit-lab/index.html`。先在“并排播放”里用同一份噪声比较 w=0、1、8。再选块 8、第一层、第一头，换“向左”与“向右”看看组外权重占比。中间的算式允许连续调整 w，使用同一带噪输入上的真实速度；下方每一档影片则独立完成整段采样。

在 `reproduce` 目录运行。与实验 L 共用 PyTorch 环境和轻正则 VAE，脚本会核对基座文件的 SHA-256。第三条从头训练 Transformer，VAE 始终冻结。

```sh
python dit_lab.py --device cpu --replay ../dit-lab/weights.pt --out replay/dit-cpu
python dit_lab.py --replay ../dit-lab/weights.pt --eval-seed 68 --out replay/dit-new-noise
python dit_lab.py --seed 67 --updates 5200 --out replay/dit-full
```

| 要核对的环节 | 源码与操作 | 默认读数 |
| --- | --- | --- |
| 分块和还原 | `patchify` → `unpatchify`。组、行、左右块三项确定编号。块内按通道，再按列排列。 | 输入 (3,2,3,4)，分块后 (18,4)，顺序还原后每个数都回到原格。 |
| 条件与训练 | `TinyDiT` 读取 18 个块、进度、方向类别；`train` 中每样本以 0.2 概率丢弃标签。 | 两层、三头、隐藏宽度 48，69,844 个参数。84 条训练路线，批大小 64。标准化、训练标签与实验 L 相同。 |
| 通信与因果对照 | `probe` 保持中间组不变，只替换两侧。在两层都屏蔽跨组权重时重新跑网络。 | 正常通信平均速度变化 1.496；只留同组为 0。 |
| 同状态合成 | `guided` 使用空预测 + w × (有条件预测 − 空预测)。w=0 和 1 时只计算所需分支，其他系数用同一批次算两支。 | 块 8 分量 0，空条件 1.009，向左 1.123，w=2 得约 1.237。 |
| 整片运动 | `integrate` 默认做 24 步欧拉，`compare` 先看首末位移，再逐帧比对合法模板。 | w=1 时方向 256/256、指定方向严格命中 105/256；w=8 时分别为 255/256、44/256。 |

保持权重，换成评估种子 68 时，w=1、2、8 分别有 86、99、42 段六帧全中（各 256 段）。照上面的第二条命令可复查。比较两批种子时，注意最高分那一档也会变化。

按时间组屏蔽注意力只在推理时进行。演示里的两个方向是类别 ID，没有自然语言编码器。若只想重排可视化，可用标准库运行 `python dit_report.py ../dit-lab/numbers.json`。检查实现可阅读 [DiT 作者代码](https://github.com/facebookresearch/DiT/blob/main/models.py) 的 `DiTBlock.forward`、`forward_with_cfg`，并对照 [CFG 原论文](https://arxiv.org/abs/2207.12598) 的强度约定。实验 M 把引导用于全部潜速度分量。

## 噪声、干净值、扩散 v 与流速度（公式练习 N，第 52–54 页）

双击 `prediction-lab/index.html`。第一题沿用第 7 页的亮像素。把“模型交出”设为 v，保持偏差为 0，只将“采样器按”切换为 ε。手算的还原值从 0.8 变为 1.01。第三格小车显示同一种误读作用于所有像素的结果。点击原车中的任一点或输入行列号，可以把该点代回公式。

在 `reproduce` 目录重画三幅图，生成可单独双击的交互页面：

```sh
python3 prediction_lab.py --seed 17 --out replay/formulas
```

在同一目录进入 Python 还能逐行复算：

```python
from prediction_lab import exercise, read_prediction, trace

x, targets = exercise(clean=0.8, noise=-1.0, signal=0.8)
print(round(x, 3), targets)                         # 0.04；v 标签为 -1.28
for kind, value in targets.items():
    y, eps = read_prediction(x, value, kind, 0.8)
    print(kind, round(y, 3), round(eps, 3))         # 三次都得到 0.8、-1.0
print(trace(clean=0.8, noise=-1.0, time=0.5))      # 每条路径：干净系数、噪声系数、x、dx/dτ
```

换 `--seed` 只重抽小车的噪声。两条路径题固定端点，网页可以切换噪声 −1 和 +1。三种输出的反演在裁剪和阈值处理之前比较；输入处于端点时，不可唯一反演的分量显示“不定”。图上所有演算都直接使用已知数据和噪声。参见 [Salimans 与 Ho 附录 D](https://arxiv.org/pdf/2202.00512#page=15)、[DDIM 调度器中的三种 prediction_type](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py) 和 [Flow Matching 原论文](https://arxiv.org/abs/2210.02747)。

## 固定版本跟读 HunyuanVideo（源码练习 O，第 55–58 页）

打开 `hunyuan-lab/index.html`。在“教学小片”上点块 57，拖到分量 22，检查潜地址 [0,5,2,3,6]。选“发布示例”看 118,800 个视觉块来自哪里。在采样区保持 4 步，切换 shift=1 与 9，比较单个粒子和全批 256 个粒子的终点误差。

在本目录用标准库复现图和数据：

```sh
python3 hunyuan_trace.py --out replay/hunyuan
```

| 想核对的值 | 在本地读哪里 | 在作者代码读哪里 |
| --- | --- | --- |
| 帧数、高宽、块数 | `layout` 向上对齐空间尺寸；帧数检查 4k+1 | [像素尺寸](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/hyvideo/inference.py#L446-L465)、[潜时间和初始噪声](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py#L917-L938)、[块输入](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/hyvideo/modules/embed_layers.py#L39-L63) |
| 一个输出数写到哪里 | `address` 分开块号和块内分量 | [还原潜张量](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/hyvideo/modules/models.py#L664-L676) |
| 强度 6 编成什么 | `guidance_features(6)` 得 256 个数，索引 96 和 224 约为 0.96017、−0.27942 | [三角编码](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/hyvideo/modules/embed_layers.py#L93-L119)、[调制向量](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/hyvideo/modules/models.py#L584-L599)、[CFG 分支](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py#L962-L1010) |
| 查询点和步长 | `knots` 返回 N+1 个边界；`solve` 将实验 K 的速度改写为对 σ 求导 | [时间表](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/hyvideo/diffusion/schedulers/scheduling_flow_match_discrete.py#L142-L155)、[时间偏移](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/hyvideo/diffusion/schedulers/scheduling_flow_match_discrete.py#L185-L186)、[欧拉步](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/hyvideo/diffusion/schedulers/scheduling_flow_match_discrete.py#L235-L252) |

此处选用作者标签 `HunyuanVideo`、提交 `e965abb39f104f618ebd0d9ae27630ee8e55cecc` 的文生视频示例。尺寸、蒸馏引导路径和时间表依据该版本。网页的粒子来自实验 K，复用其解析速度和分位终点。完整权重不参与本练习。原作者的[示例命令](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/e965abb39f104f618ebd0d9ae27630ee8e55cecc/README.md#L231-L244)与[技术报告](https://arxiv.org/html/2412.03603v6)可对照阅读。

**可选的作者源码核验。** 若有该提交的本地源码，且在已安装 PyTorch、Diffusers 的 Python 环境中，可运行以下命令，把最后一项换成源码所在目录。脚本只加载原项目的调度器文件，逐项比较 9 组 σ 序列、模型时间标签及 5 个粒子每一步的新位置；无需下载模型。验证环境为 PyTorch 2.1.2、Diffusers 0.30.2。

```sh
python verify_hunyuan.py /path/to/HunyuanVideo
```

## 实验 P · 一张首帧点图，如何改变生成视频？

阅读 PDF 第 18、66、67 页。打开 `../flow-hint/index.html` 即可离线看片。试着固定种子 9317 和“向右”，把点由 (7,3) 移到 (7,7)，再改为“向左”。网页中每排六张图是同一生成进度的六个视频帧。展开“播放两段影片”可以同步播放开启和关闭位置支路的结果，也可以拖到任意一帧。拖动“已更新次数”可以看生成进度从 0 变到 24。第二节在同一个当前状态重新测量关闭位置支路后的速度。

在本目录运行。第一条只需 Python 标准库。其余命令用已安装 PyTorch 的 Python，随包提供 VAE、基座与支路三份权重。用 Windows 启动器时可把 `python3` 换成 `py -3`。

```sh
python3 flow_hint_report.py ../flow-hint --step 12 --case 1 --token 8
python3 flow_hint.py --device cpu --replay ../flow-hint/branch.pt --point 7 6 --direction right --strength 1 --noise-seed 9317 --out replay/my-clip
python3 flow_hint.py --device cpu --replay ../flow-hint/branch.pt --out replay/flow-hint-cpu
python3 flow_hint.py --device cpu --seed 437 --updates 3600 --out replay/flow-hint-training
```

第二条在 `replay/my-clip/sample.svg` 写入自己指定的影片，终端打印（列，行）中心。第三条重算 28 条留出路线 × 8 份噪声的成绩，生成新的 `numbers.json`、网页与配套 SVG。第四条从实验 M 的固定方向 DiT 初始化副本，重训两个零出口的空间控制支路并评测。有 CUDA 时也可选择 `--device cuda`。训练时固定 VAE 与方向 DiT，84 条训练路线用于计算潜码标准化参数。支路权重保留统计量以及两个基座文件的 SHA-256。不同设备的浮点运算可有细小偏差。

`hint_map` 将首帧点转换为一个 12×16 亮点；`PointBranch` 将其投影并向基座的两个块各送一个残差。相同的点图供三组时间块使用。`train` 以直线路径上的 y−z 监督速度，`solve` 从随机潜码出发走 24 次欧拉更新，解码后 `score` 用最亮 3×3 窗口中心检查六帧。推理时输入只含起点、离散方向与噪声。训练集的所有运动均为水平匀速，可自由输入范围内的网格点试验超出这些路线的情况。

随包的 CPU 成绩依次为 λ=0、0.5、1 时六帧全中 1、199、215 段（每档 224 段）。这三档所有影片都走对方向。网页下方保留首次失误。位置支路借鉴冻结基座和零初始化出口，基座采用教学用视频 DiT。[ControlNet 论文](https://arxiv.org/abs/2302.05543) · [作者代码](https://github.com/lllyasviel/ControlNet) · [Flow Matching 论文](https://arxiv.org/abs/2210.02747) · [DiT 作者代码](https://github.com/facebookresearch/DiT)。

## 实验 P · 两种速度与局部光流匹配（PDF 68–69）

双击 `../flow-hint/motion.html`。保持默认种子、向右、生成步 12、视频帧 2。观察同一像素 (9,3) 如何在影片轴和生成轴上显示不同亮度，再试 12 帧/秒。播放速度翻倍，第 8 号块分量 1 的生成速度仍约 −2.2970。窗口表取相邻成片、枚举 25 个整数位移，保留并列最优。理想影片由实际给定的路线绘制，解码影片取自随包第 24 步生成记录。两者可切换。

```sh
python3 motion_readout.py ../flow-hint --step 12 --frame 2 --component 1 --fps 6
python3 motion_readout.py ../flow-hint --frame 2 --source ideal --window point
python3 motion_readout.py ../flow-hint --frame 2 --source decoded --window square
```

以上命令仅使用标准库。可加 `--direction left`、`--pixel 8 3` 或 `--anchor background`。第三条默认应得到唯一最优位移 (+1,0)，九格亮度绝对差总和 86。完整光流网络可跟读 [RAFT 论文](https://arxiv.org/abs/2003.12039) 与 [作者项目](https://github.com/princeton-vl/RAFT)。

## 实验 P · 真实生成器的采样步数（PDF 第 70–72 页）

打开 `../flow-hint/first-film.html#budget`，选择 4 次查询并生成三排。每排从相同的 72 个初始数出发。沿 24 步网格走 4 次只到 1/6；欧拉走 4 个 1/4 大小的步抵达 1；显式中点走 2 个 1/2 大小的大步，每步查询两次也抵达 1。可选 2、4、8、12、24 次预算。页面会输出与当前起点、噪声、方向和预算相符的 Python 命令。

```sh
python3 solver_report.py ../flow-hint --budget 4
python3 solver_lab.py --sample --point 7 3 --direction right --strength 1 --noise-seed 9317 --budget 4 --out replay/solver-sample
python3 solver_lab.py --out replay/solver
```

第一条用标准库查询随包记录。第二条用 PyTorch 实算三排，生成 `paths.svg` 和带每次查询的 `sample.json`。第三条在 CPU 上重算完整的 224 段配对评测、96/192 步中点参照、图和成绩单。从 `replay/solver/solver-scores.html` 读结果。对比两种积分方法时按网络查询次数配平。四次预算的实际六帧全中数为欧拉 219、中点 212；24 次均为 215。误差针对学到的速度场，路线得分针对给定目标。[Flow Matching 论文](https://arxiv.org/abs/2210.02747) · [torchdiffeq 中的固定步长实现](https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/fixed_grid.py)。

## 重新出讲义（可选）

在解压后的原目录结构中运行 `python3 build_kit.py --render`，脚本重算 A–H、K、N、O 的默认示例，从随包训练记录重画 I、J、L、M、P，把图内嵌进讲义，再用本机 Chrome/Chromium 打印 PDF 并更新离线包。浏览器不在 PATH 时可用 `--chrome /path/to/browser`。只打包现有 PDF 则运行 `python3 build_kit.py`。若改过算法，请用对应的 metrics.json 或 numbers.json 核对正文、检查分页后再分发。

## 实验 J · 零出口的头两次更新（PDF 第 73–74 页）

打开 `control-lab/gate-training.html`。停在更新 1 次，把视线移到“挪点产生的六帧最大绝对差”。接上支路改变了预测，约 0.2174；将同一帧的点从 (6,5) 移到 (10,5)，预测差依旧为零。换到更新 2 次，选择“放大当前阶段”可看见约 0.0000527 的变化。这里比较同一个带噪视频、同一次扩散网络预测。第 35–36 页才检验生成的方块走哪条路线。

在本目录运行前两条命令只需 Python 3 标准库。第三条用 CPU 和 PyTorch，从随包的实验 I 基座重新演算实验 J 的前两批 Adam 更新；第 3,200 步的对照直接读取随包权重。输出目录中的 HTML 可直接打开。

```bash
python3 gate_lesson.py ../control-lab --batch 1
python3 gate_lesson.py ../control-lab --batch 2 --channel 1 --output 2 --sample 3
python3 gate_training.py --out replay/gates
```

第一条默认看粗出口的权重 W[0,0]。网格里橙框的一项是 −0.00000290208，加完一批的 6,912 项约为 −0.00140784。网页也列出轻轻调高、调低这枚权重时的两次损失。比较它们，可独立核对这个导数。再展开“32 项带符号的乘积”，看第二批的梯度怎样沿着已经更新的权重返回同一位置。图中帧、行、列与通道都从 0 数。

要亲手写第 74 页的加法，可在本目录启动 Python，粘贴下面的代码。前四行读取第一批训练的记录；每格存三个通道，切片 `[::3]` 从第零个数开始，每隔三个取一个。

```python
import json
from pathlib import Path
from hint_report import values
batch = json.loads(Path("../control-lab/gate-training.json").read_text())["batches"][0]
h, g = values(batch["h"])[::3], values(batch["g"])[::3]
print(len(h), sum(a * b for a, b in zip(h, g)))
```

会得到 6912 和约 −0.001407839。试着把两处切片都改成 `[1::3]`，再去网页选择输入、输出通道 1，检查新的和。

原版将可训练副本的修正加回冻结网络。可对照论文图 2 与作者代码里的点图编码末层、多个零初始化出口；讲义里的实验采用小型六帧网络，可以在普通 CPU 重演。[ControlNet 论文](https://arxiv.org/abs/2302.05543) · [作者的 ControlNet 实现](https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py#L134-L151)

## 一台扩散网络的两把速度尺（PDF 第 75–77 页）

打开 `pixel-lab/path-clock.html`，保留相邻的 `clock-data.js`。第 75 页的圆点标的是干净与噪声的混合系数。把干净系数设为 0.8、噪声端点设为 −1，先自己算 0.8 份干净、0.6 份雪花的总份数，再对照两种读数。第 76 页从原来的扩散录像选一格重做一步。第 77 页的四排影片由同一组权重从相同雪花独立生成。

在本目录运行。前两条命令只用标准库，第三条需要 CPU 和 PyTorch。权重在包内。

```bash
python3 path_coordinates.py --pair 0.8 -1 --signal 0.8
python3 path_coordinates.py --case 0 --step 12 --frame 2 --row 5 --col 7
python3 path_rollout.py --out replay/path-clock
```

第一条应得到圆弧状态 0.04、v = −1.28、直线状态约 0.0285714。换算的直线速度为 +1.8，等于两端相减。第二条的输入来自六帧小网络的实际录像；下一格按欧拉算约为 0.396763603，按原录像换算为 0.396763577。把步数换成 0 或 23 可以核对首尾的不同步长。

下面三行可粘贴到在本目录启动的 Python 解释器里，直观看到四个量。把雪花端点从 −1 改成 +1 时，同一配比改为怎样的运动？

```python
from path_coordinates import known_pair
p = known_pair(clean=0.8, noise=-1.0, signal=0.8)
print(p["x"], p["v"], p["line"], p["converted"])
```

整段实验从原录像取八份初始雪花。每排独立做 24 次网络前向计算。终点 RMSE 以重新运行的确定性 DDIM 为参照。匹配节点并换算速度时约为 2.39×10⁻⁷，匹配节点但直接采用 v 时约为 2.667，正确换算速度并均分直线时钟时约为 0.01750。这里检验坐标和节点能否重现同一终点，评估运动质量则可用第 32 页的合法模板。不同硬件或 PyTorch 版本的末位可能有差异。

两种表示的等价需要相同端点配对与条件，适用的圆弧系数满足 a²+σ²=1、两数非负。直线路径用 X=τy+(1−τ)ε。每一步都把直线状态 X 还原成 (a+σ)X，把对应的角度时刻交给原模型。模型给出 v 后，从本次估计的两个端点相减得到直线速度。纸上推导及程序使用确定性更新。相关原文与实现为 [Progressive Distillation 附录 D，式 (26)](https://arxiv.org/pdf/2202.00512#page=14)、[Flow Matching](https://arxiv.org/abs/2210.02747) 和 [DDIM 作者项目](https://github.com/ermongroup/ddim)。
