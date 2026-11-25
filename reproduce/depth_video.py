"""用 ControlVideo 作者实现重跑五帧深度控制，并在第一次预测中读取真实的跳跃连接。"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import zlib

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import numpy as np
import torch
from PIL import Image

W,H,F = 256,192,5
PROMPT = "Studio product photograph of a single orange rubber ball on a smooth light blue tabletop, simple background, soft lighting, realistic photograph."
NEGATIVE = "text, watermark, multiple balls, distorted, collage"


def maps(edited):
    """程序绘制相对深度灰度。背景一致，只在 B 的帧 2 将球心右移 48 像素。"""
    yy,xx = np.mgrid[:H,:W]
    floor = np.clip((yy-65)/125,0,1)
    background = 24 + 106*floor
    result = []
    for f in range(F):
        cx = 80+24*f+(48 if edited and f==2 else 0)
        radius = ((xx-cx)**2+(yy-102)**2)/36**2
        ball = 166+75*np.sqrt(np.maximum(0,1-radius))
        gray = np.rint(np.where(radius<=1,ball,background)).astype("uint8")
        result.append(Image.fromarray(gray).convert("RGB"))
    return result


def packed(image):
    return base64.b64encode(zlib.compress(image.tobytes(),9)).decode("ascii")


def array_data(x):
    a = x.detach().to(device="cpu",dtype=torch.float32).contiguous().numpy().astype("<f4",copy=False)
    return {"shape":list(a.shape),"data":base64.b64encode(a.tobytes()).decode("ascii")}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(2**20),b""):
            sha.update(chunk)
    return sha.hexdigest()


def plane(tensor):
    # CFG 批次 1 是正面文字。保存第 0 通道的五张完整特征图。
    return tensor[1,0].detach().clone()


@torch.no_grad()
def take(pipe, frames, scale, seed, steps):
    trace = {}
    def control(module,args,out):
        if "r" in trace:
            return
        residuals,mid = out
        trace["shapes"] = [list(v.shape[1:]) for v in [*residuals,mid]]
        trace["tick"] = int(args[1])
        trace["input"] = hashlib.sha256(args[0].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        trace["r"] = plane(residuals[0])
    def before(module,args,out):
        if "h" not in trace:
            trace["h"] = plane(out)
    def after(module,args):
        if "joined" not in trace:
            # 最后一个上采样块的最后一个 ResNet 读拼接结果，尾部是最早留下的 320 通道。
            trace["joined"] = plane(args[0][:,-320:])
    hooks = [pipe.controlnet.register_forward_hook(control),
             pipe.unet.conv_in.register_forward_hook(before),
             pipe.unet.up_blocks[-1].resnets[-1].register_forward_pre_hook(after)]
    try:
        result = pipe(PROMPT,negative_prompt=NEGATIVE,width=W,height=H,video_length=F,frames=frames,
                      num_inference_steps=steps,guidance_scale=7.5,controlnet_conditioning_scale=scale,
                      smooth_steps=[],eta=0.,generator=torch.Generator("cuda").manual_seed(seed)).videos
    finally:
        for hook in hooks:
            hook.remove()
    if not torch.equal(trace["h"]+trace["r"],trace["joined"]):
        raise ValueError("读取的解码器输入与 FP16 的实际特征加法不一致")
    arr = result[0].permute(1,2,3,0).float().cpu().numpy()
    if not np.isfinite(arr).all():
        raise ValueError("解码读数含有非有限值")
    pixels = (arr.clip(0,1)*255).round().astype("uint8")
    for name in ("h","r","joined"):
        trace[name] = array_data(trace[name])
    return {"film":[packed(frame) for frame in pixels],"trace":trace},pixels


def load_video_unet(cls,folder):
    # 作者版将 SD 的二维下采样/上采样块换成视频模块。公开配置有时也显式写有二维中间块。
    config=json.loads((folder/"config.json").read_text(encoding="utf-8"))
    config["_class_name"]=cls.__name__
    config["down_block_types"]=[key.replace("2D","3D") for key in config["down_block_types"]]
    config["up_block_types"]=[key.replace("2D","3D") for key in config["up_block_types"]]
    config["mid_block_type"]="UNetMidBlock3DCrossAttn"
    model=cls.from_config(config)
    path=folder/"diffusion_pytorch_model.bin"
    if path.is_file():
        parameters=torch.load(path,map_location="cpu")
    else:
        from safetensors.torch import load_file
        path=folder/"diffusion_pytorch_model.safetensors"
        parameters=load_file(str(path),device="cpu")
    model.load_state_dict(parameters,strict=True)
    return model.half(),path


def load_pipeline(repo,weights,seed):
    # 模型结构直接导入用户指定的作者代码检出目录。
    sys.path.insert(0,str(repo.resolve()))
    import diffusers
    import transformers
    from diffusers import AutoencoderKL,DDIMScheduler
    from transformers import CLIPTextModel,CLIPTokenizer
    from models.controlnet import ControlNetModel3D
    from models.unet import UNet3DConditionModel
    from models.pipeline_controlvideo import ControlVideoPipeline
    from models.RIFE.IFNet_HDv3 import IFNet

    torch.set_num_threads(3)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    sd=weights/"stable-diffusion-v1-5"
    opts={"local_files_only":True}
    tok=CLIPTokenizer.from_pretrained(str(sd/"tokenizer"),**opts)
    text=CLIPTextModel.from_pretrained(str(sd/"text_encoder"),**opts).half()
    vae=AutoencoderKL.from_pretrained(str(sd/"vae"),**opts).half()
    unet,unet_file=load_video_unet(UNet3DConditionModel,sd/"unet")
    net=ControlNetModel3D.from_pretrained_2d(str(weights/"sd-controlnet-depth")).half()
    flow=IFNet(ckpt_path=str(weights/"flownet.pkl")).half()
    pipe=ControlVideoPipeline(vae=vae,text_encoder=text,tokenizer=tok,unet=unet,controlnet=net,
                              scheduler=DDIMScheduler.from_pretrained(str(sd/"scheduler"),**opts),interpolater=flow)
    for model in (unet,net,text,vae,flow):
        model.eval()
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    weights_used=["sd-controlnet-depth/diffusion_pytorch_model.bin",str(unet_file.relative_to(weights)),
                  "stable-diffusion-v1-5/vae/diffusion_pytorch_model.safetensors","stable-diffusion-v1-5/text_encoder/model.safetensors"]
    sources=["models/controlnet.py","models/unet.py","models/pipeline_controlvideo.py","models/attention.py","models/unet_blocks.py"]
    provenance={"torch":torch.__version__,"diffusers":diffusers.__version__,"transformers":transformers.__version__,
                "torch_cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
                "modules":{n:digest(repo/n) for n in sources},"weights":{n:digest(weights/n) for n in weights_used}}
    return pipe,provenance


def capture(repo,weights,seed,steps):
    pipe,provenance=load_pipeline(repo,weights,seed)
    record={**provenance,"seed":seed,"steps":steps,"guidance":7.5,"prompt":PROMPT,"negative":NEGATIVE,
            "size":[W,H],"scheduler":"DDIM","eta":0.0,"smooth_steps":[],"shared_frame_noise":True,
            "maps":{},"runs":{},"delta":{}}
    hints={"A":maps(False),"B":maps(True)}
    pixels={}
    for label,frames in hints.items():
        record["maps"][label]=[packed(im.convert("L")) for im in frames]
        for strength in (0,1):
            key=label+str(strength)
            record["runs"][key],pixels[key]=take(pipe,frames,float(strength),seed,steps)
            print("生成完成",key,flush=True)
    for strength in (0,1):
        diff=np.abs(pixels["B"+str(strength)].astype(np.int16)-pixels["A"+str(strength)].astype(np.int16))
        record["delta"][str(strength)]={"mae":diff.mean((1,2,3)).tolist(),"changed":np.any(diff!=0,-1).sum((1,2)).tolist()}
    if len({run["trace"]["input"] for run in record["runs"].values()})!=1:
        raise ValueError("四次首次查询的带噪输入不同")
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True,help="ControlVideo 作者仓库")
    parser.add_argument("--weights",type=Path,required=True,help="包含 stable-diffusion-v1-5、sd-controlnet-depth 和 flownet.pkl 的目录")
    parser.add_argument("--seed",type=int,default=1947)
    parser.add_argument("--steps",type=int,default=20,choices=range(1,101),metavar="1..100")
    parser.add_argument("--out",type=Path,default=Path("replay/depth-video"))
    args=parser.parse_args()
    data=capture(args.repo,args.weights,args.seed,args.steps)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"record.json").write_text(json.dumps(data,separators=(",",":")),encoding="utf-8")
    from depth_report import render
    render(data,args.out)
    print(data["delta"],flush=True)
