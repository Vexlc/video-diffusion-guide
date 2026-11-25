"""导出浏览器可运行的视频生成器。导出需要 PyTorch，使用成品页面只需浏览器。"""
import argparse
import base64
import hashlib
from pathlib import Path

import torch

from dit_lab import patchify
from first_film_report import render
from latent_lab import export_decoder


def export(vae,base,branch,saved,checkpoint,out):
    from flow_hint import EVAL_SEED, initial_states
    def arrays(net):
        return {name:base64.b64encode(t.detach().cpu().contiguous().numpy().astype("<f4",copy=False)
                                      .tobytes()).decode("ascii") for name,t in net.state_dict().items()}
    starts=patchify(initial_states(3,torch.device("cpu"))).flatten(1).tolist()
    spec={"base":arrays(base),"branch":arrays(branch),"decoder":export_decoder(vae),
          "mean":saved["mean"].flatten().tolist(),"spread":saved["spread"].flatten().tolist(),
          "starts":{str(EVAL_SEED+i):values for i,values in enumerate(starts)},
          "hashes":{**saved["base_hashes"],"branch":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
    with_guide=(out.resolve().name=="flow-hint" and (out.resolve().parent/"video-diffusion-lab.html").is_file())
    render(spec,out,with_guide=with_guide)


if __name__=="__main__":
    from flow_hint import ROOT,load_trained
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--out",type=Path,default=Path("replay/first-film"))
    args=parser.parse_args()
    checkpoint=args.checkpoint or args.root/"flow-hint/branch.pt"
    export(*load_trained(args.root,checkpoint,torch.device("cpu")),checkpoint,args.out)
    print(args.out/"first-film.html")
