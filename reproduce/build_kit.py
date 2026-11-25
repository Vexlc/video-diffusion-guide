"""从图解源码重新计算示例、打包；--render 可选用本机 Chrome 重印 PDF。"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import reproduce
import workshop
import walk_figures
import pixel_report
import path_report
import control_report
import gate_lesson
import flow_coupling
import latent_report
import dit_report
import attention_report
import page_index
import prediction_lab
import hunyuan_trace
import flow_hint_report
import first_film_report
import solver_report
import scheduler_report
import forward_figures
import variance_lab
import window_report
import reading_route
import playback_report
import depth_report
import cfg_report
import route_pair_report
import timing_report
import notebook_report


def pack(root, render, chrome):
    with tempfile.TemporaryDirectory() as scratch:
        stage = Path(scratch) / "kit"
        stage.mkdir()
        shutil.copyfile(root/"index.html", stage/"index.html")
        reproduce.run(17, 32, 80, stage / "example-seed17")
        workshop.run(17, 32768, 80, stage / "workshop")
        walk_figures.render(stage/"workshop")
        flow_coupling.run(41, 16384, stage / "flow-lab")
        prediction_lab.run(17, stage / "prediction-lab")
        forward_figures.render(stage / "prediction-lab", course=True)
        variance_lab.render(stage / "prediction-lab", course=True)
        hunyuan_trace.run(stage / "hunyuan-lab")
        for directory, weight, renderer in [("pixel-lab", "weights.pt", pixel_report.render),
                                            ("control-lab", "branch.pt", control_report.render),
                                            ("latent-lab", "weights.pt", latent_report.render),
                                            ("dit-lab", "weights.pt", dit_report.render),
                                            ("flow-hint", "branch.pt", flow_hint_report.render)]:
            target = stage/directory
            target.mkdir()
            for name in ("numbers.json", weight):
                shutil.copyfile(root/directory/name, target/name)
            if directory == "pixel-lab":
                from clock_report import Recording
                record = root/directory/"clock-data.js"
                if Recording(record).weight_sha256 != hashlib.sha256((target/weight).read_bytes()).hexdigest():
                    raise ValueError("双时钟记录与实验 I 权重不匹配")
                shutil.copyfile(record, target/record.name)
                timing_file = root/directory/"timing.json"
                timing = json.loads(timing_file.read_text(encoding="utf-8"))
                for key, item in (("recording", record), ("weights", target/weight)):
                    if timing[key] != hashlib.sha256(item.read_bytes()).hexdigest():
                        raise ValueError("两种采样步数的录像与模型或原始采样记录不匹配")
                shutil.copyfile(timing_file, target/timing_file.name)
                timing_report.render(timing,target,course=True)
                runs_file = root/directory/"path-runs.json"
                runs = json.loads(runs_file.read_text(encoding="utf-8"))
                if runs["hashes"]["weights"] != hashlib.sha256((target/weight).read_bytes()).hexdigest():
                    raise ValueError("两种坐标的对照使用了不同版本的实验 I 权重")
                shutil.copyfile(runs_file,target/runs_file.name)
                path_report.render(runs,target/record.name,target,guide=True)
                capture=root/directory/"temporal.json"
                metadata=json.loads(capture.read_text(encoding="utf-8"))
                for key,path in (("base_sha256",target/weight),("recording_sha256",record)):
                    if metadata[key]!=hashlib.sha256(path.read_bytes()).hexdigest():
                        raise ValueError("跨帧记录与权重或带噪片段不匹配")
                shutil.copyfile(capture,target/capture.name)
            if directory == "dit-lab":
                message = root/directory/"message.json"
                trace = json.loads(message.read_text(encoding="utf-8"))
                for key, path in (("weights_sha256", target/weight), ("numbers_sha256", target/"numbers.json"),
                                  ("vae_sha256", stage/"latent-lab/weights.pt")):
                    if trace[key] != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise ValueError("注意力读取记录与随包模型或输入不匹配")
                shutil.copyfile(message, target/message.name)
                attention_report.render(trace, target, course=True)
            if directory == "control-lab":
                probe = root/directory/"probe.json"
                metadata = json.loads(probe.read_text(encoding="utf-8"))
                for key, path in [("base_sha256", stage/"pixel-lab/weights.pt"), ("branch_sha256", target/weight)]:
                    if metadata[key] != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise ValueError("提示点探针与随包的网络权重不匹配")
                shutil.copyfile(probe,target/probe.name)
                gate_file = root/directory/"gate-training.json"
                gates = json.loads(gate_file.read_text(encoding="utf-8"))
                if gates["hashes"] != {"base": metadata["base_sha256"], "branch": metadata["branch_sha256"],
                                       "probe": hashlib.sha256(probe.read_bytes()).hexdigest()}:
                    raise ValueError("前两步训练记录与提示点探针或网络权重不匹配")
                shutil.copyfile(gate_file, target/gate_file.name)
                gate_lesson.render(gates, target, guide=True)
            data = json.loads((target/"numbers.json").read_text(encoding="utf-8"))
            if directory == "flow-hint":
                required = [(target/weight, data["branch_sha256"]),
                            (stage/"latent-lab/weights.pt", data["base_hashes"]["vae"]),
                            (stage/"dit-lab/weights.pt", data["base_hashes"]["dit"])]
                for path, expected in required:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                        raise ValueError(f"综合实验录像与 {path.name} 的权重不配套")
                first_film_report.repack(root/directory/"first-film.html",data,target)
                source_scores=root/directory/"solver.json"
                scores=json.loads(source_scores.read_text(encoding="utf-8"))
                if scores["hashes"]!={**data["base_hashes"],"branch":data["branch_sha256"]}:
                    raise ValueError("求解器对照使用的权重与随包模型不同，请重跑 solver_lab.py")
                shutil.copyfile(source_scores,target/source_scores.name)
                solver_report.render(scores,target)
                record=root/directory/"windows.json"
                windows=json.loads(record.read_text(encoding="utf-8"))
                if windows["hashes"]!=scores["hashes"]:
                    raise ValueError("控制时段记录使用的权重与实验 P 不一致")
                shutil.copyfile(record,target/record.name)
                window_report.render(windows,target,course=True)
                shutil.copyfile(root/directory/"reading-targets.json",target/"reading-targets.json")
                reading_route.render(stage,target)
                playback_file=root/directory/"playback.json"
                playback=json.loads(playback_file.read_text(encoding="utf-8"))
                if playback["hashes"]!=scores["hashes"]:
                    raise ValueError("播放对照记录与实验 P 权重不配套")
                shutil.copyfile(playback_file,target/playback_file.name)
                playback_report.render(playback,target,course=True)
            renderer(data, target)
        trace = json.loads((root/"scheduler-lab/record.json").read_text(encoding="utf-8"))
        scheduler_report.check_files(trace, stage)
        scheduler_report.render(trace, stage/"scheduler-lab", course=True)
        depth = stage/"depth-lab"
        depth.mkdir()
        shutil.copyfile(root/"depth-lab/record.json",depth/"record.json")
        depth_record = json.loads((depth/"record.json").read_text(encoding="utf-8"))
        depth_report.render(depth_record,depth,course=True)
        shutil.copyfile(root/"depth-lab/cfg.json", depth/"cfg.json")
        guidance = json.loads((depth/"cfg.json").read_text(encoding="utf-8"))
        for key in ("modules", "weights", "seed", "size", "steps", "prompt", "negative"):
            if depth_record[key] != guidance[key]:
                raise ValueError(f"深度与文字对照的 {key} 不一致")
        for scale in (0, 1):
            old = depth_record["runs"][f"A{scale}"]["film"]
            new = guidance["runs"][f"w7.5-d{scale}"]["film"]
            if [depth_report.unpack(f) for f in old] != [depth_report.unpack(f) for f in new]:
                raise ValueError("两次实测的共同对照组成片不一致")
        cfg_report.render(guidance, depth, course=True)
        pair = stage/"pair-lab"
        pair.mkdir()
        shutil.copyfile(root/"pair-lab/weights.json",pair/"weights.json")
        route_pair_report.render(pair/"weights.json",pair,course=True)
        notebook_report.render(stage/"starter-lab",course=True)
        source = stage / "reproduce"
        source.mkdir()
        shutil.copyfile(root/"reproduce/author_endpoints.json", source/"author_endpoints.json")
        for path in sorted((root / "reproduce").iterdir()):
            if path.is_file() and path.suffix in (".py", ".md", ".html", ".js"):
                shutil.copyfile(path, source/path.name)

        def embed(match):
            kind, name = match.groups()
            directories = {"notebook":"starter-lab","pair":"pair-lab", "depth": "depth-lab", "pixel": "pixel-lab", "control": "control-lab", "flow": "flow-lab",
                           "latent":"latent-lab", "dit":"dit-lab", "workshop": "workshop",
                           "prediction": "prediction-lab", "hunyuan": "hunyuan-lab", "combo": "flow-hint", "scheduler": "scheduler-lab"}
            blob = (stage/directories[kind]/(name+".svg")).read_bytes()
            return f'data-{kind}="{name}" src="data:image/svg+xml;base64,{base64.b64encode(blob).decode()}"'

        name = "video-diffusion-lab.html"
        html = re.sub(r'data-(notebook|pair|depth|workshop|pixel|control|flow|latent|dit|prediction|hunyuan|combo|scheduler)="([a-z-]+)" src="[^"]*"', embed, (root/name).read_text(encoding="utf-8"))
        (stage/name).write_text(html, encoding="utf-8")
        page_index.render(stage, stage/"page-index.html")
        pdf = root/"video-diffusion-guide.pdf"
        if render:
            binary = chrome or next((shutil.which(name) for name in ("google-chrome", "chromium", "chrome")
                                    if shutil.which(name)), None)
            if not binary:
                raise SystemExit("重新打印需要 Chrome 或 Chromium；用 --chrome 指定可执行文件。")
            subprocess.run([binary, "--headless", "--disable-gpu", "--disable-extensions", "--disable-background-networking",
                            "--no-pdf-header-footer", "--generate-pdf-document-outline",
                            "--no-first-run", "--no-default-browser-check",
                            f"--user-data-dir={scratch}/profile", f"--print-to-pdf={stage/pdf.name}",
                            (stage/name).as_uri()], check=True, timeout=120)
            if not (stage/pdf.name).is_file():
                raise RuntimeError("浏览器未输出 PDF")
            shutil.copyfile(stage/pdf.name, pdf)
        else:
            shutil.copyfile(pdf, stage/pdf.name)
        bundle = root / "video-diffusion-kit.zip"
        with tempfile.TemporaryDirectory(dir=root) as destination:
            candidate = Path(destination)/bundle.name
            with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(stage.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(stage))
            os.replace(candidate, bundle)
        return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="重新渲染并替换 PDF；否则打包已有 PDF")
    parser.add_argument("--chrome", help="Chrome 或 Chromium 可执行路径（仅 --render 需要）")
    args = parser.parse_args()
    print("完整离线包：", pack(Path(__file__).resolve().parent.parent, args.render, args.chrome))
