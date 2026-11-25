"""核验浏览器生成器的权重来源、固定对照题和页面图册中没有的起点。"""
import base64
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from first_film_report import read_model, render, repack


ROOT=Path(__file__).resolve().parents[1]
LAB=ROOT/"flow-hint"
NODE=shutil.which("node")


class FirstFilmTests(unittest.TestCase):
    def test_rebuild_and_checkpoint_guard(self):
        rec=json.loads((LAB/"numbers.json").read_text())
        path=LAB/"first-film.html"
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            repack(path,rec,out)
            self.assertEqual(path.read_bytes(),(out/path.name).read_bytes())
            render(read_model(path),out)
            standalone=(out/path.name).read_text()
            self.assertNotIn('href="../',standalone)
            self.assertNotIn("__GUIDE_NAV__",standalone)
            self.assertIn('href="#results"',standalone)
            with self.assertRaises(ValueError):
                repack(path,{**rec,"branch_sha256":"changed"},out)

    def node_check(self,*args):
        if not NODE or int(subprocess.check_output([NODE,"-p","process.versions.node.split('.')[0]"],text=True))<18:
            self.skipTest("数值核对需要 Node.js 18+")
        run=subprocess.run([NODE,str(ROOT/"reproduce/verify_first_film.js"),str(LAB),*map(str,args)],
                           check=True,text=True,capture_output=True)
        return json.loads(run.stdout)

    def test_all_recorded_runs(self):
        self.assertEqual(self.node_check()["checked"],36)

    @unittest.skipIf(not importlib.util.find_spec("torch"),"直接计算新起点参照需要 PyTorch")
    def test_points_outside_the_recording_against_pytorch(self):
        import torch
        import flow_hint as lab
        from dit_lab import patchify
        device=torch.device("cpu")
        lab.setup(device,437)
        vae,base,branch,saved=lab.load_trained(ROOT,LAB/"branch.pt",device)
        spec=read_model(LAB/"first-film.html")
        for key,net in (("base",base),("branch",branch)):
            for name,t in net.state_dict().items():
                self.assertEqual(base64.b64decode(spec[key][name]),t.numpy().astype("<f4").tobytes())
        def packed(t):
            return base64.b64encode(t.cpu().numpy().astype("<f4").tobytes()).decode()
        records=[]
        starts=lab.initial_states(3,device)
        for col,row,direction,strength,seed in [(4,6,"right",1.,9317),(11,9,"left",.5,9318),(0,0,"left",1.,9319)]:
            p=dict(col=col,row=row,direction=direction,strength=strength,seed=seed)
            x=starts[seed-9317:seed-9317+1]
            states=[patchify(x)]
            speeds=[]
            def decoded(state):
                return vae.decode(saved["mean"]+saved["spread"]*state)
            pixels={0:packed(decoded(x))}
            def observe(n,current,v,following,t,hint):
                states.append(patchify(following))
                speeds.append(patchify(v))
                if n+1 in (12,24):
                    pixels[n+1]=packed(decoded(following))
            labels=torch.tensor([lab.LEFT if direction=="left" else lab.RIGHT])
            last=lab.solve(base,branch,x,labels,torch.tensor([[col,row]]),strength,observe)
            records.append(dict(p=p,x=packed(torch.cat(states)),u=packed(torch.cat(speeds)),decoded=pixels,
                                centers=lab.brightest_centers(decoded(last))[0].tolist()))
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/"values.json"
            file.write_text(json.dumps(records))
            self.assertEqual(self.node_check(file)["checked"],len(records))
