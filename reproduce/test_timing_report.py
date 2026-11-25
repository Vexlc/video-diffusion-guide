import importlib.util
import io
import json
import unittest

from timing_report import DEFAULT, animation, frames, measure


class VideoTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DEFAULT.read_text(encoding="utf-8"))

    def test_same_call_count_different_progress(self):
        ticks = self.data["ticks"]
        self.assertEqual([len(ticks[k]) for k in ("6","24")],[6,24])
        for run in ticks.values():
            self.assertAlmostEqual(run[0][1],1)
            self.assertEqual(run[-1][2:],[1.,0.])
        self.assertAlmostEqual(ticks["24"][5][3],.9238795,places=6)
        at_three = measure(self.data,1,2,5,10,3)
        at_six = measure(self.data,1,2,5,10,6)
        self.assertEqual(at_three["各片此处亮度"],{"partial":238,"6":101,"24":49})
        self.assertEqual(at_three["各片此处亮度"],at_six["各片此处亮度"])
        self.assertEqual(at_three["总不同格数"],1015)
        self.assertEqual([at_three["片长秒"],at_six["片长秒"]],["2","1"])
        self.assertEqual([at_three["末帧起始秒"],at_six["末帧起始秒"]],["5/3","5/6"])

    @unittest.skipUnless(importlib.util.find_spec("PIL"),"独立解码动态 PNG 需要 Pillow")
    def test_animation_matches_every_source_pixel_and_delay(self):
        from PIL import Image
        for case in (1,8):
            for budget in ("6","24"):
                film = frames(self.data,budget,case)
                for fps in (3,6):
                    with Image.open(io.BytesIO(animation(film,16,12,fps))) as clip:
                        self.assertEqual(clip.n_frames,6)
                        self.assertEqual(clip.size,(192,144))
                        self.assertEqual(clip.info["loop"],1)
                        for i,source in enumerate(film):
                            clip.seek(i)
                            self.assertAlmostEqual(clip.info["duration"],1000/fps)
                            expected = Image.frombytes("L",(16,12),source).resize((192,144),resample=0)
                            self.assertEqual(clip.convert("L").tobytes(),expected.tobytes())

    @unittest.skipUnless(importlib.util.find_spec("torch"),"重新运行采样器需要 PyTorch")
    def test_each_budget_traverses_entire_angle_schedule(self):
        import torch
        from pixel_lab import amplitudes,sample
        y = torch.full((1,6,1,1),-.7)
        z = torch.full_like(y,1.3)
        for budget in (1,6,24):
            times=[]
            def oracle(x,t):
                times.append(t.item())
                a,s=amplitudes(t)
                self.assertTrue(torch.allclose(x,a*y+s*z,atol=2e-6))
                return a*z-s*y
            end,_ = sample(oracle,z,steps=budget)
            self.assertEqual(len(times),budget)
            self.assertTrue(torch.allclose(end,y,atol=2e-6))
            self.assertTrue(torch.allclose(torch.tensor(times),torch.arange(budget,0,-1)/budget))
        with self.assertRaises(ValueError):
            sample(oracle,z,steps=0)
