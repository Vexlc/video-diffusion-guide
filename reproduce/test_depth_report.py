"""用随包的实际像素和三处独立读取的特征，验证公开权重实验的对照与 FP16 加法。"""
import json
import struct
import unittest

from depth_report import ROOT, differences, pixel, readout, unpack
from hint_report import values


class DepthReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/"depth-lab/record.json").read_text(encoding="utf-8"))

    def test_only_middle_condition_was_moved(self):
        a,b=self.data["maps"]["A"],self.data["maps"]["B"]
        self.assertEqual([i for i in range(5) if unpack(a[i])!=unpack(b[i])],[2])
        w,h=self.data["size"]
        for name,frames in self.data["maps"].items():
            for f,image in enumerate(frames):
                raw=unpack(image)
                self.assertEqual(len(raw),w*h)
                center=80+24*f+(48 if name=="B" and f==2 else 0)
                self.assertEqual(pixel(raw,w,center,102,1),[241])

    def test_reported_colors_and_frame_scores(self):
        w,h=self.data["size"]
        for strength in (0,1):
            a,b=[self.data["runs"][label+str(strength)]["film"] for label in ("A","B")]
            for f in range(5):
                left,right=unpack(a[f]),unpack(b[f])
                self.assertEqual(len(left),w*h*3)
                stats=differences(left,right)
                self.assertEqual(stats,{k:v[f] for k,v in self.data["delta"][str(strength)].items()})
                if strength==0:self.assertEqual(left,right)
            if strength==1:
                self.assertEqual(pixel(unpack(a[3]),w,152,102,3),[255,118,0])
                self.assertEqual(pixel(unpack(b[3]),w,152,102,3),[0,96,183])
        self.assertEqual(round(self.data["delta"]["1"]["mae"][3],2),36.33)

    def test_first_forward_on_the_same_state(self):
        traces={k:run["trace"] for k,run in self.data["runs"].items()}
        self.assertEqual(len({v["input"] for v in traces.values()}),1)
        for name,trace in traces.items():
            self.assertEqual(trace["h"],traces["A0"]["h"])
            self.assertEqual(trace["tick"],951)
            self.assertEqual(len(trace["shapes"]),13)
            self.assertEqual(trace["shapes"][0],[320,5,24,32])
            self.assertEqual(trace["shapes"][-1],[1280,5,3,4])
            h,r,joined=[values(trace[k]) for k in ("h","r","joined")]
            self.assertEqual(len(joined),5*24*32)
            for a,b,c in zip(h,r,joined):
                self.assertEqual(struct.unpack("<e",struct.pack("<e",a+b))[0],c)
            if name.endswith("0"):self.assertTrue(all(v==0 for v in r))
        for f in range(5):
            a,b=(values(traces[k]["r"])[f*768:(f+1)*768] for k in ("A1","B1"))
            self.assertEqual(a==b,f!=2)
        sums=[readout(traces[name],2,16,12) for name in ("A1","B1")]
        self.assertEqual([round(v["joined"],5) for v in sums],[.52979,.50342])
