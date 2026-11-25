"""用已知位移和网络原始记录检查录像的两只钟、窗口轴序与并列解。"""
import json
import unittest
from pathlib import Path

from flow_hint_report import cell,film
from motion_readout import match_window, pair, picture, probe_record, reference_frame, two_times


class MotionReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.data=json.loads((root/"flow-hint/numbers.json").read_text())
        cls.routes=probe_record(cls.data)

    def test_trace_matches_full_precision_and_all_source_frames(self):
        for case,label in enumerate(("left","right")):
            route=self.routes[label]
            for n in range(25):
                reference=film(self.data["records"]["1.0"],case,n)
                self.assertEqual([picture(route,n,f) for f in range(6)],reference)
            for n,k in ((0,3),(12,1),(23,0)):
                trace=two_times(route,n,2,k,6,(9,3))
                expected=cell(self.data,1,case,n,8,k,0)
                self.assertEqual((trace["u"],trace["x"],trace["next_x"]),
                                 (expected["v"],expected["x"],expected["next"]))

    def test_change_only_playback_rate(self):
        route=self.routes["right"]
        first=two_times(route,12,2,1,6,(9,3))
        fast=two_times(route,12,2,1,12,(9,3))
        self.assertEqual(first["brightness"],[[198,9],[251,77],[255,255]])
        self.assertEqual(first["pixels_per_second"],[6,0])
        self.assertEqual(fast["pixels_per_second"],[12,0])
        self.assertLess(first["u"],0)
        self.assertAlmostEqual(first["x"]+first["u"]/24,first["next_x"],places=7)
        for key in set(first)-{"pixels_per_second"}:
            self.assertEqual(first[key],fast[key])

    def test_ideal_ties_and_measured_nonzero_score(self):
        for label,sign in (("left",-1),("right",1)):
            route=self.routes[label]
            for f in range(5):
                ideal=pair(route,f,"ideal")
                at=route["target"][f]
                for w,count in (("point",9),("strip",3),("square",1)):
                    result=match_window(*ideal,at,w)
                    self.assertEqual(len(result["best"]),count)
                    self.assertIn([sign,0],result["best"])
                    self.assertEqual(result["lowest"],0)
        pictures=pair(self.routes["right"],2,"decoded")
        local=match_window(*pictures,(9,3),"square")
        self.assertEqual(local["best"],[[1,0]])
        self.assertEqual(local["lowest"],5+28+15+38)
        one=match_window(*pictures,(9,3),"point")
        self.assertEqual(len(one["best"]),5)

    def test_uniform_background_and_edges(self):
        a,b=reference_frame((9,3)),reference_frame((10,3))
        self.assertEqual(len(match_window(a,b,(5,8),"square")["best"]),25)
        border=match_window(a,b,(1,1),"square")
        self.assertIsNone(border["costs"][0][0])
        self.assertEqual(border["costs"][2][2],0)
        with self.assertRaises(ValueError):
            match_window(a,b,(0,0),"square")
