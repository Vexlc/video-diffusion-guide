"""核对入门路线的纸上答案和批量对照，覆盖首帧边界与反例。"""
import unittest
from statistics import mean

from first_walk import cohort, direction_changes, lesson


class FirstWalkTests(unittest.TestCase):
    def test_four_book_examples(self):
        a=lesson("A",17)
        self.assertAlmostEqual(a["before"],.8)
        self.assertAlmostEqual(a["after"],-.09486107354653715)
        b=lesson("B",17)
        self.assertEqual((b["before"],b["after"],b["total"]),(0,142,144))
        c=lesson("C",17)
        self.assertEqual((c["before"],c["after"]),(5,0))
        self.assertEqual(direction_changes([0,1,1+2e-7,1,0,0,-1,0]),2)
        d=lesson("D",17)
        self.assertAlmostEqual(d["before"],.28545619027572616)
        self.assertAlmostEqual(d["after"],.46340728651795066)

    def test_each_frame_matches_its_own_truth(self):
        rows,report=cohort()
        for r in rows:
            for key in ("aligned","delayed"):
                errors=[abs(v-t) for v,t in zip(r[key],r["truth"])]
                self.assertEqual(len(errors),8)
                self.assertEqual(errors,r[key+"_error"])
                self.assertEqual(mean(errors),r[key+"_mae"])
            self.assertEqual(r["difference"][0],0.)
            self.assertAlmostEqual(mean(r["difference"]),r["delayed_mae"]-r["aligned_mae"])
        self.assertEqual((report["segments"],report["late_worse"],report["tied"]),(64,63,0))
        self.assertEqual(report["late_better_seeds"],[21])
        self.assertAlmostEqual(report["aligned_mean"],.2110676377261257)
        self.assertAlmostEqual(report["delayed_mean"],.47039472105943675)
        self.assertEqual(sum(d<0 for d in rows[21]["difference"]),5)
        self.assertAlmostEqual(rows[21]["difference"][2],-.365053,delta=1e-4)
