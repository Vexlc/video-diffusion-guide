"""检查每个讲义问题都能定位到实际实验、函数和终端入口。"""
from html.parser import HTMLParser
from pathlib import Path
import shlex
import unittest

from first_steps import crossing, gate_updates, pixel
from page_index import BOOK, cards


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids=set()

    def handle_starttag(self,tag,attrs):
        ident=dict(attrs).get("id")
        if ident:
            self.ids.add(ident)


class PageIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]
        cls.pages=list(cards(cls.root))

    def test_real_targets_sources_and_commands(self):
        destinations={}
        ids={p["id"] for p in self.pages}
        self.assertEqual(len(ids),len(self.pages))
        for p in self.pages:
            with self.subTest(page=p["number"]):
                local,_,fragment=p["target"].partition("#")
                if local not in destinations:
                    anchors=Anchors()
                    anchors.feed((self.root/local).read_text(encoding="utf-8"))
                    destinations[local]=anchors.ids
                if fragment:
                    self.assertIn(fragment,destinations[local] | (ids if local==BOOK else set()))
                source=(self.root/"reproduce"/p["file"]).read_text().splitlines()
                excerpt=p["source"].splitlines()
                self.assertEqual(excerpt,source[p["first"]-1:p["first"]-1+len(excerpt)])
                command=shlex.split(p["command"])
                self.assertTrue((self.root/"reproduce"/command[1]).is_file())
                self.assertTrue(p["title"] and p["formula"] and p["action"])

    def test_page_31_reads_real_pixel(self):
        p=next(p for p in self.pages if p["id"]=="chapter-31")
        self.assertEqual(p["entry"],"clock_report.Recording.pixel")
        self.assertIn("two-clocks.html#update",p["target"])

    def test_three_first_calculations(self):
        self.assertAlmostEqual(pixel(-.7)[0],.04)
        self.assertAlmostEqual(pixel(-.7)[1],.575)
        self.assertAlmostEqual(pixel(-1)[1],.8)
        self.assertEqual(crossing(.5),((4,0),8,4))
        self.assertEqual(crossing(.75),((4,1),4,3))
        before,after=gate_updates()
        self.assertAlmostEqual(before[-1],0)
        self.assertAlmostEqual(before[-2],-1.6)
        self.assertAlmostEqual(after[-1],-.064)
        self.assertAlmostEqual(after[3],.6)
