"""分别用逐题加总、已知高斯分布和浏览器算子核对练习里的读数。"""
import base64
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from flow_coupling import NORMAL, cdf, local_check, quantile, training_draws
from flow_tickets import RECORD, gate, inspect, summary, window_records


class FlowTicketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples=list(training_draws(41,120000))

    def test_canonical_window_and_direct_loss(self):
        rows=window_records(self.samples,.25,.5,.025)
        self.assertEqual((len(rows),rows[0][0]),(1065,61))
        stats=summary(rows)
        self.assertEqual(sum(a for a,b in stats["bins"]),189)
        self.assertEqual(sum(b for a,b in stats["bins"]),876)
        self.assertAlmostEqual(stats["mean"],local_check(41)[1]["mean"],places=12)
        for guess in (-4.,0.,stats["mean"],2.9):
            direct=sum((guess-r[-1])**2 for r in rows)/len(rows)
            self.assertAlmostEqual(direct,stats["variance"]+(guess-stats["mean"])**2,places=12)
        self.assertIsNone(summary([])["mean"])

    def test_flux_from_separate_cdf_difference(self):
        for x,t in [(-1,.2),(0,.6),(.25,.5),(1.2,.97)]:
            r=gate(self.samples[:240],x,t,.001)
            self.assertEqual(r["right"]-r["left"],r["before"]-r["after"])
            dt=1e-5
            self.assertAlmostEqual((cdf(x,t-dt)-cdf(x,t+dt))/(2*dt),r["instantaneous_rate"],places=8)

    def test_ode_sides_independent_quantile_evaluation(self):
        rows=self.samples[:19]
        for x,t in [(-.5,0.),(.25,.5),(0,.5),(.5,1.)]:
            r=gate(rows,x,t,.025)
            sides=[(quantile(NORMAL.cdf(z),r["early"])<x,quantile(NORMAL.cdf(z),r["late"])<x) for z,y,s in rows]
            self.assertEqual(r["ode_right"],sum(first and not last for first,last in sides))
            self.assertEqual(r["ode_left"],sum(last and not first for first,last in sides))
        r=gate(self.samples,0,.5,.025)
        self.assertEqual((r["right"],r["left"],r["ode_right"],r["ode_left"]),(1388,1361,0,0))

    @unittest.skipUnless(shutil.which("node"),"浏览器对照需 Node.js 18+")
    def test_browser_math_matches_raw_python_samples(self):
        if int(subprocess.check_output(["node","-p","process.versions.node.split('.')[0]"],text=True))<18:
            self.skipTest("需要 Node.js 18 或更高版本")
        source=Path(__file__).with_name("ticket_math.js").read_text()
        batch=self.samples[:2000]
        packet=b"".join(RECORD.pack(z,y,int(sign>0)) for z,y,sign in batch)
        cases=[(x,t,r,g,d) for x,t in [(-3,0),(0,.5),(.25,.5),(1.5,1)]
               for r,g in [(.01,-1),(.3,0),(.025,1)] for d in [.005,.1]]
        script=source+'''
const input=JSON.parse(require("node:fs").readFileSync(0,"utf8")),d=readTickets(input.blob);
const results=input.cases.map(([x,t,r,g,dt])=>{
 const w=localStats(d,d.count,x,t,r,g);
 return {count:w.ids.length,mean:w.mean,bins:w.bins,gate:countGate(d,d.count,x,t,dt)};
});
console.log(JSON.stringify(results));
'''
        payload={"blob":base64.b64encode(packet).decode("ascii"),"cases":cases}
        result=subprocess.run(["node","-e",script],input=json.dumps(payload),text=True,capture_output=True,check=True)
        for args,js in zip(cases,json.loads(result.stdout)):
            expected=inspect(batch,*args)
            window=expected["window"]
            self.assertEqual((js["count"],js["bins"]),(window["count"],window["bins"]))
            if js["count"]:
                self.assertAlmostEqual(js["mean"],window["mean"],places=12)
            else:
                self.assertIsNone(js["mean"])
            for py_key,js_key in [("net","net"),("before","before"),("after","after"),("ode_right","odeRight"),
                                  ("ode_left","odeLeft"),("ode_before","odeBefore"),("ode_after","odeAfter"),
                                  ("exact_rate","exact"),("instantaneous_rate","instant")]:
                self.assertAlmostEqual(js["gate"][js_key],expected["gate"][py_key],places=10)


if __name__=="__main__":
    unittest.main()
