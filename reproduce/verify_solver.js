"use strict";
const fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict");
const net=require("./first_film.js"),root=process.argv[2];
const html=fs.readFileSync(path.join(root,"first-film.html"),"utf8");
const spec=JSON.parse(html.match(/<script id="model" type="application\/json">(.*?)<\/script>/s)[1]);
const model=net.filmModel(spec),d=JSON.parse(fs.readFileSync(process.argv[3]||path.join(root,"solver.json"),"utf8"));
const s=d.settings||{point:[7,3],direction:"right",strength:1,noise_seed:9317};
const settings={col:s.point[0],row:s.point[1],direction:s.direction,strength:s.strength,seed:s.noise_seed};
let maxState=0,maxSpeed=0,maxGray=0,centerDiffs=0;
for(const row of d.rows){
 const run=net.budgetRun(model,settings,row.budget,row.mode);
 assert.equal(run.queries.length,row.calls);assert.equal(run.updates,row.updates);assert.equal(run.at,row.at);
 run.path.forEach((x,i)=>maxState=Math.max(maxState,Math.abs(x-row.path[i])));
 for(const [i,[t,x,u]] of run.queries.entries()){
  const [pt,px,pu]=row.queries[i];assert.ok(Math.abs(t-pt)<.000001);
  maxState=Math.max(maxState,Math.abs(x-px));maxSpeed=Math.max(maxSpeed,Math.abs(u-pu));
 }
 const bytes=row.film.flat(2);
 run.frames.forEach((v,i)=>maxGray=Math.max(maxGray,Math.abs(Math.round((v+1)*127.5)-bytes[i])));
 run.centers.forEach((c,i)=>{if(c[0]!==row.centers[i][0]||c[1]!==row.centers[i][1])centerDiffs++});
}
console.log(JSON.stringify({runs:d.rows.length,maxState,maxSpeed,maxGray,centerDiffs}));
assert.ok(maxState<.0001&&maxSpeed<.0002&&maxGray<=1);
assert.equal(centerDiffs,0);
