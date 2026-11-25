"use strict";
// 在 Node.js 18+ 上运行与离线页面相同的算子，与 PyTorch 录像或新计算的轨迹比较。
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const net=require("./first_film.js"),{floats}=require("./pair_decoder.js");
const root=process.argv[2];
const html=fs.readFileSync(path.join(root,"first-film.html"),"utf8");
const model=net.filmModel(JSON.parse(html.match(/<script id="model" type="application\/json">(.*?)<\/script>/s)[1]));
let checked=0,maxState=0,maxSpeed=0,maxGray=0,maxPixel=0;
function maxDiff(actual,expected){
 assert.equal(actual.length,expected.length);
 let err=0;for(let i=0;i<actual.length;i++)err=Math.max(err,Math.abs(actual[i]-expected[i]));
 return err;
}
function compare(p,states,speeds,centers,checkPixels){
 const run=net.prepare(model,p);
 for(let n=0;n<=24;n++){
  maxState=Math.max(maxState,maxDiff(run.x,states.subarray(n*72,(n+1)*72)));
  if(n===0||n===12||n===24){
   const frames=net.decodeFilm(model,run.x);
   checkPixels(n,frames);
   if(n===24)assert.deepEqual(net.centers(frames),centers);
  }
  if(n<24)maxSpeed=Math.max(maxSpeed,maxDiff(net.advance(model,run,n),speeds.subarray(n*72,(n+1)*72)));
 }
 checked++;
}
if(process.argv[3]){
 const rows=JSON.parse(fs.readFileSync(process.argv[3],"utf8"));
 for(const r of rows)compare(r.p,floats(r.x),floats(r.u),r.centers,(n,frames)=>{
   maxPixel=Math.max(maxPixel,maxDiff(frames,floats(r.decoded[n])));
 });
}else{
 const d=JSON.parse(fs.readFileSync(path.join(root,"numbers.json"),"utf8"));
 for(const [gain,r] of Object.entries(d.records)){
  const x=floats(r.x.data),u=floats(r.v.data),bytes=Buffer.from(r.pixels,"base64");
  for(let c=0;c<12;c++){
   const example=d.examples[c%4],p={col:example.point[0],row:example.point[1],seed:d.eval_seed+Math.floor(c/4),
                                   direction:example.label==="向左"?"left":"right",strength:Number(gain)};
   compare(p,x.subarray(c*25*72,(c+1)*25*72),u.subarray(c*24*72,(c+1)*24*72),r.centers[c],(n,frames)=>{
    const gray=Uint8Array.from(frames,v=>Math.round((v+1)*127.5)),start=(c*25+n)*1152;
    maxGray=Math.max(maxGray,maxDiff(gray,bytes.subarray(start,start+1152)));
   });
  }
 }
}
console.log(JSON.stringify({checked,maxState,maxSpeed,maxPixel,maxGray}));
assert.ok(maxState<.0001&&maxSpeed<.0002&&maxPixel<.0003&&maxGray<=1);
