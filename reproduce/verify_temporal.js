// 在 Node 18+ 上用独立的 PyTorch 输出核对网页实时计算的每个像素。
"use strict";
const fs=require("node:fs"), path=require("node:path"), vm=require("node:vm");
const source=["hint_network.js","temporal_network.js"].map(file=>fs.readFileSync(path.join(__dirname,file),"utf8")).join("\n");
const {unpack,makeTemporalNetwork}=vm.runInNewContext(source+"\n;({unpack,makeTemporalNetwork})",{atob});
const data=JSON.parse(fs.readFileSync(process.argv[2]||path.join(__dirname,"../pixel-lab/temporal.json"),"utf8"));
const predict=makeTemporalNetwork(data);
let count=0, worst=0;
function compare(key,edit,expected){
  const actual=predict(key,edit).values;
  if(actual.length!==expected.length)throw Error("输出尺寸不符");
  for(let i=0;i<expected.length;i++){
    const error=Math.abs(actual[i]-expected[i]);
    if(!(error<1e-4))throw Error(key+" 像素 "+i+" 的绝对误差 "+error);
    worst=Math.max(worst,error);count++;
  }
}
for(const test of data.edits)compare(test.key,test.edit,unpack(test.v).values);
for(const [key,run] of Object.entries(data.runs))compare(key,null,unpack(run).values.subarray(0,6*12*16));
console.log(count+" 个预测值通过核对，最大绝对误差 "+worst.toExponential(4));
