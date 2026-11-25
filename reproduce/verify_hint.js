// 用独立 PyTorch 前向的输出，逐个元素核对离线网页使用的算子。
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname,"hint_network.js"),"utf8");
const {makeHintNetwork, unpack} = vm.runInNewContext(script+"\n;({makeHintNetwork,unpack})",{atob});
const input = process.argv[2] || path.join(__dirname,"../control-lab/probe.json");
const data=JSON.parse(fs.readFileSync(input,"utf8")), network=makeHintNetwork(data);
let largest=0,count=0;
for(const [name,test] of Object.entries(data.answers)){
  const residuals=network.branch(test.points);
  const prediction=network.decode(residuals,...test.scales);
  let error=0;
  for(const [key,actual] of Object.entries({...residuals,...prediction})){
    const expected=unpack(test[key]);
    if(actual.values.length!==expected.values.length) throw Error(name+" "+key+" 数量不符");
    for(let i=0;i<actual.values.length;i++){
      const delta=Math.abs(actual.values[i]-expected.values[i]);
      if(!Number.isFinite(delta)||delta>1e-4)throw Error(name+" "+key+"["+i+"] 误差 "+delta);
      error=Math.max(delta,error);count++;
    }
  }
  console.log(name,error.toExponential(3));largest=Math.max(largest,error);
}
console.log(count+" 个特征值与输出通过核对，最大绝对差 "+largest.toExponential(4));
