"use strict";
// 输入按 18 个块排列，每块 4 个潜码分量。隐藏层每块 48 维。
const TOKENS=18, DIM=48, STEPS=24;
const decoderOps=typeof module==="undefined"?{floats,decoderLayers,decodeLatentPair}:require("./pair_decoder.js");

function linear(input,params,name){
  const weights=params[name+".weight"],bias=params[name+".bias"];
  const outputs=bias.length,inputs=weights.length/outputs,rows=input.length/inputs;
  const result=new Float32Array(rows*outputs);
  for(let row=0;row<rows;row++)for(let o=0;o<outputs;o++){
    let sum=bias[o];
    const first=row*inputs,start=o*inputs;
    for(let i=0;i<inputs;i++)sum+=input[first+i]*weights[start+i];
    result[row*outputs+o]=sum;
  }
  return result;
}
function normalize(h){
  const result=new Float32Array(h.length);
  for(let row=0;row<h.length;row+=DIM){
    let mean=0,squares=0;
    for(let i=0;i<DIM;i++)mean+=h[row+i];mean/=DIM;
    for(let i=0;i<DIM;i++)squares+=(h[row+i]-mean)**2;
    const inv=1/Math.sqrt(squares/DIM+1e-5);
    for(let i=0;i<DIM;i++)result[row+i]=(h[row+i]-mean)*inv;
  }
  return result;
}
const silu=x=>x/(1+Math.exp(-x));
function gelu(x){
  // GELU = x Φ(x)。有理多项式近似高斯误差函数。
  const z=Math.abs(x)/Math.SQRT2,t=1/(1+.3275911*z);
  const poly=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t;
  const erf=Math.sign(x)*(1-poly*Math.exp(-z*z));
  return .5*x*(1+erf);
}
function attention(qkv){
  const headSize=16,stride=3*DIM,result=new Float32Array(TOKENS*DIM);
  const scores=new Float64Array(TOKENS);
  for(let head=0;head<3;head++)for(let query=0;query<TOKENS;query++){
    const offset=head*headSize,q=query*stride+offset;
    let largest=-Infinity;
    for(let key=0;key<TOKENS;key++){
      const k=key*stride+DIM+offset;
      let dot=0;for(let i=0;i<headSize;i++)dot+=qkv[q+i]*qkv[k+i];
      scores[key]=dot/Math.sqrt(headSize);largest=Math.max(largest,scores[key]);
    }
    let mass=0;
    for(let k=0;k<TOKENS;k++){scores[k]=Math.exp(scores[k]-largest);mass+=scores[k]}
    for(let i=0;i<headSize;i++){
      let mixed=0;
      for(let k=0;k<TOKENS;k++)mixed+=scores[k]/mass*qkv[k*stride+2*DIM+offset+i];
      result[query*DIM+offset+i]=mixed;
    }
  }
  return result;
}
function conditioned(h,mod,offset){
  const row=normalize(h);
  for(let i=0;i<row.length;i++){const c=i%DIM;row[i]=row[i]*(1+mod[offset+DIM+c])+mod[offset+c]}
  return row;
}
function block(params,layer,h,cond){
  const key="blocks."+layer+".";
  const mod=linear(cond.map(silu),params,key+"modulate.1");
  const qkv=linear(conditioned(h,mod,0),params,key+"qkv");
  const projected=linear(attention(qkv),params,key+"projection");
  const after=h.map((v,i)=>v+mod[2*DIM+i%DIM]*projected[i]);
  const wide=linear(conditioned(after,mod,3*DIM),params,key+"mlp.0");
  const output=linear(wide.map(gelu),params,key+"mlp.2");
  return after.map((v,i)=>v+mod[5*DIM+i%DIM]*output[i]);
}
function pointInput(col,row){
  // 与训练时的 unfold 顺序相同。点图的 6 个空间块各有 4×8 格。
  const grid=new Float32Array(6*32);
  for(let p=0;p<6;p++)for(let y=0;y<4;y++)for(let x=0;x<8;x++){
    const dx=p%2*8+x-col,dy=Math.floor(p/2)*4+y-row;
    grid[p*32+y*8+x]=Math.exp(-(dx*dx+dy*dy)/8);
  }
  return grid;
}
function filmModel(spec){
  const unpack=group=>Object.fromEntries(Object.entries(group).map(([name,data])=>[name,decoderOps.floats(data)]));
  return {base:unpack(spec.base),branch:unpack(spec.branch),decoder:decoderOps.decoderLayers(spec.decoder),
          mean:spec.mean,spread:spec.spread,starts:spec.starts};
}
function prepare(model,settings){
  const start=model.starts[String(settings.seed)];
  if(!start)throw Error("请选择随包提供的噪声种子");
  const hint=linear(pointInput(settings.col,settings.row),model.branch,"hint");
  const label=(settings.direction==="left"?1:2)*DIM;
  return {x:Float32Array.from(start),hint,label,strength:settings.strength};
}
function velocity(model,run,t){
  const base=model.base,branch=model.branch;
  const time=linear(linear([t,t*t,Math.sin(Math.PI*t)],base,"time.0").map(silu),base,"time.2");
  const cond=time.map((v,i)=>v+base["label.weight"][run.label+i]);
  let h=linear(run.x,base,"input").map((v,i)=>v+base.position[i]);
  let extra=run.strength===0?null:h.map((v,i)=>v+run.hint[i%(6*DIM)]);
  for(let k=0;k<2;k++){
    h=block(base,k,h,cond);
    if(extra){
      extra=block(branch,k,extra,cond);
      const residual=linear(extra,branch,"exits."+k);
      h=h.map((v,i)=>v+run.strength*residual[i]);
    }
  }
  const final=normalize(h).map((v,i)=>v*base["output.0.weight"][i%DIM]+base["output.0.bias"][i%DIM]);
  return linear(final,base,"output.1");
}
function advance(model,run,n,steps=STEPS){
  const v=velocity(model,run,n/steps);
  run.x=run.x.map((x,i)=>x+v[i]/steps);
  return v;
}
function decodeFilm(model,tokens){
  const frames=new Float32Array(6*192);
  for(let group=0;group<3;group++){
    const latent=new Float32Array(24);
    for(let row=0;row<3;row++)for(let col=0;col<4;col++)for(let chan=0;chan<2;chan++){
      const token=group*6+row*2+Math.floor(col/2),component=2*chan+col%2;
      latent[chan*12+row*4+col]=model.mean[chan]+model.spread[chan]*tokens[token*4+component];
    }
    frames.set(decoderOps.decodeLatentPair(latent,model.decoder),group*384);
  }
  return frames;
}
function centers(frames){
  const result=[];
  for(let f=0;f<6;f++){
    let max=-Infinity,point;
    for(let row=1;row<=10;row++)for(let col=1;col<=14;col++){
      let sum=0;
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)sum+=frames[f*192+(row+dy)*16+col+dx];
      if(sum>max){max=sum;point=[col,row]}
    }
    result.push(point);
  }
  return result;
}
function budgetRun(model,settings,budget,mode){
  if(![2,4,8,12,24].includes(budget)||!["stop","euler","midpoint"].includes(mode))throw Error("无效的查询预算或积分方法");
  const steps=mode==="stop"?STEPS:(mode==="midpoint"?budget/2:budget);
  const count=mode==="midpoint"?budget/2:budget;
  const run=prepare(model,settings),index=32,path=[run.x[index]],queries=[];
  for(let n=0;n<count;n++){
    const time=n/steps,previous=run.x;
    if(mode==="midpoint"){
      const v0=velocity(model,run,time);
      const trial=previous.map((x,i)=>x+v0[i]/(2*steps));
      const halfTime=(n+.5)/steps;
      const vHalf=velocity(model,{...run,x:trial},halfTime);
      queries.push([time,previous[index],v0[index]],[halfTime,trial[index],vHalf[index]]);
      run.x=previous.map((x,i)=>x+vHalf[i]/steps);
    }else{
      const v=advance(model,run,n,steps);
      queries.push([time,previous[index],v[index]]);
    }
    path.push(run.x[index]);
  }
  const frames=decodeFilm(model,run.x);
  return {mode,steps,updates:count,at:count/steps,path,queries,frames,centers:centers(frames)};
}
function windowRun(model,settings,begin,end){
  if(!Number.isInteger(begin)||!Number.isInteger(end)||begin<0||begin>end||end>STEPS)throw Error("窗口需要 0 到 24 之间两个有序的整数");
  const run=prepare(model,settings),component=8*4,path=[run.x[component]],speeds=[],scales=[];
  for(let n=0;n<STEPS;n++){
    run.strength=begin<=n&&n<end?settings.strength:0;
    scales.push(run.strength);
    speeds.push(advance(model,run,n)[component]);
    path.push(run.x[component]);
  }
  const frames=decodeFilm(model,run.x);
  return {path,speeds,scales,frames,centers:centers(frames)};
}
if(typeof module!=="undefined")module.exports={filmModel,prepare,advance,decodeFilm,centers,budgetRun,windowRun};
