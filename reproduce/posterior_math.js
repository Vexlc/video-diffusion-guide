// 与 posterior_lab.py 使用相同的两种干净值及独立标准高斯噪声。
const bodyValue=.8, backgroundValue=-.55, a=.8, sigma=.6, radius=.1;
const phi=z=>Math.exp(-.5*z*z)/Math.sqrt(2*Math.PI);
const noiseFor=(x,y)=>(x-a*y)/sigma;
function estimate(x,mode,neighbor){
  let prior=.5;
  if(mode==='shared'){
    const body=phi(noiseFor(neighbor,bodyValue)),back=phi(noiseFor(neighbor,backgroundValue));
    prior=body/(body+back);
  }
  const zBody=noiseFor(x,bodyValue),zBack=noiseFor(x,backgroundValue);
  const body=prior*phi(zBody),back=(1-prior)*phi(zBack);
  const p=body/(body+back),noise=p*zBody+(1-p)*zBack;
  return {body:p,clean:p*bodyValue+(1-p)*backgroundValue,noise,score:-noise/sigma,
    density:(body+back)/sigma,loss:p*(zBody-noise)**2+(1-p)*(zBack-noise)**2};
}
function windowMass(x,y){
  const lo=noiseFor(x-radius,y),hi=noiseFor(x+radius,y);
  // 在实际窄窗上做复合 Simpson 积分，避免计算两个极接近的尾部概率之差。
  const intervals=64,step=(hi-lo)/intervals;
  let total=phi(lo)+phi(hi);
  for(let i=1;i<intervals;i++)total+=(i%2?4:2)*phi(lo+i*step);
  return [total*step/3,phi(lo)-phi(hi)];
}
function windowExpected(x,mode,neighbor){
  let total=0,moment=0,bright=0;
  for(const y of [bodyValue,backgroundValue]){
    const [mass,first]=windowMass(x,y);
    const factor=mode==='shared'?windowMass(neighbor,y)[0]:1;
    total+=mass*factor;moment+=first*factor;
    if(y===bodyValue)bright=mass*factor;
  }
  return {body:bright/total,noise:moment/total};
}
function trainingSet(count,seed=539){
  let state=(seed>>>0)||0x9e3779b9;
  function uniform(){state^=state<<13;state^=state>>>17;state^=state<<5;return ((state>>>0)+.5)/2**32}
  function normal(){return Math.sqrt(-2*Math.log(uniform()))*Math.cos(2*Math.PI*uniform())}
  const set={x:new Float64Array(count),shared:new Float64Array(count),independent:new Float64Array(count),
    label:new Float64Array(count),body:new Uint8Array(count)};
  for(let i=0;i<count;i++){
    const y=normal()>0?bodyValue:backgroundValue,other=normal()>0?bodyValue:backgroundValue;
    const z=normal(),otherZ=normal();
    set.x[i]=a*y+sigma*z;set.label[i]=z;set.body[i]=Number(y===bodyValue);
    set.shared[i]=a*y+sigma*otherZ;set.independent[i]=a*other+sigma*otherZ;
  }
  return set;
}
function sampleWindow(set,count,x,mode,neighbor){
  let kept=0,bright=0,total=0;
  for(let i=0;i<count;i++){
    if(Math.abs(set.x[i]-x)>radius || (mode!=='hidden' && Math.abs(set[mode][i]-neighbor)>radius))continue;
    bright+=set.body[i];kept++;total+=set.label[i];
  }
  return {kept,body_count:bright,noise:kept?total/kept:null};
}
