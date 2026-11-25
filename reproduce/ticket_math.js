// 每个样本依次保存两个小端 float64 和一个分量标记。id 从 1 开始。
function readTickets(encoded) {
  const raw=Uint8Array.from(atob(encoded),letter=>letter.charCodeAt(0));
  const view=new DataView(raw.buffer), count=raw.length/17;
  const z=new Float64Array(count), y=new Float64Array(count), sign=new Int8Array(count);
  for(let i=0;i<count;i++) {
    z[i]=view.getFloat64(17*i,true);y[i]=view.getFloat64(17*i+8,true);sign[i]=view.getUint8(17*i+16)?1:-1;
  }
  return {z,y,sign,count};
}

function fieldAt(x,t) {
  const variance=(1-t)**2+(.5*t)**2;
  const slope=(.25*t-(1-t))/variance;
  const p=(1+Math.tanh(2*t*x/variance))/2;
  const left=-2+slope*(x+2*t),right=2+slope*(x-2*t);
  const density=(Math.exp(-((x+2*t)**2)/(2*variance))+Math.exp(-((x-2*t)**2)/(2*variance)))/(2*Math.sqrt(2*Math.PI*variance));
  return {p,left,right,density,v:(1-p)*left+p*right};
}

function localStats(data,count,x,t,radius,group) {
  const ids=[], bins=Array.from({length:16},()=>[0,0]);
  let sum=0, left=0;
  for(let i=0;i<count;i++) {
    const pos=(1-t)*data.z[i]+t*data.y[i];
    if(Math.abs(pos-x)>radius || (group!==0 && data.sign[i]!==group))continue;
    const u=data.y[i]-data.z[i];
    ids.push(i);sum+=u;left+=Number(data.sign[i]<0);
    const b=Math.max(0,Math.min(15,Math.floor(u)+8));bins[b][Number(data.sign[i]>0)]++;
  }
  if(!ids.length)return {ids,bins,left,mean:null,variance:null};
  const mean=sum/ids.length;
  let deviation=0;
  for(const i of ids)deviation+=(data.y[i]-data.z[i]-mean)**2;
  return {ids,bins,left,mean,variance:deviation/ids.length};
}

function normalCdf(x) {
  // 对标准正态密度从 0 到 |x| 做分段八点 Gauss-Legendre 积分。
  const end=Math.min(9,Math.abs(x)), parts=Math.max(1,Math.ceil(end)), half=end/(2*parts);
  const nodes=[.1834346424956498,.525532409916329,.7966664774136267,.9602898564975363];
  const weights=[.362683783378362,.3137066458778873,.2223810344533745,.1012285362903763];
  let area=0;
  for(let j=0;j<parts;j++){
    const mid=(2*j+1)*half;
    for(let k=0;k<nodes.length;k++){
      const d=nodes[k]*half;
      area+=half*weights[k]*(Math.exp(-.5*(mid-d)**2)+Math.exp(-.5*(mid+d)**2))/Math.sqrt(2*Math.PI);
    }
  }
  return Math.max(0,Math.min(1,.5+Math.sign(x)*area));
}

function mixtureCdf(x,t) {
  const scale=Math.sqrt((1-t)**2+(.5*t)**2);
  return (normalCdf((x-2*t)/scale)+normalCdf((x+2*t)/scale))/2;
}

function countGate(data,count,x,t,halfSpan) {
  const early=Math.max(0,t-halfSpan),late=Math.min(1,t+halfSpan);
  // 把门左侧的概率折算成初始高斯上的阈值，避免逐个求中途位置。
  function cutoff(time) {
    const probability=mixtureCdf(x,time);
    if(probability===0)return -Infinity;
    if(probability===1)return Infinity;
    let lo=-9,hi=9;
    for(let k=0;k<55;k++){
      const mid=(lo+hi)/2;
      if(normalCdf(mid)<probability)lo=mid;else hi=mid;
    }
    return (lo+hi)/2;
  }
  const cutEarly=cutoff(early),cutLate=cutoff(late);
  let before=0, after=0, left=0, right=0, odeBefore=0, odeAfter=0;
  for(let i=0;i<count;i++) {
    const pre=(1-early)*data.z[i]+early*data.y[i]<x;
    const post=(1-late)*data.z[i]+late*data.y[i]<x;
    odeBefore+=Number(data.z[i]<cutEarly);odeAfter+=Number(data.z[i]<cutLate);
    before+=Number(pre);after+=Number(post);right+=Number(pre&&!post);left+=Number(!pre&&post);
  }
  const net=right-left,field=fieldAt(x,t);
  return {early,late,before,after,right,left,net,odeBefore,odeAfter,odeRight:Math.max(0,odeBefore-odeAfter),odeLeft:Math.max(0,odeAfter-odeBefore),empirical:net/(count*(late-early)),
    exact:(mixtureCdf(x,early)-mixtureCdf(x,late))/(late-early),instant:field.density*field.v};
}
