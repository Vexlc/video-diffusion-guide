// 实验 J 的单次前向传播。特征的内存顺序是 [帧, 通道, 行, 列]。
"use strict";
function unpack(spec) {
  const bytes = Uint8Array.from(atob(spec.data), c => c.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const values = Float32Array.from({length: bytes.length / 4}, (_, i) => view.getFloat32(4*i, true));
  return {shape: spec.shape, values};
}

function volume(frames, channels, height, width) {
  return {shape: [frames, channels, height, width], values: new Float32Array(frames*channels*height*width)};
}

function convolution(input, weights, bias, stride=1) {
  const [frames, channels, height, width] = input.shape;
  const [outChannels, , kernel] = weights.shape;
  const border = Math.floor(kernel/2);
  const outH = Math.floor((height-1)/stride)+1, outW = Math.floor((width-1)/stride)+1;
  const output = volume(frames, outChannels, outH, outW);
  const padH = height+2*border, padW = width+2*border;
  const padded = new Float32Array(channels*padH*padW);
  const w = weights.values;
  for (let f=0; f<frames; f++) {
    for (let c=0; c<channels; c++) for (let y=0; y<height; y++) {
      const start = ((f*channels+c)*height+y)*width;
      padded.set(input.values.subarray(start, start+width), (c*padH+y+border)*padW+border);
    }
    for (let oc=0; oc<outChannels; oc++) for (let y=0; y<outH; y++) for (let x=0; x<outW; x++) {
      let total = bias.values[oc], wi = oc*channels*kernel*kernel;
      for (let c=0; c<channels; c++) for (let ky=0; ky<kernel; ky++) {
        let pi = (c*padH+y*stride+ky)*padW+x*stride;
        for (let kx=0; kx<kernel; kx++) total += padded[pi++]*w[wi++];
      }
      output.values[((f*outChannels+oc)*outH+y)*outW+x] = total;
    }
  }
  return output;
}

function normalizeAndActivate(input, gamma, beta) {
  const [frames, channels, h, w] = input.shape;
  const perGroup = channels/4 * h*w;
  const output = volume(...input.shape);
  for (let f=0; f<frames; f++) for (let group=0; group<4; group++) {
    const start = (f*4+group)*perGroup;
    let sum = 0, squares = 0;
    for (let i=start; i<start+perGroup; i++) sum += input.values[i];
    const mean = sum/perGroup;
    for (let i=start; i<start+perGroup; i++) squares += (input.values[i]-mean)**2;
    const inverseStd = 1/Math.sqrt(squares/perGroup+1e-5);
    for (let i=start; i<start+perGroup; i++) {
      const c = Math.floor(i/(h*w)) % channels;
      const z = (input.values[i]-mean)*inverseStd*gamma.values[c]+beta.values[c];
      output.values[i] = z/(1+Math.exp(-z));
    }
  }
  return output;
}

function addFeatures(first, second, gain=1) {
  const result = volume(...first.shape);
  for (let i=0; i<result.values.length; i++) result.values[i] = first.values[i]+gain*second.values[i];
  return result;
}

function shiftFrames(input) {
  const [frames, channels, height, width] = input.shape;
  const plane = height*width, quarter = channels/4, output = volume(...input.shape);
  for (let f=0; f<frames; f++) for (let c=0; c<channels; c++) {
    const source = c<quarter ? Math.max(0,f-1) : (c<2*quarter ? Math.min(frames-1,f+1) : f);
    const i=(source*channels+c)*plane;
    output.values.set(input.values.subarray(i,i+plane), (f*channels+c)*plane);
  }
  return output;
}

function joinResolutions(fine, coarse, temporal=true) {
  const [frames, cFine, h, w] = fine.shape, [, cCoarse, lowH, lowW] = coarse.shape;
  const output = volume(frames,cFine+cCoarse,h,w);
  const shifted = temporal ? shiftFrames(coarse) : coarse;
  for (let f=0; f<frames; f++) {
    const start=f*(cFine+cCoarse)*h*w;
    output.values.set(fine.values.subarray(f*cFine*h*w,(f+1)*cFine*h*w),start);
    for (let c=0; c<cCoarse; c++) for (let y=0; y<h; y++) for (let x=0; x<w; x++) {
      const lowY=Math.floor(y*lowH/h), lowX=Math.floor(x*lowW/w);
      output.values[start+((cFine+c)*h+y)*w+x]=shifted.values[((f*cCoarse+c)*lowH+lowY)*lowW+lowX];
    }
  }
  return output;
}

function makeHintNetwork(data) {
  const load = collection => Object.fromEntries(Object.entries(collection).map(([key,spec])=>[key,unpack(spec)]));
  const fixed = load(data.fixed), weights = load(data.layers);
  const conv = (x,key,stride=1) => convolution(x,weights[key+".weight"],weights[key+".bias"],stride);
  const block = (x,key,stride=1) => normalizeAndActivate(conv(x,key+".layers.0",stride),
    weights[key+".layers.1.weight"],weights[key+".layers.1.bias"]);

  function branch(points) {
    const hint = volume(6,1,12,16);
    points.forEach((point,frame)=>{if(point)hint.values[(frame*12+point[1])*16+point[0]]=1});
    const high = addFeatures(fixed.enter,conv(hint,"branch.hint"));
    const low = block(shiftFrames(high),"branch.down",2);
    const middle = addFeatures(low,block(shiftFrames(low),"branch.middle"));
    return {fine: conv(high,"branch.fine"), coarse: conv(middle,"branch.coarse")};
  }
  function decode(residuals, fineScale, coarseScale) {
    const high = addFeatures(fixed.skip,residuals.fine,fineScale);
    const low = addFeatures(fixed.low,residuals.coarse,coarseScale);
    const joined = block(joinResolutions(high,low),"base.refine");
    const v = conv(joined,"base.exit");
    const clean = volume(...v.shape);
    for(let i=0;i<v.values.length;i++) clean.values[i]=data.a*fixed.x.values[i]-data.sigma*v.values[i];
    return {v,clean};
  }
  return {fixed,branch,decode};
}
