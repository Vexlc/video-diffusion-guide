"use strict";
function floats(base64){const bytes=Uint8Array.from(atob(base64),ch=>ch.charCodeAt(0));const view=new DataView(bytes.buffer);return Float32Array.from({length:bytes.length/4},(_,i)=>view.getFloat32(4*i,true))}
function decoderLayers(specs){return specs.map(spec=>({...spec,w:floats(spec.weights)}))}
function convolution(input,h,w,layer){
  const up=layer.kind==="up",oh=up?2*h:h,ow=up?2*w:w,k=layer.kernel;
  const output=new Float64Array(layer.out*oh*ow);
  for(let o=0;o<layer.out;o++)output.fill(layer.bias[o],o*oh*ow,(o+1)*oh*ow);
  for(let c=0;c<layer.in;c++)for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const v=input[(c*h+y)*w+x];
    for(let o=0;o<layer.out;o++){
      const offset=(up?c*layer.out+o:o*layer.in+c)*k*k;
      for(let ky=0;ky<k;ky++){
        const destY=up?2*y+ky-1:y+1-ky;
        if(destY<0||destY>=oh)continue;
        const target=(o*oh+destY)*ow;
        for(let kx=0;kx<k;kx++){
          const destX=up?2*x+kx-1:x+1-kx;
          if(destX>=0&&destX<ow)output[target+destX]+=v*layer.w[offset+ky*k+kx];
        }
      }
    }
  }
  return [output,oh,ow];
}
function decodeLatentPair(group,layers){
  let out=Float64Array.from(group),h=3,w=4;
  layers.forEach((layer,index)=>{
    [out,h,w]=convolution(out,h,w,layer);
    for(let i=0;i<out.length;i++)out[i]=index===layers.length-1?2/(1+Math.exp(-out[i]))-1:out[i]/(1+Math.exp(-out[i]));
  });return out;
}

if(typeof module!=="undefined")module.exports={floats,decoderLayers,decodeLatentPair};
