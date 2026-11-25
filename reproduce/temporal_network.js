// 复用实验 J 的卷积、GroupNorm、SiLU 与上采样；在选中的交接处只改一个数。
function makeTemporalNetwork(data) {
  const skip=unpack(data.taps[""]);
  const weights=Object.fromEntries(Object.entries(data.layers).map(([k,v])=>[k,unpack(v)]));
  function conv(input,key,stride=1) {
    return convolution(input,weights[key+".weight"],weights[key+".bias"],stride);
  }
  function block(input,key,stride=1) {
    return normalizeAndActivate(conv(input,key+".layers.0",stride),
      weights[key+".layers.1.weight"],weights[key+".layers.1.bias"]);
  }
  function addAt(tensor,frame,channel,row,col,delta) {
    const [,channels,h,w]=tensor.shape;
    tensor.values[((frame*channels+channel)*h+row)*w+col]+=delta;
  }
  return function predict(key,edit=null) {
    function route(tensor,layer) {
      let routed=key[layer]==="1"?shiftFrames(tensor):tensor;
      if(edit && edit[0]===layer && edit[5]!==0) {
        routed={shape:routed.shape,values:routed.values.slice()};
        addAt(routed,...edit.slice(1));
      }
      return routed;
    }
    const down=block(route(skip,0),"down",2);
    const low=addFeatures(down,block(route(down,1),"middle"));
    const joined=joinResolutions(skip,low,key[2]==="1");
    if(edit && edit[0]===2) {
      const [,frame,c,row,col,delta]=edit;
      addAt(joined,frame,skip.shape[1]+c,row,col,delta);
    }
    return conv(block(joined,"refine"),"exit");
  };
}
