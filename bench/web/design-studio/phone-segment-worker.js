/* Local CPU/WASM inference only. No camera frame leaves this worker. */
'use strict';
let segmenter=null,BrushMode=null,activeEpoch=null,imageIdentity=null;
function localURL(path){const url=new URL(path,self.location.href);if(url.origin!==self.location.origin)throw Error('Runtime assets must be hosted on this origin.');return url.href;}
async function verified(asset){if(!asset||!Number.isSafeInteger(asset.bytes)||!(/^[a-f0-9]{64}$/i).test(asset.sha256))throw Error('Runtime asset manifest is invalid.');const response=await fetch(localURL(asset.url));if(!response.ok)throw Error('A local runtime asset is unavailable.');const bytes=await response.arrayBuffer();if(bytes.byteLength!==asset.bytes)throw Error('Runtime asset size verification failed.');const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');if(hash!==asset.sha256.toLowerCase())throw Error('Runtime asset integrity verification failed.');return bytes;}
self.onmessage=async({data})=>{
  const {id,type,epoch}=data;
  try{
    if(type==='initialize'){
      segmenter?.close();segmenter=null;activeEpoch=epoch;imageIdentity=null;
      const response=await fetch('/phone-runtime/manifest.json',{cache:'no-store'});if(!response.ok)throw Error('Automatic selection runtime is not installed.');
      const manifest=await response.json();if(manifest.schema!=='visualizeit_phone_runtime_v1'||manifest.api!=='interactive_segmenter_v2_strokes'||typeof manifest.runtime_version!=='string')throw Error('The installed runtime does not declare the v2 stroke API.');
      const model=await verified(manifest.model);await verified(manifest.module);
      const runtime=await import(localURL(manifest.module.url));
      // The verified 1.1.1 declarations define these enum values, but the JS
      // bundle omits the enum export. Do not apply this bridge to other versions.
      BrushMode=runtime.BrushMode||(manifest.runtime_version==='1.1.1'?{POSITIVE:1,NEGATIVE:2}:null);
      if(!runtime.InteractiveSegmenter||!runtime.FilesetResolver||!BrushMode||BrushMode.POSITIVE===undefined||BrushMode.NEGATIVE===undefined)throw Error('The installed runtime lacks the v2 stroke API.');
      const files=await runtime.FilesetResolver.forVisionTasks(localURL(manifest.wasm_root));
      segmenter=await runtime.InteractiveSegmenter.createFromOptions(files,{baseOptions:{modelAssetBuffer:new Uint8Array(model),delegate:'CPU'}});
      if(typeof segmenter.setImage!=='function'||typeof segmenter.segment!=='function'){segmenter.close();segmenter=null;throw Error('The installed runtime exposes an incompatible segmentation API.');}
      self.postMessage({id,type:'ready',epoch});
    }else if(type==='process'){
      let result=null;
      try{
        if(!segmenter||epoch!==activeEpoch)throw Error('Selection session is no longer active.');
        const identity=JSON.stringify(['epoch','frameId','timestamp','width','height','orientation'].map(k=>data.tag[k]));
        if(identity!==imageIdentity){segmenter.setImage(data.image);imageIdentity=identity;}
        result=segmenter.segment(data.strokes.map(s=>({brushMode:s.mode==='negative'?BrushMode.NEGATIVE:BrushMode.POSITIVE,point:s.points,isCompleted:true})));
        if(!result||typeof result.getAsFloat32Array!=='function')throw Error('The runtime returned an incompatible v2 confidence mask.');
        if(result.width!==data.tag.width||result.height!==data.tag.height)throw Error('The selection mask dimensions do not match its captured frame.');
        const values=result.getAsFloat32Array(),mask=new Uint8Array(values.length);let selected=0;
        for(let i=0;i<values.length;i++){mask[i]=values[i]>=0.5?255:0;if(mask[i])selected++;}
        if(!selected)throw Error('No object was selected. Tap another point on the object.');
        self.postMessage({id,type:'mask',epoch,tag:data.tag,mask:mask.buffer},[mask.buffer]);
      }finally{result?.close();data.image?.close();}
    }else if(type==='reset'){activeEpoch=epoch;imageIdentity=null;self.postMessage({id,type:'reset',epoch});}
    else if(type==='close'){segmenter?.close();segmenter=null;activeEpoch=null;self.close();}
  }catch(error){self.postMessage({id,type:'error',epoch,tag:data.tag,error:String(error.message||error)});}
};
