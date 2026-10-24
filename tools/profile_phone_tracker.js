'use strict';
// Bounded manufactured CPU/support audit; neither phone speed nor model accuracy.
const {performance}=require('node:perf_hooks');
const {Tracker}=require('../bench/web/design-studio/phone-tracker.js');
const W=192,H=144;
function noise(seed){const out=new Uint8Array(W*H);for(let i=0;i<out.length;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;out[i]=seed>>>24;}return out;}
const object=noise(12345),background=noise(54321);
function fixture(kind,dx=0){
  const rgba=new Uint8ClampedArray(W*H*4),mask=new Uint8Array(W*H);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const inside=x>=64&&x<128&&y>=28&&y<116,ox=x-dx;
    let v=background[y*W+x];
    if(ox>=64&&ox<128&&y>=28&&y<116){v=object[y*W+ox];if(kind==='background-dominates')v=110+Math.round(v/7);if(kind==='transparent-background')v=background[y*W+x];}
    if(kind==='flat')v=100;
    const i=(y*W+x)*4;rgba[i]=rgba[i+1]=rgba[i+2]=v;rgba[i+3]=255;if(inside)mask[y*W+x]=255;
  }return {rgba,mask};
}
const rows=[];
for(const kind of ['textured-object','background-dominates','transparent-background','flat']){
  const startTimes=[],updateTimes=[],support=[];
  for(let run=0;run<9;run++){
    const t=new Tracker(),a=fixture(kind),b=fixture(kind,2);let now=performance.now();t.start({rgba:a.rgba,width:W,height:H,timestamp:100,epoch:1,frameId:1});const startMs=performance.now()-now;
    const maskAccepted=t.setMask({mask:a.mask,width:W,height:H,epoch:1,frameId:1});now=performance.now();const result=t.update({rgba:b.rgba,width:W,height:H,timestamp:200});const updateMs=performance.now()-now;
    if(run>0){startTimes.push(startMs);updateTimes.push(updateMs);support.push({maskAccepted,valid:result.valid,reason:result.reason,dx:result.dx,diagnostics:result.diagnostics});}
  }
  const stats=xs=>{const s=xs.slice().sort((a,b)=>a-b);return {median_ms:(s[3]+s[4])/2,p95_ms:s[Math.ceil(.95*s.length)-1],samples:s.length};};
  rows.push({kind,start:stats(startTimes),update:stats(updateTimes),representative:support[0],all_results_equal:support.every(s=>JSON.stringify(s)===JSON.stringify(support[0]))});
}
console.log(JSON.stringify({schema:'visualizeit_phone_tracker_cpu_support_v1',timestamp:new Date().toISOString(),runtime:process.version,resolution:[W,H],warmup:1,repetitions:8,manufactured:true,physical_phone:false,transparent_case_warning:'A valid background match through transparency is not surface attachment evidence.',rows},null,2));
