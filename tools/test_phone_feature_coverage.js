'use strict';
// Independent manufactured translations; no mask/pose ground truth enters tracking.
const assert=require('node:assert/strict');
const {Tracker}=require('../bench/web/design-studio/phone-tracker.js');
const W=192,H=144;
function texture(seed){const out=new Uint8Array(W*H);for(let i=0;i<out.length;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;out[i]=seed>>>24;}return out;}
const bg=texture(54321),target=texture(12345);
function image(dx=0,kind='eligible-low-contrast',backgroundDx=0){
  const rgba=new Uint8ClampedArray(W*H*4);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const sx=x-dx,inside=sx>=64&&sx<128&&y>=28&&y<116;let v=bg[y*W+((x-backgroundDx+W)%W)];
    if(inside){v=110+Math.round(target[y*W+sx]/7);if(kind==='flat-object')v=125;if(kind==='stripes')v=sx%8<4?230:20;if(kind==='dense')v=target[y*W+sx];}
    if(kind==='sparse')v=x>=48&&x<144&&y>=24&&y<120?bg[y*W+x]:125;
    const j=(y*W+x)*4;rgba[j]=rgba[j+1]=rgba[j+2]=v;rgba[j+3]=255;
  }return rgba;
}
function mask(){const m=new Uint8Array(W*H);for(let y=28;y<116;y++)for(let x=64;x<128;x++)m[y*W+x]=255;return m;}
function start(kind){const t=new Tracker();t.start({rgba:image(0,kind),width:W,height:H,timestamp:100,epoch:3,frameId:9});return t;}
function attach(t){return t.setMask({mask:mask(),width:W,height:H,epoch:3,frameId:9});}
let count=0;const check=(name,fn)=>{fn();count++;console.log('PASS '+name);};
check('eligible object retains support against stronger background and recovers known translation',()=>{
  const t=start();assert.equal(attach(t),true);const r=t.update({rgba:image(2),width:W,height:H,timestamp:200});
  assert.equal(r.valid,true,JSON.stringify(r));assert.ok(Math.abs(r.dx-2)<=1&&Math.abs(r.dy)<=1,JSON.stringify(r));assert.ok(r.diagnostics.maskedInitialFeatures>=8);
});
check('delayed mask retains eligible object motion independently of static background',()=>{
  const t=start();t.update({rgba:image(2),width:W,height:H,timestamp:200});t.update({rgba:image(4),width:W,height:H,timestamp:300});attach(t);const r=t.result();
  assert.equal(r.valid,true,JSON.stringify(r));assert.ok(Math.abs(r.dx-4)<=1&&Math.abs(r.dy)<=1,JSON.stringify(r));
});
check('object motion remains distinct from independently moving background',()=>{
  const t=start();t.update({rgba:image(2,'eligible-low-contrast',-2),width:W,height:H,timestamp:200});t.update({rgba:image(4,'eligible-low-contrast',-4),width:W,height:H,timestamp:300});attach(t);const r=t.result();
  assert.equal(r.valid,true,JSON.stringify(r));assert.ok(Math.abs(r.dx-4)<=1&&Math.abs(r.dy)<=1,JSON.stringify(r));
});
check('flat target cannot borrow surrounding background support',()=>{const t=start('flat-object');attach(t);assert.equal(t.result().valid,false);});
check('repeated target stripes cannot borrow surrounding background support',()=>{const t=start('stripes');attach(t);assert.equal(t.result().valid,false);});
check('feature cap, patch footprint and quality floors remain intact',()=>{const t=start();assert.ok(t.features.length<=192);for(const f of t.features){assert.ok(f.x0>=10&&f.y0>=10&&f.x0<W-10&&f.y0<H-10);assert.ok(f.score>=25);}for(let i=0;i<t.features.length;i++)for(let j=i+1;j<t.features.length;j++){const a=t.features[i],b=t.features[j];assert.ok((a.x0-b.x0)**2+(a.y0-b.y0)**2>=25);}});
check('selection is deterministic for identical inputs',()=>{assert.deepEqual(start().features,start().features);});
check('dense eligible texture fills four feature slots per normalized cell',()=>{const t=start('dense'),counts=new Uint16Array(48);for(const f of t.features)counts[Math.min(5,Math.floor(6*f.y0/H))*8+Math.min(7,Math.floor(8*f.x0/W))]++;assert.equal(t.features.length,192);assert.ok(Array.from(counts).every(n=>n===4),Array.from(counts).join(','));});
check('sparse texture backfills unused cell quotas without wasting eligible capacity',()=>{const t=start('sparse'),d=t.result().diagnostics;assert.ok(d.eligibleCandidates>0);assert.equal(t.features.length,Math.min(192,d.eligibleCandidates));assert.ok(d.backfillFeatures>0);assert.equal(d.quotaSelectedFeatures+d.backfillFeatures,t.features.length);});
check('wrong epoch does not admit an otherwise valid mask',()=>{const t=start(),before=JSON.stringify(t.result());assert.equal(t.setMask({mask:mask(),width:W,height:H,epoch:4,frameId:9}),false);assert.equal(JSON.stringify(t.result()),before);});
console.log(`${count} manufactured feature-coverage controls passed; physical surface accuracy remains unverified.`);
