'use strict';
// Manufactured translations are exact array-index shifts, independent of tracker code.
// These fixtures test software contracts, not camera/model accuracy or phone speed.
const assert=require('node:assert/strict');
const {Tracker}=require('../bench/web/design-studio/phone-tracker.js');
const W=160,H=120;
function noise(seed){const a=new Uint8Array(W*H);for(let i=0;i<a.length;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;a[i]=seed>>>24;}return a;}
const background=noise(54321),object=noise(12345),unrelated=noise(999);
function frame(dx=0,dy=0,{occlude=false,flat=false,stripes=false,scale=1,rotation=0}={}){
  const rgba=new Uint8ClampedArray(W*H*4),co=Math.cos(rotation),si=Math.sin(rotation);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const xx=x-dx-80,yy=y-dy-60,ox=Math.round((co*xx+si*yy)/scale+80),oy=Math.round((-si*xx+co*yy)/scale+60);
    let value=ox>=30&&ox<130&&oy>=20&&oy<100?object[oy*W+ox]:background[y*W+x];
    if(occlude&&x>=30+dx&&x<130+dx&&y>=20+dy&&y<100+dy)value=unrelated[y*W+x];
    if(flat)value=100;if(stripes)value=x%8<4?230:20;
    const i=(y*W+x)*4;rgba[i]=rgba[i+1]=rgba[i+2]=value;rgba[i+3]=255;
  }return rgba;
}
function mask(width=W,height=H){const out=new Uint8Array(width*height);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const rx=(x+.5)*W/width,ry=(y+.5)*H/height;if(rx>=30&&rx<130&&ry>=20&&ry<100)out[y*width+x]=255;}return out;}
function begin(rgba=frame()){const t=new Tracker();t.start({rgba,width:W,height:H,timestamp:100,epoch:1,frameId:7});return t;}
function advance(t,dx,dy,timestamp,options){return t.update({rgba:frame(dx,dy,options),width:W,height:H,timestamp});}
function attach(t,width=W,height=H){return t.setMask({mask:mask(width,height),width,height,epoch:1,frameId:7});}
let count=0;function test(name,fn){fn();count++;console.log(`PASS ${name}`);}
test('delayed mask follows moving object rather than static background',()=>{const t=begin();advance(t,2,1,200);advance(t,4,2,300);assert.equal(t.result().valid,false);assert.equal(attach(t),true);const r=t.result();assert.equal(r.valid,true,JSON.stringify(r));assert.ok(Math.abs(r.dx-4)<=1&&Math.abs(r.dy-2)<=1,JSON.stringify(r));});
test('immediate mask preserves displacement signs and cumulative reference',()=>{const t=begin();attach(t);advance(t,-2,1,200);advance(t,-4,2,300);const r=t.result();assert.equal(r.valid,true,JSON.stringify(r));assert.ok(Math.abs(r.dx+4)<=1&&Math.abs(r.dy-2)<=1);});
test('640px mask maps to tracking reference',()=>{const t=begin();assert.equal(attach(t,640,480),true);advance(t,3,-2,200);const r=t.result();assert.equal(r.valid,true,JSON.stringify(r));assert.ok(Math.abs(r.dx*4-12)<=4&&Math.abs(r.dy*4+8)<=4);});
test('occlusion and delayed reply cannot resurrect selection',()=>{const t=begin();attach(t);advance(t,0,0,200,{occlude:true});assert.equal(t.result().valid,false);advance(t,0,0,300);attach(t);assert.equal(t.result().valid,false);});
test('loss while awaiting segmentation remains terminal',()=>{const t=begin();advance(t,0,0,200,{flat:true});attach(t);advance(t,0,0,300);assert.equal(t.result().valid,false);});
test('flat and repeated stripe images cannot support attachment',()=>{for(const options of [{flat:true},{stripes:true}]){const t=begin(frame(0,0,options));attach(t);assert.equal(t.result().valid,false);}});
test('mask from wrong epoch or frame is rejected without changing state',()=>{const t=begin();const before=JSON.stringify(t.result());for(const changed of [{epoch:2},{frameId:8}])assert.equal(t.setMask({mask:mask(),width:W,height:H,epoch:1,frameId:7,...changed}),false);assert.equal(JSON.stringify(t.result()),before);attach(t);assert.equal(t.result().valid,true);});
test('stale and long-gap timestamps invalidate permanently',()=>{for(const timestamp of [100,99,851]){const t=begin();attach(t);advance(t,0,0,timestamp);assert.equal(t.result().valid,false);advance(t,0,0,900);assert.equal(t.result().valid,false);}});
test('dimension change invalidates selection',()=>{const t=begin();attach(t);t.update({rgba:new Uint8Array(100*100*4),width:100,height:100,timestamp:200});assert.equal(t.result().valid,false);});
test('unsupported large rotation and scale fail closed',()=>{for(const options of [{rotation:Math.PI/6},{scale:1.3}]){const t=begin();attach(t);advance(t,0,0,200,options);assert.equal(t.result().valid,false);}});
test('reset clears attachment and permits a fresh selection',()=>{const t=begin();attach(t);t.reset();assert.equal(t.result().valid,false);t.start({rgba:frame(),width:W,height:H,timestamp:500,epoch:2,frameId:9});assert.equal(t.result().valid,false);});
test('matching texture in one quadrant is insufficient spatial support',()=>{const rgba=frame(0,0,{flat:true}),texture=frame();for(let y=25;y<55;y++)for(let x=35;x<75;x++){const i=(y*W+x)*4;rgba.set(texture.subarray(i,i+4),i);}const t=begin(rgba);attach(t);assert.equal(t.result().valid,false);});
console.log(`${count} manufactured tracker controls passed.`);
