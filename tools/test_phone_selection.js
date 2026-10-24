'use strict';
const assert=require('node:assert/strict');
const {Session,sameFrame,pointInCanvas,coverPointInCanvas,boundedSize}=require('../bench/web/design-studio/phone-selection.js');
// object-fit:cover scales 360 source pixels to 844 screen pixels; central crop
// therefore spans 390*360/844 source pixels around source x=320.
const portrait={left:0,top:0,width:390,height:844};
assert.deepEqual(coverPointInCanvas(195,422,portrait,640,360),{x:.5,y:.5});
assert.ok(Math.abs(coverPointInCanvas(0,422,portrait,640,360).x-(.5-195*360/(844*640)))<1e-12);
assert.equal(coverPointInCanvas(-1,422,portrait,640,360),null);
const live=new Session();live.permission(live.begin());const tapped=live.select({x:.5,y:.5},640,360,'0',10);assert.equal(live.phase,'live');const retapped=live.select({x:.2,y:.3},640,360,'0',20);assert.equal(live.accept(tapped,new Uint8Array(1)),false);assert.equal(live.accepts(retapped),true);
const s=new Session();const first=s.begin();assert.equal(s.permission(first),true);const frame=s.capture(640,480,'0:1280x960',10);assert.equal(s.accept(frame,new Uint8Array(4)),true);
assert.equal(s.lock({x:.5,y:.5}),true);assert.equal(s.accept(frame,new Uint8Array(4)),false,'live candidate cannot overwrite frozen revision');
const locked=s.tag();assert.equal(s.accept(locked,new Uint8Array(4)),true);s.correct('negative',[{x:.1,y:.2}]);assert.equal(s.accept(locked,new Uint8Array(4)),false,'old correction rejected');assert.equal(s.strokes.length,2);assert.equal(s.undo(),true);assert.equal(s.strokes.length,1);assert.equal(s.undo(),false,'initial seed retained');
const old=s.tag();const next=s.begin();assert.equal(s.permission(first),false,'late camera permission rejected');assert.equal(s.permission(next),true);assert.equal(s.accept(old,new Uint8Array(4)),false,'old session mask rejected');s.capture(480,640,'90:960x1280',20);s.lock({x:.5,y:.5},true);assert.equal(s.manual,true);s.correct('positive',[{x:.2,y:.2}]);s.clear();assert.equal(s.strokes.length,0,'manual reset clears all strokes');
for(const key of ['epoch','frameId','timestamp','width','height','orientation','revision'])assert.equal(sameFrame(old,{...old,[key]:key==='orientation'?'other':old[key]+1}),false,`${key} identity is required`);
assert.deepEqual(boundedSize(1920,1080),{width:640,height:360});assert.deepEqual(boundedSize(480,640),{width:480,height:640});assert.deepEqual(pointInCanvas(150,100,{left:50,top:50,width:200,height:100}),{x:.5,y:.5});assert.equal(pointInCanvas(49,100,{left:50,top:50,width:200,height:100}),null);s.reset();assert.equal(s.frame,null);assert.equal(s.mask,null);console.log('Phone selection state: all assertions passed.');

// Exercise request ownership and cancellation without a browser or model.
const {PhoneSegmenterClient}=require('../bench/web/design-studio/phone-camera.js');
const workers=[];
global.Worker=class {
  constructor(url){this.url=url;this.messages=[];this.terminated=false;workers.push(this);}
  postMessage(message){this.messages.push(message);}
  terminate(){this.terminated=true;}
  reply(value){this.onmessage({data:value});}
};
(async()=>{
  const client=new PhoneSegmenterClient();
  const cancelled=client.initialize(10);const oldWorker=workers.at(-1);const cancelledCheck=assert.rejects(cancelled,/cancelled/i);client.close();await cancelledCheck;assert.equal(oldWorker.terminated,true);
  const ready=client.initialize(11);const worker=workers.at(-1);const init=worker.messages.at(-1);
  worker.reply({id:init.id,epoch:10,type:'ready'});assert.equal(client.ready,false,'wrong session ready response ignored');
  oldWorker.reply({id:init.id-1,epoch:10,type:'ready'});assert.equal(client.ready,false,'late old worker response ignored');
  worker.reply({id:init.id,epoch:11,type:'ready'});await ready;assert.equal(client.ready,true);
  const tag={epoch:11,frameId:1,timestamp:1,width:2,height:2,orientation:'0',revision:0};
  const result=client.process({},tag,[{mode:'positive',points:[{x:.5,y:.5}]}]);await assert.rejects(client.process({},tag,[]),/already running/);assert.equal(worker.messages.filter(m=>m.type==='process').length,1,'busy request is never queued');
  const request=worker.messages.at(-1);worker.reply({id:request.id,epoch:11,type:'mask',tag,mask:new ArrayBuffer(4)});assert.equal((await result).tag,tag);
  const pending=client.process({},tag,[]);const pendingCheck=assert.rejects(pending,/cancelled/i);client.close();await pendingCheck;assert.equal(worker.terminated,true);assert.equal(client.ready,false);console.log('Phone worker client: all assertions passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
