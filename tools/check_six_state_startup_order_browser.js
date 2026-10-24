// Frame/worker ordering controls for the bounded startup-copy repair. No physical camera.
async original=>{
 const checks=[],check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
 async function fixture({frames=false,workerDelay=10,workerError=false}={}){
  const page=await original.context().newPage();await page.bringToFront();
  await page.addInitScript(({frames,workerDelay,workerError})=>{
   window.__startup={frames,tracks:[]};
   const request=HTMLVideoElement.prototype.requestVideoFrameCallback;
   HTMLVideoElement.prototype.requestVideoFrameCallback=function(cb){const video=this;const forward=(now,meta)=>{if(!__startup.frames){request.call(video,forward);return;}cb(now,meta);};return request.call(video,forward);};
   HTMLVideoElement.prototype.getVideoPlaybackQuality=()=>({totalVideoFrames:0});
   window.Worker=class{postMessage(d){if(d.type==='initialize')setTimeout(()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:workerError?'error':'ready',error:workerError?'Manufactured worker failure':undefined}});},workerDelay);}terminate(){this.closed=true;}};
   Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:async()=>{const c=document.createElement('canvas');c.width=160;c.height=120;const x=c.getContext('2d');let n=0;const paint=()=>{x.fillStyle=n++%2?'#a6a0b0':'#94939b';x.fillRect(0,0,160,120);};paint();const media=c.captureStream(15),timer=setInterval(paint,50);for(const t of media.getTracks()){const stop=t.stop.bind(t);t.stop=()=>{clearInterval(timer);stop();};__startup.tracks.push(t);}return media;}});
  },{frames,workerDelay,workerError});
  await page.goto('http://127.0.0.1:8892/camera');await page.locator('#start').click();return page;
 }
 const first=await fixture();await first.waitForFunction(()=>PhoneCamera.diagnostics().runtime==='ready'&&PhoneCamera.diagnostics().acceptedFrames===0);
 check(/waiting for camera frames/i.test(await first.locator('#camera-status').textContent()),'Worker-first startup stays waiting before verified pixels');
 await first.evaluate(()=>__startup.frames=true);await first.waitForFunction(()=>PhoneCamera.diagnostics().renderMode==='synchronized');await first.waitForFunction(()=>/Hold Analyze/.test(document.querySelector('#camera-status').textContent));
 check(true,'First verified pixels replace waiting status without watchdog timeout');
 await first.evaluate(()=>__startup.frames=false);await first.waitForFunction(()=>PhoneCamera.diagnostics().issue?.kind==='frames');const loss=await first.evaluate(()=>PhoneCamera.diagnostics().issue.reason);
 await first.evaluate(()=>__startup.frames=true);await first.waitForFunction(()=>PhoneCamera.diagnostics().renderMode==='synchronized');
 check(await first.evaluate(reason=>PhoneCamera.diagnostics().issue?.reason===reason,loss),'Resuming frames cannot clear existing failure evidence');await first.evaluate(()=>PhoneCamera.close());await first.close();
 const second=await fixture({frames:true,workerDelay:350});await second.waitForFunction(()=>PhoneCamera.diagnostics().runtime==='ready'&&PhoneCamera.diagnostics().renderMode==='synchronized');
 check(/Hold Analyze/.test(await second.locator('#camera-status').textContent()),'Frames-first startup gets ready hint after worker initialization');await second.evaluate(()=>PhoneCamera.close());await second.close();
 const failed=await fixture({workerError:true});await failed.waitForFunction(()=>PhoneCamera.diagnostics().runtime==='unavailable');const reason=await failed.evaluate(()=>PhoneCamera.diagnostics().issue?.reason);
 await failed.evaluate(()=>__startup.frames=true);await failed.waitForFunction(()=>PhoneCamera.diagnostics().renderMode==='synchronized');
 check(await failed.evaluate(r=>PhoneCamera.diagnostics().issue?.reason===r,reason)&&/unavailable/i.test(await failed.locator('#camera-status').textContent()),'Fresh pixels preserve worker-failure message and unavailable selection');await failed.evaluate(()=>PhoneCamera.close());await failed.close();
 return {checks:checks.length,names:checks,physicalPhoneVerified:false};
}
