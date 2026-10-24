// Manufactured frames and failures: software regression, not physical iPhone proof.
async page=>{
  const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
  await page.addInitScript(()=>{
    window.__iphoneDebug={starved:true,drawFailure:false,ticks:0,tracks:[]};
    const request=HTMLVideoElement.prototype.requestVideoFrameCallback;
    HTMLVideoElement.prototype.requestVideoFrameCallback=function(callback){
      const video=this;const forward=(now,meta)=>{
        if(__iphoneDebug.starved){request.call(video,forward);return;}
        callback(now,meta);
      };return request.call(video,forward);
    };
    HTMLVideoElement.prototype.getVideoPlaybackQuality=()=>({totalVideoFrames:0,droppedVideoFrames:0});
    const draw=CanvasRenderingContext2D.prototype.drawImage;
    CanvasRenderingContext2D.prototype.drawImage=function(image,...args){
      if(image instanceof HTMLVideoElement&&__iphoneDebug.drawFailure)throw new DOMException('Manufactured camera copy failure','InvalidStateError');
      return draw.call(this,image,...args);
    };
    window.Worker=class{
      postMessage(d){if(d.type==='initialize'||d.type==='reset')setTimeout(()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:d.type==='initialize'?'ready':'reset'}});},10);
        else if(d.type==='process'){d.image?.close();setTimeout(()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:'mask',tag:d.tag,mask:new Uint8Array(d.tag.width*d.tag.height).buffer}});},200);}}
      terminate(){this.closed=true;}
    };
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:async()=>{
      const c=document.createElement('canvas');c.width=160;c.height=120;const ctx=c.getContext('2d');
      const paint=()=>{__iphoneDebug.ticks++;ctx.fillStyle=__iphoneDebug.ticks%2?'#398754':'#7954a1';ctx.fillRect(0,0,160,120);ctx.fillStyle='#fff';ctx.fillRect(__iphoneDebug.ticks%100,30,15,15);};paint();
      const media=c.captureStream(15),timer=setInterval(paint,70);for(const t of media.getTracks()){const stop=t.stop.bind(t);t.stop=()=>{clearInterval(timer);stop();};__iphoneDebug.tracks.push(t);}return media;
    }});
  });
  const outbound=[];page.on('request',r=>{if(r.method()!=='GET')outbound.push(r.url());});
  await page.goto('http://127.0.0.1:8892/camera');await page.setViewportSize({width:393,height:650});
  check(/open.*camera.*begin/i.test(await page.locator('#camera-status').innerText()),'Initial artwork does not overwrite the camera instruction');
  await page.locator('#start').click();await page.waitForTimeout(1900);
  check(await page.evaluate(()=>{const v=document.querySelector('#camera'),s=getComputedStyle(v);return !v.hidden&&s.display!=='none'&&s.visibility!=='hidden'&&Number(s.opacity)>0;}),'Camera video remains renderable for Safari');
  check(await page.evaluate(()=>{const c=document.querySelector('#scene');return c.hidden||getComputedStyle(c).display==='none'||getComputedStyle(c).visibility==='hidden';}),'Startup callback starvation shows raw video without a stale canvas overlay');
  check(await page.locator('#confirm-object').isDisabled()&&await page.locator('#apply-preview').isDisabled(),'Unverified startup frames cannot confirm or apply artwork');
  check(/preview|frame|unavailable|waiting/i.test(await page.locator('#camera-status').innerText()),'Startup starvation gives an actionable visible state');
  const ticks=await page.evaluate(()=>__iphoneDebug.ticks);await page.waitForTimeout(200);
  check(await page.evaluate(()=>__iphoneDebug.ticks)>ticks,'Raw camera fixture continues during callback starvation');
  await page.evaluate(()=>{__iphoneDebug.starved=false;});await page.waitForTimeout(500);
  check(await page.locator('#scene').isVisible(),'Verified decoded frames restore synchronized composition');
  check(await page.evaluate(()=>{const c=document.querySelector('#scene'),p=c.getContext('2d').getImageData(2,2,1,1).data;return p[3]>0&&(p[0]+p[1]+p[2])>50;}),'Recovered synchronized canvas contains camera pixels');
  await page.evaluate(()=>{__iphoneDebug.starved=true;});await page.waitForTimeout(110);
  const beforeGeometry=await page.evaluate(()=>{const c=document.querySelector('#scene');return {width:c.width,height:c.height,pixel:Array.from(c.getContext('2d').getImageData(2,2,1,1).data)};});
  await page.evaluate(()=>{const v=document.querySelector('#camera');Object.defineProperty(v,'videoWidth',{configurable:true,value:240});Object.defineProperty(v,'videoHeight',{configurable:true,value:160});});await page.waitForTimeout(80);
  check(await page.evaluate(b=>{const c=document.querySelector('#scene');return c.width===b.width&&c.height===b.height&&Array.from(c.getContext('2d').getImageData(2,2,1,1).data).join()===b.pixel.join();},beforeGeometry),'Rejected duplicate frame with changed geometry does not erase or resize accepted pixels');
  await page.evaluate(()=>{__iphoneDebug.starved=false;});await page.waitForTimeout(300);
  check(await page.evaluate(()=>document.querySelector('#scene').width===240&&PhoneCamera.diagnostics().geometry.endsWith('240x160')),'Fresh frame commits new geometry together with its pixels');
  await page.evaluate(()=>{const v=document.querySelector('#camera');delete v.videoWidth;delete v.videoHeight;});await page.waitForTimeout(200);
  await page.evaluate(()=>{__iphoneDebug.drawFailure=true;});await page.waitForTimeout(600);
  check(await page.evaluate(()=>document.querySelector('#scene').hidden||getComputedStyle(document.querySelector('#scene')).display==='none'),'Drawing failures suppress attachment canvas and expose raw video');
  check(await page.locator('#confirm-object').isDisabled(),'Drawing failure cannot confirm stale results');
  await page.evaluate(()=>{__iphoneDebug.drawFailure=false;});await page.waitForTimeout(400);
  check(await page.locator('#scene').isVisible(),'A successful fresh frame recovers after drawing failure');
  for(const viewport of [{width:320,height:568},{width:393,height:650},{width:390,height:844},{width:844,height:390}]){
    await page.setViewportSize(viewport);
    await page.evaluate(()=>{document.querySelector('#camera-status').textContent='Object moved out of tracking. Tap it again. Your camera preview remains live.';document.querySelector('#show-design').hidden=false;});
    check(await page.evaluate(()=>{const a=document.querySelector('#camera-status').getBoundingClientRect(),b=document.querySelector('#show-design').getBoundingClientRect();return !(a.left<b.right&&a.right>b.left&&a.top<b.bottom&&a.bottom>b.top);}),`Status and artwork control do not overlap at ${viewport.width}x${viewport.height}`);
  }
  await page.locator('#camera-menu-toggle').click();await page.locator('#stop').click();
  check(await page.evaluate(()=>__iphoneDebug.tracks.every(t=>t.readyState==='ended')),'Stop releases all camera tracks');
  await page.evaluate(()=>{__iphoneDebug.drawFailure=true;});await page.locator('#start').click();await page.waitForTimeout(400);
  check(await page.evaluate(()=>PhoneCamera.diagnostics().phase==='live'&&PhoneCamera.diagnostics().renderMode==='raw'&&__iphoneDebug.tracks.at(-1).readyState==='live'),'Initial camera-copy failure preserves the live raw stream');
  check(await page.evaluate(()=>PhoneCamera.diagnostics().captureError?.name==='InvalidStateError'),'Camera copy exception is inspectable without images');
  await page.evaluate(()=>{__iphoneDebug.drawFailure=false;});await page.waitForTimeout(400);
  check(await page.locator('#scene').isVisible(),'Initial camera-copy failure recovers on a fresh frame');
  await page.evaluate(()=>PhoneCamera.close());
  await page.evaluate(()=>{__iphoneDebug.starved=true;__iphoneDebug.originalPlay=HTMLVideoElement.prototype.play;HTMLVideoElement.prototype.play=function(){__iphoneDebug.originalPlay.call(this).catch(()=>{});return new Promise((resolve,reject)=>{__iphoneDebug.finishPlay=resolve;__iphoneDebug.rejectPlay=reject;});};__iphoneDebug.start=PhoneCamera.initialize();});await page.waitForTimeout(1900);
  check(await page.evaluate(()=>PhoneCamera.diagnostics().phase==='live'&&PhoneCamera.diagnostics().renderMode==='raw'),'Unsettled play promise does not prevent the first-frame watchdog');
  check(await page.evaluate(()=>{
    const warning=document.querySelector('#selection-issue')||document.querySelector('#camera-status');
    const reason=PhoneCamera.diagnostics().issue?.reason;
    return warning.getClientRects().length>0&&(/selection (?:is )?(?:unavailable|paused)|verifi(?:ed|able).*frames/i.test(warning.textContent)||(reason==='no-verified-frames'&&/paused|unavailable/i.test(warning.textContent)));
  }),'Unsettled play promise has an explicit selection-unavailable message');
  await page.evaluate(()=>{__iphoneDebug.starved=false;});await page.waitForTimeout(500);
  check(await page.evaluate(async()=>{const epoch=PhoneCamera.selection.epoch;await PhoneCamera.select({x:.5,y:.5});return PhoneCamera.selection.epoch>epoch;}),'Recovered frames permit a selection while play completion is deferred');
  await page.evaluate(async()=>{__iphoneDebug.finishPlay();await __iphoneDebug.start;});
  check(await page.evaluate(()=>PhoneCamera.diagnostics().phase==='live'&&__iphoneDebug.tracks.at(-1).readyState==='live'),'Late play completion cannot stop the current camera after selection changes its epoch');
  await page.evaluate(()=>PhoneCamera.close());
  check(await page.evaluate(()=>__iphoneDebug.tracks.every(t=>t.readyState==='ended')),'Stop also releases tracks while play is pending');
  await page.evaluate(()=>{__iphoneDebug.start=PhoneCamera.initialize();});await page.waitForTimeout(400);
  await page.evaluate(async()=>{await PhoneCamera.select({x:.5,y:.5});__iphoneDebug.rejectPlay(new DOMException('Manufactured playback rejection','NotSupportedError'));await __iphoneDebug.start;});
  check(await page.evaluate(()=>PhoneCamera.diagnostics().phase==='error'&&__iphoneDebug.tracks.every(t=>t.readyState==='ended')),'Late play rejection cleans up the owned camera despite a changed selection epoch');
  await page.evaluate(()=>{HTMLVideoElement.prototype.play=__iphoneDebug.originalPlay;});
  check(outbound.length===0,'No camera frames, masks or diagnostics are uploaded');
  return {checks:checks.length,names:checks,physicalIPhoneVerified:false};
}
