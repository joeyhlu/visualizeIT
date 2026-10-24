// Deterministic masks and camera pixels isolate UI/identity behavior, not ML quality.
async page=>{
  const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
  await page.addInitScript(()=>{
    window.__outline={mode:'mask',freeze:false,hold:true,tracks:[]};
    window.Worker=class{
      postMessage(d){if(d.type==='initialize'||d.type==='reset')return setTimeout(()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:d.type==='initialize'?'ready':'reset'}});},10);
        if(d.type==='process'){d.image?.close();const mask=new Uint8Array(d.tag.width*d.tag.height);if(__outline.mode!=='empty')for(let y=20;y<100;y++)for(let x=30;x<130;x++)mask[y*d.tag.width+x]=255;const reply=()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:'mask',tag:d.tag,mask:mask.buffer}});};if(__outline.hold)__outline.release=reply;else setTimeout(reply,250);}}
      terminate(){this.closed=true;}
    };
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:async()=>{
      const c=document.createElement('canvas');c.width=160;c.height=120;const ctx=c.getContext('2d'),pixels=ctx.createImageData(160,120);let seed=12345;
      for(let i=0;i<160*120;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const j=i*4;pixels.data[j]=pixels.data[j+1]=pixels.data[j+2]=seed>>>24;pixels.data[j+3]=255;}
      const draw=()=>{if(!__outline.freeze)ctx.putImageData(pixels,0,0);};draw();const media=c.captureStream(15),timer=setInterval(draw,70);
      for(const t of media.getTracks()){const stop=t.stop.bind(t);t.stop=()=>{clearInterval(timer);stop();};__outline.tracks.push(t);}return media;
    }});
  });
  const outbound=[];page.on('request',r=>{if(r.method()!=='GET')outbound.push(r.url());});
  await page.goto('http://127.0.0.1:8892/camera');await page.bringToFront();await page.setViewportSize({width:393,height:650});
  await page.locator('#start').click();await page.waitForFunction(()=>PhoneCamera.diagnostics().runtime==='ready'&&PhoneCamera.diagnostics().renderMode==='synchronized');
  await page.locator('#scene').click();
  check(await page.locator('#reticle .reticle-ring').isVisible(),'A tap is acknowledged immediately before segmentation finishes');
  check(/(tap|analyze) point/i.test(await page.locator('#reticle').innerText()),'Tap marker is labeled separately from an object outline');
  await page.waitForFunction(()=>typeof __outline.release==='function');await page.evaluate(()=>{__outline.hold=false;__outline.release();});
  await page.waitForFunction(()=>PhoneCamera.diagnostics().selection.outlineVisible);
  check(await page.locator('#confirm-object').isEnabled(),'Valid observed outline enables confirmation');
  check(await page.evaluate(()=>{const p=document.querySelector('#scene').getContext('2d').getImageData(30,50,1,1).data;return p[0]===176&&p[1]===248&&p[2]===214;}),'Candidate has the high-contrast mint observed boundary');
  await page.locator('#confirm-object').click();await page.locator('#design-button').click();await page.locator('#choose-generate').click();await page.locator('#phone-prompt').fill('blue and cream waves');await page.locator('#generate-artwork').click();await page.locator('#view-options-toggle').click();await page.locator('#apply-preview').click();await page.locator('#close-preview').click();
  check(await page.evaluate(()=>{const p=document.querySelector('#scene').getContext('2d').getImageData(30,50,1,1).data;return p[0]===176&&p[1]===248&&p[2]===214;}),'Observed boundary remains visible after artwork is applied');
  await page.evaluate(()=>{__outline.mode='empty';});await page.locator('#scene').click();
  await page.waitForFunction(()=>PhoneCamera.diagnostics().selection.segmentation==='rejected');
  check(await page.evaluate(()=>PhoneCamera.diagnostics().selection.maskPixels===0&&!PhoneCamera.diagnostics().selection.outlineVisible),'Empty mask cannot become a found outline');
  check(await page.evaluate(()=>!!PhoneCamera.diagnostics().issue)&&await page.locator('#camera-status').isVisible()&&await page.locator('#confirm-object').isDisabled(),'Empty mask presents a persistent actionable failure');
  await page.evaluate(()=>{__outline.mode='mask';window.__setMask=PhoneCamera.tracker.setMask;PhoneCamera.tracker.setMask=()=>false;});await page.locator('#scene').click();
  await page.waitForFunction(()=>PhoneCamera.diagnostics().selection.segmentation==='rejected');
  check(await page.evaluate(()=>PhoneCamera.diagnostics().selection.segmentationReason==='mask-not-registered'),'Rejected mask registration is separate from successful segmentation');
  await page.evaluate(()=>{PhoneCamera.tracker.setMask=__setMask;});await page.locator('#scene').click();await page.waitForFunction(()=>PhoneCamera.diagnostics().selection.outlineVisible);
  await page.evaluate(()=>{__outline.freeze=true;});await page.waitForTimeout(400);
  check(await page.locator('#confirm-object').isDisabled()&&await page.evaluate(()=>!PhoneCamera.diagnostics().selection.outlineVisible),'Stalled frames suppress both outline and attachment');
  await page.evaluate(()=>{__outline.freeze=false;});await page.waitForTimeout(400);
  check(await page.evaluate(()=>!!PhoneCamera.diagnostics().issue)&&await page.locator('#camera-status').isVisible(),'Frame recovery preserves the reason selection was lost');
  check(await page.locator('#confirm-object').isDisabled(),'Frame recovery cannot silently resurrect a lost selection');
  await page.evaluate(()=>{__outline.hold=true;__outline.release=null;});await page.locator('#scene').click();
  await page.waitForFunction(()=>typeof __outline.release==='function');
  await page.evaluate(()=>{__outline.freeze=true;});await page.waitForTimeout(320);
  await page.evaluate(()=>{__outline.freeze=false;});await page.waitForTimeout(150);
  await page.evaluate(()=>{__outline.hold=false;__outline.release();});await page.waitForFunction(()=>PhoneCamera.diagnostics().selection.segmentation!=='pending');
  check(!/outline found/i.test(await page.locator('#camera-status').innerText()),'Late mask cannot announce found after frame loss during segmentation');
  check(await page.locator('#confirm-object').isDisabled()&&await page.evaluate(()=>!!PhoneCamera.diagnostics().issue)&&await page.locator('#camera-status').isVisible(),'Late mask preserves the loss explanation and requires a new tap');
  await page.evaluate(()=>PhoneCamera.close());check(await page.evaluate(()=>__outline.tracks.every(t=>t.readyState==='ended')),'Camera closes all test tracks');
  check(outbound.length===0,'Camera frames and local diagnostics are never uploaded');
  return {checks:checks.length,names:checks,manufactured:true,physicalPhoneVerified:false};
}
