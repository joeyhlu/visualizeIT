// Software-only UI checks. Manufactured camera and segmentation, actual tracker.
async page => {
  const checks=[];const check=(ok,label)=>{if(!ok)throw Error(label);checks.push(label);};
  await page.addInitScript(()=>{
    window.__liveTest={tracks:[],requests:[],mode:'normal',processes:0,flat:false};
    class FixtureWorker {
      postMessage(d){
        if(d.type==='initialize'||d.type==='reset')return setTimeout(()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:d.type==='initialize'?'ready':'reset'}});},10);
        if(d.type==='process'){
          __liveTest.processes++;d.image?.close();const mask=new Uint8Array(d.tag.width*d.tag.height);
          for(let y=20;y<100;y++)for(let x=30;x<130;x++)mask[y*d.tag.width+x]=255;
          setTimeout(()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:'mask',tag:d.tag,mask:mask.buffer}});},650);
        }
      }
      terminate(){this.closed=true;}
    }
    window.Worker=FixtureWorker;
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:async constraints=>{
      __liveTest.requests.push(constraints);
      if(__liveTest.mode==='denied')throw new DOMException('Manufactured denial','NotAllowedError');
      const c=document.createElement('canvas');c.width=160;c.height=120;const ctx=c.getContext('2d'),base=ctx.createImageData(160,120);
      let seed=12345;for(let i=0;i<160*120;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const v=seed>>>24;base.data[i*4]=base.data[i*4+1]=base.data[i*4+2]=v;base.data[i*4+3]=255;}
      const texture=document.createElement('canvas');texture.width=160;texture.height=120;texture.getContext('2d').putImageData(base,0,0);
      let tick=0;const draw=()=>{if(__liveTest.freeze)return;const x=Math.abs((tick++%16)-8);ctx.fillStyle='#777';ctx.fillRect(0,0,160,120);if(!__liveTest.flat)ctx.drawImage(texture,x,0);};draw();
      const stream=c.captureStream(12),timer=setInterval(draw,100);__liveTest.tracks.push(...stream.getTracks());
      for(const track of stream.getTracks()){const stop=track.stop.bind(track);track.stop=()=>{clearInterval(timer);stop();};}
      if(__liveTest.mode==='delayed')await new Promise(resolve=>__liveTest.release=resolve);
      return stream;
    }});
  });
  const requests=[];page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push({url:r.url(),method:r.method()});});
  await page.goto('http://127.0.0.1:8892/');
  await page.locator('#prompt').fill('navy and cream waves text CAMERA');await page.locator('#prompt-form button').click();
  const artwork=await page.locator('#artwork').evaluate(c=>c.toDataURL());
  await page.locator('#open-camera').click();await page.waitForURL('**/camera');
  check(await page.locator('#artwork-preview').evaluate(c=>c.toDataURL())===artwork,'Exact studio artwork survives handoff');
  for(const width of [320,390,1280]){await page.setViewportSize({width,height:844});check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`No horizontal overflow at ${width}px`);}
  await page.setViewportSize({width:390,height:844});await page.locator('#start').click();
  await page.waitForFunction(()=>PhoneCamera.selection.phase==='live');
  await page.waitForTimeout(350);
  check(await page.evaluate(()=>__liveTest.processes===0),'No automatic segmentation before tap');
  const before=await page.locator('#scene').evaluate(c=>c.toDataURL());
  await page.locator('#scene').click({position:{x:180,y:250}});
  await page.waitForFunction(()=>__liveTest.processes===1);
  check(await page.evaluate(()=>__liveTest.tracks.some(t=>t.readyState==='live')),'Tap keeps camera tracks running');
  await page.waitForTimeout(300);
  check(await page.locator('#scene').evaluate(c=>c.toDataURL())!==before,'Camera continues changing while segmentation is pending');
  await page.waitForFunction(()=>PhoneCamera.selection.mask!==null);
  await page.waitForTimeout(150);
  check(await page.locator('#placement').isEnabled(),'Verified moving mask enables artwork placement');
  const appliedBefore=await page.evaluate(()=>PhoneCamera.previewState().appliedRevision);
  await page.locator('#phone-prompt').fill('pink and navy dots text LIVE');await page.locator('#phone-prompt-form button[type=submit]').click();
  check(await page.locator('#artwork-preview').evaluate(c=>c.toDataURL())!==artwork,'Prompt generates new local artwork while camera runs');
  check(await page.locator('#artwork-reveal').isVisible(),'Generated artwork opens persistent inspector');
  await page.waitForTimeout(1100);
  check(await page.locator('#artwork-inspector').isVisible(),'Inspector remains open after entrance animation');
  check(await page.evaluate(()=>PhoneCamera.previewState().appliedRevision)===appliedBefore,'Generation does not implicitly apply pending design');
  const pending=await page.evaluate(()=>PhoneCamera.previewState().pendingRevision);
  await page.locator('#apply-preview').click();
  check(await page.evaluate(()=>PhoneCamera.previewState().appliedRevision)===pending,'Apply commits selected pending design');
  check(await page.locator('#artwork-inspector').isVisible(),'Apply retains interactive preview');
  await page.emulateMedia({reducedMotion:'reduce'});await page.locator('#phone-prompt-form button[type=submit]').click();
  check(await page.locator('#artwork-inspector').isVisible(),'Reduced motion keeps persistent preview available');
  await page.emulateMedia({reducedMotion:'no-preference'});
  await page.evaluate(()=>__liveTest.freeze=true);
  await page.waitForFunction(()=>document.querySelector('#placement').disabled);
  check(await page.evaluate(()=>__liveTest.tracks.some(t=>t.readyState==='live')),'Decoded-frame stall hides attachment despite live camera track');
  check(await page.locator('#artwork-inspector').isVisible(),'Tracking stall preserves design inspector');
  await page.locator('#close-preview').click();
  check(await page.locator('#artwork-inspector').isHidden(),'Close dismisses preview explicitly');
  await page.evaluate(()=>__liveTest.freeze=false);
  await page.waitForTimeout(250);await page.locator('#scene').click({position:{x:180,y:250}});
  await page.waitForFunction(()=>__liveTest.processes===2&&PhoneCamera.selection.mask!==null&&!document.querySelector('#placement').disabled);
  check(true,'A new tap restores attachment after stalled-frame loss');
  await page.evaluate(()=>__liveTest.flat=true);
  await page.waitForFunction(()=>document.querySelector('#placement').disabled);
  check(true,'Lost texture suppresses attachment');
  check(requests.every(r=>r.method==='GET'),'No camera frames or masks uploaded');
  check(await page.evaluate(()=>__liveTest.requests.every(r=>r.audio===false)),'No microphone requested');
  await page.locator('#generate-artwork').click();
  check(await page.locator('#artwork-inspector').isVisible(),'Inspector remains usable after loss');
  await page.locator('#stop').click();
  check(await page.evaluate(()=>__liveTest.tracks.every(t=>t.readyState==='ended')&&PhoneCamera.selection.mask===null),'Stop releases camera and mask');
  check(await page.evaluate(()=>{const p=PhoneCamera.previewState();return !p.open&&p.scale===1&&p.pendingRevision===p.appliedRevision;}),'Stop closes inspector and discards pending artwork');
  await page.evaluate(()=>__liveTest.mode='denied');await page.locator('#start').click();
  await page.waitForFunction(()=>PhoneCamera.selection.phase==='error');check(true,'Denied permission returns recoverable error');
  await page.evaluate(()=>__liveTest.mode='delayed');await page.locator('#start').click();
  await page.waitForFunction(()=>typeof __liveTest.release==='function');await page.locator('#stop').click();await page.evaluate(()=>__liveTest.release());
  await page.waitForFunction(()=>__liveTest.tracks.every(t=>t.readyState==='ended'));
  check(await page.evaluate(()=>PhoneCamera.selection.phase==='idle'),'Late camera permission cannot restart cancelled session');
  return {schema:'visualizeit_live_camera_ui_checks_v1',timestamp:new Date().toISOString(),checks,manufactured_camera_and_masks:true,physical_phone_verified:false};
}
