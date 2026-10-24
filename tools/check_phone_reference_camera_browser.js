// New human reference contract: compact chat composer, not the old shutter dock.
// Deterministic pixels/masks test software behavior, never physical phone accuracy.
async page=>{
  const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
  await page.addInitScript(()=>{
    window.__reference={freeze:false,flat:false,hold:false,tracks:[],requests:0};
    Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async()=>{throw new DOMException('Manufactured clipboard denial','NotAllowedError');}}});
    window.Worker=class{
      postMessage(d){if(d.type==='initialize'||d.type==='reset')return setTimeout(()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:d.type==='initialize'?'ready':'reset'}});},10);
        if(d.type==='process'){__reference.requests++;d.image?.close();const mask=new Uint8Array(d.tag.width*d.tag.height);for(let y=20;y<100;y++)for(let x=30;x<130;x++)mask[y*d.tag.width+x]=255;const reply=()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:'mask',tag:d.tag,mask:mask.buffer}});};if(__reference.hold)__reference.release=reply;else setTimeout(reply,250);}}
      terminate(){this.closed=true;}
    };
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:async()=>{
      const c=document.createElement('canvas');c.width=160;c.height=120;const ctx=c.getContext('2d'),p=ctx.createImageData(160,120);let seed=12345;
      for(let i=0;i<160*120;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const j=i*4;p.data[j]=p.data[j+1]=p.data[j+2]=seed>>>24;p.data[j+3]=255;}
      const draw=()=>{if(__reference.freeze)return;if(__reference.flat){ctx.fillStyle='#888';ctx.fillRect(0,0,160,120);}else ctx.putImageData(p,0,0);};draw();const media=c.captureStream(15),timer=setInterval(draw,70);
      for(const t of media.getTracks()){const stop=t.stop.bind(t);t.stop=()=>{clearInterval(timer);stop();};__reference.tracks.push(t);}return media;
    }});
  });
  const uploads=[];page.on('request',r=>{if(r.method()!=='GET')uploads.push(r.url());});
  await page.goto('http://127.0.0.1:8892/camera');await page.bringToFront();
  for(const [width,height] of [[320,568],[393,650],[390,844],[844,390]]){
    await page.setViewportSize({width,height});
    check(await page.locator('#phone-prompt').isVisible(),`Prompt composer is visible by default at ${width}x${height}`);
    check(await page.evaluate(()=>{const p=document.querySelector('#phone-prompt-form').getBoundingClientRect(),f=document.querySelector('#phone-prompt-form').getBoundingClientRect();return p.height<=88&&f.height<=72&&p.left>=0&&p.right<=innerWidth&&p.top>=0&&p.bottom<=innerHeight;}),`Default composer stays a compact capsule at ${width}x${height}`);
    check(!(await page.locator('#confirm-object').isVisible()),`No dead shutter before a candidate at ${width}x${height}`);
    check(await page.evaluate(()=>['import-design','generate-artwork'].every(id=>{const r=document.getElementById(id).getBoundingClientRect();return r.width>=44&&r.height>=44&&r.left>=0&&r.right<=innerWidth&&r.bottom<=innerHeight;})),`Import and send targets remain reachable at ${width}x${height}`);
  }
  await page.setViewportSize({width:393,height:650});await page.locator('#start').click();await page.waitForFunction(()=>PhoneCamera.diagnostics().runtime==='ready'&&PhoneCamera.diagnostics().renderMode==='synchronized');
  check(await page.evaluate(()=>__reference.requests===0),'No segmentation runs without a tap');
  await page.locator('#scene').click();await page.waitForFunction(()=>PhoneCamera.diagnostics().selection.outlineVisible);
  check(await page.locator('#confirm-object').isVisible()&&await page.locator('#confirm-object').isEnabled(),'Fresh outline exposes a contextual Use object control');
  check(await page.evaluate(()=>{const r=document.querySelector('#confirm-object').getBoundingClientRect();return r.height<=52;}),'Confirmation is compact rather than a large shutter');
  await page.locator('#confirm-object').click();await page.locator('#phone-prompt').fill('green and cream waves');await page.locator('#generate-artwork').click();
  check(await page.locator('#artwork-inspector').isVisible(),'Composer submission opens artwork inspection');await page.waitForTimeout(1100);
  check(await page.locator('#artwork-inspector').isVisible(),'Artwork persists after its bottom spin');
  await page.locator('#apply-preview').click();await page.locator('#close-preview').click();
  check(await page.locator('#phone-prompt').isVisible(),'X returns to the compact composer');
  await page.evaluate(()=>{__reference.freeze=true;});await page.waitForTimeout(400);
  check(await page.evaluate(()=>!PhoneCamera.diagnostics().selection.outlineVisible),'Loss still suppresses the live outline');
  check(await page.evaluate(()=>{const message=document.querySelector('#camera-status');const r=message.getBoundingClientRect();return message.textContent.length<=100&&r.height<=48;}),'Main tracking-loss message stays short instead of becoming a technical panel');
  check(await page.evaluate(()=>document.querySelector('#phone-prompt-form').getBoundingClientRect().height<=88),'Tracking failure cannot expand the composer into a large dock');
  await page.evaluate(()=>{__reference.freeze=false;__reference.hold=true;__reference.release=null;});await page.waitForTimeout(180);await page.locator('#scene').click();
  await page.waitForFunction(()=>typeof __reference.release==='function');await page.evaluate(()=>{__reference.flat=true;});
  await page.waitForFunction(()=>PhoneCamera.diagnostics().failureSnapshot?.reason==='insufficient-survivors');
  const frozen=await page.evaluate(()=>JSON.stringify(PhoneCamera.diagnostics().failureSnapshot));
  check(await page.evaluate(()=>PhoneCamera.diagnostics().failureSnapshot.segmentation.state==='pending'),'Failure evidence is captured before the delayed mask arrives');
  await page.evaluate(()=>{__reference.hold=false;__reference.release();});await page.waitForTimeout(350);
  check(await page.evaluate(saved=>JSON.stringify(PhoneCamera.diagnostics().failureSnapshot)===saved,frozen),'Later frames and masks cannot overwrite the first failure snapshot');
  check(await page.evaluate(()=>{const d=PhoneCamera.diagnostics();return d.trace.length<=32&&d.failureSnapshot.trace.length<=32&&JSON.stringify(d).length<65536;}),'Numeric diagnostic history and payload remain bounded');
  await page.locator('#camera-menu-toggle').click();await page.locator('#copy-diagnostics').click();
  check(/copy blocked/i.test(await page.locator('#copy-status').innerText()),'Clipboard denial offers manual copy instead of failing silently');
  check(await page.evaluate(()=>{const text=document.querySelector('#camera-diagnostics').textContent;const d=JSON.parse(text);return !!d.timings&&!!d.trackerReason;}),'Detailed failure and timing evidence remains in diagnostics');
  await page.evaluate(()=>PhoneCamera.close());check(await page.evaluate(()=>__reference.tracks.every(t=>t.readyState==='ended')),'Close releases the camera');
  check(uploads.length===0,'No camera frames or diagnostics are uploaded');
  return {checks:checks.length,names:checks,manufactured:true,physicalPhoneVerified:false};
}
