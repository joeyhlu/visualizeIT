// Manufactured camera/mask fixtures; actual browser UI and translation tracker.
// Passing checks are software evidence, not real segmentation/device accuracy.
async page=>{
  // Video compositor callbacks may be suspended in an occluded browser window.
  await page.bringToFront();
  const checks=[];function check(ok,name){if(!ok)throw Error(name);checks.push(name);}
  await page.addInitScript(()=>{
    window.__glass={tracks:[],requests:[],count:0,freeze:false,flat:false};
    window.Worker=class{
      postMessage(d){if(['initialize','reset'].includes(d.type))return setTimeout(()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:d.type==='initialize'?'ready':'reset'}});},15);
        if(d.type==='process'){__glass.count++;d.image?.close();const mask=new Uint8Array(d.tag.width*d.tag.height);for(let y=20;y<100;y++)for(let x=30;x<130;x++)mask[y*d.tag.width+x]=255;setTimeout(()=>{if(!this.closed)this.onmessage?.({data:{id:d.id,epoch:d.epoch,type:'mask',tag:d.tag,mask:mask.buffer}});},350);}}
      terminate(){this.closed=true;}
    };
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:async c=>{
      __glass.requests.push(c);const canvas=document.createElement('canvas');canvas.width=160;canvas.height=120;const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(160,120);let seed=12345;
      for(let i=0;i<160*120;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;pixels.data[i*4]=pixels.data[i*4+1]=pixels.data[i*4+2]=seed>>>24;pixels.data[i*4+3]=255;}
      const texture=document.createElement('canvas');texture.width=160;texture.height=120;texture.getContext('2d').putImageData(pixels,0,0);let tick=0;
      const draw=()=>{if(__glass.freeze)return;ctx.fillStyle='#777';ctx.fillRect(0,0,160,120);if(!__glass.flat)ctx.drawImage(texture,Math.abs((tick++%16)-8),0);};draw();
      const stream=canvas.captureStream(12),timer=setInterval(draw,100);__glass.tracks.push(...stream.getTracks());for(const t of stream.getTracks()){const stop=t.stop.bind(t);t.stop=()=>{clearInterval(timer);stop();};}return stream;
    }});
  });
  const requests=[];page.on('request',r=>requests.push({method:r.method(),url:r.url()}));
  await page.goto('http://127.0.0.1:8892/camera');await page.setViewportSize({width:390,height:844});
  check(await page.locator('#ask-toggle').isVisible()&&await page.locator('#confirm-object').isVisible()&&await page.locator('#import-design').isVisible(),'Camera has Ask, Confirm and Import controls');
  check(await page.locator('#confirm-object').isDisabled(),'No confirmation without a candidate');
  await page.locator('#start').click();await page.waitForFunction(()=>PhoneCamera.selection.phase==='live');await page.waitForTimeout(250);
  check(await page.evaluate(()=>__glass.count===0),'No automatic segmentation before object tap');
  await page.locator('#scene').click({position:{x:195,y:320}});
  await page.waitForFunction(()=>PhoneCamera.previewState().candidate&&!document.querySelector('#confirm-object').disabled);
  check(await page.evaluate(()=>!PhoneCamera.previewState().confirmed),'Mask remains an unconfirmed candidate');
  check(await page.evaluate(()=>__glass.tracks.some(t=>t.readyState==='live')),'Candidate selection keeps camera running');
  await page.locator('#confirm-object').click();
  check(await page.evaluate(()=>PhoneCamera.previewState().confirmed),'Center control confirms candidate');
  check(await page.locator('#ask-panel').isVisible(),'Confirmation opens Ask');
  await page.locator('#phone-prompt').fill('pink and navy dots text GLASS');
  const oldApplied=await page.evaluate(()=>PhoneCamera.previewState().appliedRevision);
  await page.locator('#generate-artwork').click();check(await page.locator('#artwork-inspector').isVisible(),'Generation opens full-screen inspector');
  const geometry=await page.locator('#artwork-inspector').boundingBox();check(geometry.width>=388&&geometry.height>=840,'Inspector fills phone viewport');
  check(await page.locator('#artwork-inspector').evaluate(e=>{const s=getComputedStyle(e);return s.backdropFilter==='blur(2px)'&&s.backgroundColor.includes('0.16');}),'Full-screen glass keeps specified faint tint and 2px blur');
  check(await page.evaluate(()=>PhoneCamera.previewState().appliedRevision)===oldApplied,'New generation stays pending');
  await page.waitForTimeout(1100);check(await page.locator('#artwork-inspector').isVisible(),'Artwork persists after its entrance');
  check(await page.evaluate(()=>!PhoneCamera.previewState().entering),'Entrance finishes in settled interactive state');
  await page.locator('#apply-preview').click();check(await page.evaluate(()=>{const p=PhoneCamera.previewState();return p.appliedRevision===p.pendingRevision;}),'Apply commits pending artwork');
  await page.locator('#close-preview').click();
  // Real encoded files from browser canvas, including transparent PNG borders.
  const imported=await page.evaluate(async()=>{
    const c=document.createElement('canvas');c.width=80;c.height=40;const x=c.getContext('2d');x.fillStyle='#ff0000';x.fillRect(10,5,60,30);
    window.__glassPNG=await new Promise(resolve=>c.toBlob(resolve,'image/png'));
    return PhoneCamera.importFile(new File([__glassPNG],'red.png',{type:'image/png'}));
  });
  check(imported===true,'Valid local PNG imports');
  check(await page.evaluate(()=>PhoneCamera.previewState().artworkSource==='raster'),'Imported artwork is labeled raster');
  check(await page.evaluate(()=>{const c=document.querySelector('#artwork-preview'),x=c.getContext('2d');return x.getImageData(0,0,1,1).data[3]===0&&x.getImageData(c.width/2,c.height/2,1,1).data[0]===255;}),'Import preserves transparent letterboxing and artwork color');
  await page.locator('#apply-preview').click();
  check(await page.evaluate(()=>PhoneCamera.previewState().appliedSource==='raster'),'Imported image explicitly applies to confirmed object');
  await page.evaluate(()=>PhoneCamera.importFile(new File([__glassPNG],'pending-other.png',{type:'image/png'})));
  const revision=await page.evaluate(()=>PhoneCamera.previewState().pendingRevision);
  const bad=await page.evaluate(async()=>{
    const p=new Uint8Array(await __glassPNG.arrayBuffer());const v=new DataView(p.buffer);v.setUint32(16,6000);v.setUint32(20,5000);
    const files=[new File(['<svg/>'],'unsafe.svg',{type:'image/svg+xml'}),new File(['broken'],'bad.png',{type:'image/png'}),new File([p],'large.png',{type:'image/png'}),new File([new Uint8Array(12*1024*1024+1)],'oversize.png',{type:'image/png'})];
    const result=[];for(const f of files){try{result.push(await PhoneCamera.importFile(f));}catch{result.push(false);}}return result;
  });
  check(bad.every(x=>x===false),'Unsupported, corrupt, oversized dimensions and oversized files reject');
  check(await page.evaluate(()=>PhoneCamera.previewState().pendingRevision)===revision,'Rejected import preserves pending artwork');
  await page.locator('#close-preview').click();
  await page.locator('#ask-toggle').click();await page.locator('#phone-prompt').fill('green and cream waves');await page.locator('#generate-artwork').click();
  check(await page.evaluate(()=>PhoneCamera.previewState().artworkSource==='recipe'),'Generating after import returns to procedural artwork');
  await page.evaluate(()=>{const f={name:'obsolete.png',type:'image/png',size:24,slice(){return {arrayBuffer:()=>new Promise((_,reject)=>window.__rejectHeader=reject)};}};window.__lateFinished=false;void PhoneCamera.importFile(f).then(()=>window.__lateFinished=true);});
  await page.locator('#close-preview').click();await page.locator('#ask-toggle').click();await page.locator('#phone-prompt').fill('navy and cream grid');await page.locator('#generate-artwork').click();
  await page.evaluate(()=>__rejectHeader(Error('obsolete import failure')));await page.waitForFunction(()=>__lateFinished);
  check(!(await page.locator('#camera-status').textContent()).includes('obsolete import failure'),'Superseded import error cannot overwrite new generation');
  await page.evaluate(()=>__glass.freeze=true);await page.waitForTimeout(400);
  check(await page.locator('#artwork-inspector').isVisible()&&await page.locator('#apply-preview').isDisabled(),'Camera stall preserves inspection but disables Apply');
  await page.locator('#close-preview').click();
  if(await page.locator('#ask-panel').isVisible())await page.locator('#close-ask').click();
  await page.locator('#camera-menu-toggle').click();await page.locator('#stop').click();
  check(await page.evaluate(()=>__glass.tracks.every(t=>t.readyState==='ended')),'Stop releases camera tracks');
  check((await page.locator('#artwork-source-note').textContent()).includes('red.png'),'Discard restores applied raster filename with its pixels');
  check(requests.every(r=>r.method==='GET'),'No camera, mask or import upload requests');
  check(await page.evaluate(()=>__glass.requests.every(r=>r.audio===false)),'No microphone requested');
  const manifest=await page.evaluate(async()=>{const r=await fetch('/phone.webmanifest');return {status:r.status,cache:r.headers.get('cache-control'),value:await r.json()};});
  check(manifest.status===200&&manifest.value.start_url==='/camera'&&manifest.value.display==='standalone'&&manifest.cache==='no-store','Home-screen manifest uses fresh standalone camera launch');
  check(await page.evaluate(async()=>!('serviceWorker' in navigator)||(await navigator.serviceWorker.getRegistrations()).length===0),'No offline service worker installed');
  return {schema:'visualizeit_glass_camera_browser_v1',timestamp:new Date().toISOString(),checks,manufactured_camera_and_masks:true,physical_phone_verified:false};
}
