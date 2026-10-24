// UI lifecycle checks with explicitly manufactured camera frames and masks.
// These are not physical-camera, segmentation-quality, or phone-speed evidence.
async page => {
  const checks=[];
  const check=(ok,label)=>{if(!ok)throw Error(label);checks.push(label);};
  await page.addInitScript(()=>{
    window.__cameraTest={tracks:[],requests:[],mode:'normal',processes:0,terminated:0};
    class FixtureWorker {
      postMessage(data){
        if(data.type==='initialize')return setTimeout(()=>this.onmessage?.({data:{id:data.id,type:'ready',epoch:data.epoch}}),30);
        if(data.type==='process'){
          window.__cameraTest.processes++;
          data.image?.close();
          const mask=new Uint8Array(data.tag.width*data.tag.height),negative=data.strokes.some(s=>s.mode==='negative');
          for(let y=Math.floor(data.tag.height*.25);y<data.tag.height*.75;y++)for(let x=Math.floor(data.tag.width*.25);x<data.tag.width*(negative?.5:.75);x++)mask[y*data.tag.width+x]=255;
          setTimeout(()=>{if(!this.closed)this.onmessage?.({data:{id:data.id,type:'mask',epoch:data.epoch,tag:data.tag,mask:mask.buffer}});},500);
        }
      }
      terminate(){this.closed=true;window.__cameraTest.terminated++;}
    }
    window.Worker=FixtureWorker;
    const media=async constraints=>{
      window.__cameraTest.requests.push(constraints);
      if(window.__cameraTest.mode==='denied')throw new DOMException('Manufactured denial','NotAllowedError');
      const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
      const c=canvas.getContext('2d');c.fillStyle='#ece7db';c.fillRect(0,0,640,480);c.fillStyle='#446b52';c.fillRect(160,120,320,240);
      const stream=canvas.captureStream(10);window.__cameraTest.tracks.push(...stream.getTracks());
      if(window.__cameraTest.mode==='delayed')await new Promise(resolve=>window.__cameraTest.release=resolve);
      return stream;
    };
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:media,configurable:true});
  });
  const requests=[];page.on('request',request=>{if(/^https?:/.test(request.url()))requests.push({url:request.url(),method:request.method()});});
  await page.goto('http://127.0.0.1:8892/');
  await page.locator('#prompt').fill('navy and cream waves text CAMERA');await page.locator('#prompt-form button').click();
  const before=await page.locator('#artwork').evaluate(c=>c.toDataURL());
  await page.locator('#open-camera').click();await page.waitForURL('**/camera');
  check(await page.locator('#artwork-preview').evaluate(c=>c.toDataURL())===before,'Studio artwork transfers exactly to camera workspace');
  check(await page.evaluate(()=>sessionStorage.getItem('visualizeit_camera_recipe_v1'))===null,'Artwork transfer recipe is consumed and cleared');
  for(const width of [320,390,1280]){await page.setViewportSize({width,height:844});check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`Camera fits ${width}px`);}
  await page.setViewportSize({width:390,height:844});
  await page.locator('#start').click();
  await page.waitForFunction(()=>window.__cameraTest.processes>0);
  const background=await page.locator('#scene').evaluate(c=>Array.from(c.getContext('2d').getImageData(5,5,1,1).data));
  await page.locator('#scene').click({position:{x:100,y:100}});
  await page.waitForFunction(()=>PhoneCamera.selection.phase==='locked');
  check(await page.evaluate(()=>__cameraTest.tracks.every(t=>t.readyState==='ended')),'Tap during inference freezes frame and stops camera tracks');
  await page.waitForFunction(()=>PhoneCamera.selection.mask!==null);
  const area=()=>page.evaluate(()=>PhoneCamera.selection.mask.reduce((n,x)=>n+(x?1:0),0));
  const positive=await area();
  await page.locator('input[name="brush"][value="negative"]').check();await page.locator('#scene').focus();await page.keyboard.press('Space');
  await page.waitForFunction(()=>PhoneCamera.selection.mask!==null);
  check(await area()<positive,'Remove stroke updates selected area');
  await page.locator('#undo').click();await page.waitForFunction(()=>PhoneCamera.selection.mask!==null);
  check(await area()===positive,'Undo restores previous selection');
  const outside=await page.locator('#scene').evaluate(c=>Array.from(c.getContext('2d').getImageData(5,5,1,1).data));
  check(outside.join(',')===background.join(','),'Artwork is clipped away from unselected background');
  check(requests.every(r=>r.method==='GET'),'Camera and corrections upload no frames');
  check(await page.evaluate(()=>__cameraTest.requests.every(r=>r.audio===false)),'Camera never requests audio');
  await page.locator('#stop').click();
  check(await page.evaluate(()=>PhoneCamera.selection.frame===null&&PhoneCamera.selection.mask===null),'Stop clears frames and masks');
  await page.evaluate(()=>__cameraTest.mode='denied');await page.locator('#start').click();
  await page.waitForFunction(()=>document.querySelector('#camera-status').textContent.includes('denied'));
  check(await page.evaluate(()=>PhoneCamera.selection.phase==='error'),'Permission denial produces explicit recoverable state');
  await page.evaluate(()=>__cameraTest.mode='delayed');await page.locator('#start').click();
  await page.waitForFunction(()=>typeof __cameraTest.release==='function');await page.locator('#stop').click();await page.evaluate(()=>__cameraTest.release());
  await page.waitForFunction(()=>__cameraTest.tracks.every(t=>t.readyState==='ended'));
  check(await page.evaluate(()=>PhoneCamera.selection.phase==='idle'),'Late permission result cannot restart stopped camera');
  return {schema:'visualizeit_phone_camera_browser_checks_v1',timestamp_utc:new Date().toISOString(),checks,manufactured_camera_and_masks:true,physical_phone_verified:false};
}
