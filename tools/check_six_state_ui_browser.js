// Selected six-state UI contract. Manufactured camera frames verify software only.
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
      for(let i=0;i<160*120;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const j=i*4;p.data[j]=p.data[j+1]=p.data[j+2]=100+Math.round((seed>>>24)*.25);p.data[j+3]=255;}
      const draw=()=>{if(__reference.freeze)return;if(__reference.flat){ctx.fillStyle='#888';ctx.fillRect(0,0,160,120);}else ctx.putImageData(p,0,0);};draw();const media=c.captureStream(15),timer=setInterval(draw,70);
      for(const t of media.getTracks()){const stop=t.stop.bind(t);t.stop=()=>{clearInterval(timer);stop();};__reference.tracks.push(t);}return media;
    }});
  });
  const uploads=[],errors=[];page.on('request',r=>{if(r.method()!=='GET')uploads.push(r.url());});page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8892/camera');await page.bringToFront();
  const button=name=>page.getByRole('button',{name,exact:true});
  const onScreen=async locator=>locator.evaluate(el=>{const r=el.getBoundingClientRect();return r.width>=44&&r.height>=44&&r.left>=0&&r.right<=innerWidth+1&&r.top>=0&&r.bottom<=innerHeight+1;});
  const screenshot=async name=>page.screenshot({path:'.cache/six-state-'+name+'.png'});

  const surface=async name=>{await page.waitForFunction(n=>document.body.dataset.surface===n,name);await page.waitForTimeout(450);};
  const close=async()=>{await page.keyboard.press('Escape');await surface('camera');};
  for(const [width,height] of [[375,667],[393,852],[844,390],[768,1024]]){
    await page.setViewportSize({width,height});
    check(await page.locator('#primary-controls').isVisible(),`Home actions visible ${width}x${height}`);
    check(await page.evaluate(()=>{const side=document.querySelector('#ask-button .dock-circle').getBoundingClientRect(),center=document.querySelector('#analyze .dock-circle').getBoundingClientRect();return Math.abs(center.width/side.width-1.3)<.005&&Math.abs(center.height/side.height-1.3)<.005;}),`Analyze circle is exactly 1.3x ${width}x${height}`);
    for(const id of ['ask-button','analyze','design-button','ask-open','camera-menu-toggle'])check(await onScreen(page.locator('#'+id)),`${id} reachable ${width}x${height}`);
    await page.locator('#design-button').click();await surface('chooser');
    check(!(await page.locator('#primary-controls').isVisible()),`Chooser hides home controls ${width}x${height}`);if(height>width)check(await page.locator('#chooser-sheet .sheet').evaluate(el=>el.getBoundingClientRect().height>=innerHeight*.33),'Portrait chooser matches reference height');
    check(await onScreen(page.locator('#choose-generate'))&&await onScreen(page.locator('#import-design')),`Chooser actions in viewport ${width}x${height}`);
    await page.locator('#choose-generate').click();await surface('generate');
    check(await onScreen(page.locator('#generate-close')),`Generate close reachable ${width}x${height}`);await page.locator('#generate-artwork').scrollIntoViewIfNeeded();check(await onScreen(page.locator('#generate-artwork')),`Generate CTA reachable by panel scroll ${width}x${height}`);
    check(await page.evaluate(()=>{const r=document.querySelector('#prompt-ideas'),chips=[...r.children];return r.scrollWidth>r.clientWidth&&chips.every(c=>!c.querySelector('svg,img')&&c.textContent.trim())&&new Set(chips.map(c=>Math.round(c.getBoundingClientRect().top))).size===1;}),`Ideas are one horizontal text-only row ${width}x${height}`);
    check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`No page horizontal overflow ${width}x${height}`);
    await close();
    await page.locator('#ask-button').click();await surface('chat');check(await page.locator('#chat-surface #chat-close').isVisible()&&await page.locator('.camera-topbar').evaluate(el=>el.inert),'Chat close belongs to modal subtree; outside header is inert');
    check(await page.evaluate(()=>{const r=document.querySelector('#chat-surface').getBoundingClientRect();return r.top<=1&&r.left<=1&&r.width>=innerWidth-1&&r.height>=innerHeight-1;}),`Ask fills viewport ${width}x${height}`);
    check(!(await page.locator('#chat-surface .sheet-handle').count()),'Full-screen chat has no sheet handle');
    check(await onScreen(page.locator('#chat-send')),`Chat send reachable ${width}x${height}`);
    await close();
  }
  await page.setViewportSize({width:393,height:852});
  await page.locator('#start').click();await page.waitForFunction(()=>PhoneCamera.diagnostics().runtime==='ready'&&PhoneCamera.diagnostics().renderMode==='synchronized');
  check(await page.evaluate(()=>__reference.requests===0),'No segmentation before deliberate input');
  await screenshot('home');
  const calm=await page.locator('.edge-rim').evaluate(el=>getComputedStyle(el).opacity);
  const bounds=await page.locator('#analyze').boundingBox();await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);await page.mouse.down();
  await page.waitForFunction(()=>PhoneCamera.diagnostics().analyzeHeld);await page.waitForTimeout(300);
  check(await page.locator('#ask-button').isVisible()&&await page.locator('#design-button').isVisible(),'Analyze hold keeps side controls visible');
  check(Number(await page.locator('.edge-rim').evaluate(el=>getComputedStyle(el).opacity))>Number(calm),'Analyze hold measurably intensifies edge glow');await screenshot('analyze');await page.mouse.up();
  await page.waitForFunction(()=>!PhoneCamera.diagnostics().analyzeHeld);
  check(await page.evaluate(()=>__reference.requests===1),'One segmentation request per hold');
  await page.locator('#analyze').focus();await page.keyboard.down('Space');await page.waitForFunction(()=>PhoneCamera.diagnostics().analyzeHeld);await page.keyboard.up('Space');
  check(!await page.evaluate(()=>PhoneCamera.diagnostics().analyzeHeld),'Keyboard release clears active hold');
  await page.locator('#analyze').dispatchEvent('pointerdown',{button:0,pointerId:33});await page.locator('#analyze').dispatchEvent('pointercancel',{pointerId:33});
  check(!await page.evaluate(()=>PhoneCamera.diagnostics().analyzeHeld),'Pointer cancellation clears active hold');
  await page.locator('#ask-button').click();await page.locator('#chat-input').fill('A cylindrical ceramic vase, matte, blue and white stripes');await page.locator('#chat-send').click();
  check(await page.locator('#chat-log .msg.user').count()===1&&await page.locator('#chat-log .msg.assistant').count()===2,'Actual local idea exchange renders user and helper bubbles');
  check(await page.locator('#chat-use-brief').isVisible(),'Idea brief can feed generation');await screenshot('chat');
  await page.locator('#chat-use-brief').click();await surface('generate');
  check((await page.locator('#phone-prompt').inputValue()).includes('vase')&&(await page.locator('#phone-prompt').inputValue()).includes('stripes'),'Real object and pattern brief transfers to generation');
  await page.locator('#phone-prompt').fill('A minimalist chair');await page.locator('#generate-artwork').click();
  check(await page.evaluate(()=>PhoneCamera.surface()==='generate')&&/not objects/i.test(await page.locator('#prompt-status').textContent()),'Object-only input cannot masquerade as generated object');
  await page.locator('#phone-prompt').fill('blue and cream stripes');await page.locator('#phone-prompt').dispatchEvent('input');await screenshot('generate');
  const revision=await page.evaluate(()=>PhoneCamera.previewState().pendingRevision);
  await page.locator('#generate-artwork').click();await surface('result');await page.waitForTimeout(1100);
  check(await page.evaluate(r=>PhoneCamera.previewState().pendingRevision>r,revision),'Generate creates a new actual artwork revision');
  check(await page.evaluate(()=>{const p=PhoneCamera.previewState();return !p.applied&&p.open&&p.pan.x===0&&p.pan.y===0&&p.rotation.x===0&&p.rotation.y===0;}),'Result starts forward-facing and independent from attachment');
  check(!(await page.locator('#primary-controls').isVisible()),'Result sheet hides home actions');
  await screenshot('result');
  await page.waitForTimeout(900);check(await page.locator('#artwork-inspector').isVisible(),'Result persists after animation');
  const g=await page.locator('#preview-gesture').boundingBox();await page.mouse.move(g.x+g.width*.45,g.y+g.height*.45);await page.mouse.down();await page.mouse.move(g.x+g.width*.6,g.y+g.height*.5,{steps:5});
  check(await page.evaluate(()=>PhoneCamera.previewState().rotation.y!==0),'Preview drag responds in real time');await page.mouse.up();await page.waitForTimeout(700);
  check(await page.evaluate(()=>{const p=PhoneCamera.previewState();return p.pan.x===0&&p.pan.y===0&&p.rotation.x===0&&p.rotation.y===0;}),'Preview rubber-bands to its centered resting pose');
  for(let i=0;i<8;i++){await page.keyboard.press('Tab');check(await page.evaluate(()=>!!document.activeElement.closest('#artwork-inspector')),'Result contains keyboard focus');}
  await page.locator('#result-regenerate').click();await surface('generate');check(await page.locator('#phone-prompt').inputValue()==='blue and cream stripes','Regenerate preserves prior prompt');
  await page.locator('#generate-close').click();await surface('result');await page.locator('#close-preview').click();await surface('camera');
  await page.locator('#design-button').click();await surface('chooser');await screenshot('chooser');await close();
  await page.emulateMedia({reducedMotion:'reduce'});await page.locator('#design-button').click();await page.locator('#choose-generate').click();await page.locator('#generate-artwork').click();await surface('result');await page.waitForTimeout(200);
  check(!await page.evaluate(()=>PhoneCamera.previewState().entering),'Reduced motion settles without long spin');await page.locator('#close-preview').click();
  await page.evaluate(()=>PhoneCamera.close());check(await page.evaluate(()=>__reference.tracks.every(t=>t.readyState==='ended')),'Closing releases all manufactured camera tracks');
  check(uploads.length===0,'No frame, prompt or diagnostic uploads');check(errors.length===0,'No uncaught browser errors');
  return {checks:checks.length,names:checks,uploads,errors,physicalPhoneVerified:false,background:'Manufactured camera pixels; visual layout evidence only'};
}
