// Browser interaction checks only; no camera/model inference or object geometry.
async page=>{
  const checks=[];const check=(ok,label)=>{if(!ok)throw Error(label);checks.push(label);};
  const focusComposer=async()=>{if(!(await page.locator('#phone-prompt').isVisible()))await page.locator('#ask-toggle').click();await page.locator('#phone-prompt').focus();};
  await page.goto('http://127.0.0.1:8892/camera');
  await page.setViewportSize({width:390,height:844});
  await focusComposer();
  await page.locator('#phone-prompt').fill('navy and cream checker');
  await page.locator('#generate-artwork').click();await page.waitForTimeout(1200);
  const inspector=page.locator('#artwork-inspector'),card=page.locator('#preview-card');
  check(await inspector.isVisible(),'Preview persists beyond previous timeout without camera');
  check(await page.locator('#apply-preview').isDisabled(),'Applying requires a fresh selected object');
  const transform=()=>card.evaluate(c=>getComputedStyle(c).transform);
  const before=await transform(),box=await card.boundingBox();
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();
  await page.mouse.move(box.x+box.width/2+45,box.y+box.height/2+20,{steps:5});await page.mouse.up();
  check(await transform()!==before,'Dragging rotates artwork in perspective');
  const rotated=await transform();await page.mouse.wheel(0,-240);await page.waitForTimeout(100);
  check(await transform()!==rotated,'Mouse wheel changes preview zoom');
  await page.locator('#pan-preview').click();const pannedBefore=await transform(),b=await card.boundingBox();
  await page.mouse.move(b.x+b.width/2,b.y+b.height/2);await page.mouse.down();await page.mouse.move(b.x+b.width/2+30,b.y+b.height/2+25,{steps:3});await page.mouse.up();
  check(await transform()!==pannedBefore,'Move mode pans preview');
  await page.locator('#reset-preview').click();check(await transform()===before,'Reset restores default presentation');
  // Browser-dispatched two-finger input tests pointer handling, not a physical phone.
  const touch=await page.context().newCDPSession(page),area=await page.locator('#preview-gesture').boundingBox();
  const cx=area.x+area.width/2,cy=area.y+area.height/2;
  await touch.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:cx-25,y:cy,id:1},{x:cx+25,y:cy,id:2}]});
  await touch.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:cx-45,y:cy,id:1},{x:cx+45,y:cy,id:2}]});
  await touch.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
  check(await page.evaluate(()=>PhoneCamera.previewState().scale>1),'Two-finger pinch enlarges preview');
  await touch.detach();await page.locator('#reset-preview').click();
  await page.locator('#preview-gesture').focus();await page.keyboard.press('ArrowRight');
  check(await page.evaluate(()=>PhoneCamera.previewState().rotation.y===10),'Keyboard arrow rotates preview');
  await page.keyboard.press('+');check(await page.evaluate(()=>PhoneCamera.previewState().scale>1),'Keyboard zoom works');
  await page.locator('#reset-preview').click();
  await page.locator('#preview-gesture').focus();
  for(let i=0;i<12;i++){await page.keyboard.press('Tab');check(await page.evaluate(()=>!!document.activeElement.closest('#artwork-inspector')),`Inspector contains keyboard focus ${i+1}`);}
  await page.locator('#close-preview').click();check(await inspector.isHidden(),'X closes preview');
  await page.emulateMedia({reducedMotion:'reduce'});await focusComposer();await page.locator('#generate-artwork').click();
  check(await inspector.isVisible(),'Reduced motion retains interactive design');
  await page.keyboard.press('Escape');check(await inspector.isHidden(),'Escape closes preview');
  await page.emulateMedia({reducedMotion:'no-preference'});
  for(const [width,height] of [[320,844],[390,844],[1280,844],[844,390]]){await page.setViewportSize({width,height});await focusComposer();await page.locator('#generate-artwork').click();check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`Inspector fits ${width}x${height}`);await page.locator('#pan-preview').click();await page.locator('#reset-preview').click();await page.locator('#close-preview').click();check(await inspector.isHidden(),`Inspector tools and X reachable at ${width}x${height}`);}
  return {schema:'visualizeit_artwork_inspector_browser_v1',timestamp:new Date().toISOString(),checks,physical_phone_verified:false};
}
