// Installed playwright-cli run-code --filename; starts from a paired studio with a saved design.
async page => {
  const checks=[];
  const check=(ok,message)=>{if(!ok)throw new Error(message);checks.push(message);};
  const pixel=id=>page.locator(id).evaluate(canvas=>canvas.toDataURL());
  await page.setViewportSize({width:390,height:844});
  // Reload the latest UI without retaining a token in persistent storage.
  // Caller must pair again after reload before this script.
  await page.waitForFunction(()=>document.querySelector('#connection-badge').textContent==='Studio paired');
  await page.getByRole('button',{name:'Load Browser validation - navy waves',exact:true}).last().click();
  await page.waitForFunction(()=>!document.querySelector('#handoff').disabled);
  const clip=await page.locator('#clip option').nth(1).getAttribute('value');
  check(Boolean(clip),'Actual workbench catalog supplies a clip');
  await page.locator('#clip').selectOption(clip);
  await page.locator('#handoff').click();
  await page.waitForFunction(()=>!document.querySelector('#download-manifest').disabled);
  await page.locator('#refresh').click();
  await page.waitForFunction(()=>document.querySelector('#server-status').textContent.startsWith('Studio connected.'));
  check(await page.locator('#clip').inputValue()===clip && !await page.locator('#download-manifest').isDisabled(),'Refresh preserves clip-bound manifest and selection');
  await page.route('**/api/studio/clips',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({state:'connected',clips:[],message:'Manufactured removed-clip regression'})}));
  await page.locator('#refresh').click();
  await page.waitForFunction(()=>document.querySelector('#clip').options.length===1);
  check(await page.locator('#download-manifest').isDisabled() && await page.locator('#download-native').isDisabled(),'Removed clip invalidates prior handoff downloads');
  await page.unroute('**/api/studio/clips');
  await page.locator('#refresh').click();
  await page.waitForFunction(()=>document.querySelector('#clip').options.length>1);
  // Hold a real handoff response while an earlier refresh removes its clip.
  await page.locator('#clip').selectOption(clip);
  let releaseHandoff,handoffSeen;
  const handoffGate=new Promise(resolve=>releaseHandoff=resolve),handoffStarted=new Promise(resolve=>handoffSeen=resolve);
  await page.route('**/api/studio/handoffs',async route=>{const response=await route.fetch();handoffSeen();await handoffGate;await route.fulfill({response});});
  await page.locator('#handoff').click();await handoffStarted;
  await page.route('**/api/studio/clips',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({state:'connected',clips:[],message:'Manufactured removal during handoff'})}));
  await page.locator('#refresh').click();
  await page.waitForFunction(()=>document.querySelector('#clip').options.length===1);
  releaseHandoff();
  await page.waitForFunction(()=>document.querySelector('#handoff').textContent==='Save handoff');
  check(await page.locator('#download-manifest').isDisabled() && await page.locator('#download-native').isDisabled(),'Delayed handoff cannot reattach a removed clip');
  await page.unroute('**/api/studio/handoffs');await page.unroute('**/api/studio/clips');
  await page.locator('#refresh').click();
  await page.waitForFunction(()=>document.querySelector('#clip').options.length>1);
  // Save a snapshot, edit while its real response is delayed, then verify edits survive.
  let releaseSave,saveSeen;
  const saveGate=new Promise(resolve=>releaseSave=resolve),saveStarted=new Promise(resolve=>saveSeen=resolve);
  await page.route('**/api/studio/designs',async route=>{if(route.request().method()!=='POST')return route.continue();const response=await route.fetch();saveSeen();await saveGate;await route.fulfill({response});});
  await page.locator('#save').click();
  await saveStarted;
  await page.locator('#design-text').fill('Newer edits must survive');
  const changed=await pixel('#artwork');
  releaseSave();
  await page.waitForFunction(()=>!document.querySelector('#save').disabled);
  check(await pixel('#artwork')===changed && await page.locator('#handoff').isDisabled(),'Delayed save preserves newer edits and disables stale handoff');
  await page.unroute('**/api/studio/designs');
  // Local photo and placement must not trigger uploads or contaminate artwork.
  const requests=[];const watch=request=>{if(/^https?:/.test(request.url()))requests.push({url:request.url(),method:request.method()});};page.on('request',watch);
  await page.locator('.photo-section summary').click();
  const photoBytes=Uint8Array.from(atob(changed.split(',')[1]),c=>c.charCodeAt(0));
  await page.locator('#photo').setInputFiles({name:'manufactured-photo.png',mimeType:'image/png',buffer:Buffer.from(photoBytes)});
  await page.waitForFunction(()=>!document.querySelector('#placement-controls').hidden);
  const placed=await pixel('#preview');
  await page.locator('#preview').focus();await page.keyboard.press('ArrowRight');
  check(await pixel('#preview')!==placed,'Keyboard movement changes the 2D placement');
  check(await pixel('#artwork')===changed,'Photo and placement preserve artwork-only PNG');
  check(requests.length===0,'Photo selection and placement sent no network requests');
  page.off('request',watch);
  await page.locator('#clear-photo').click();
  check(await page.locator('#placement-controls').isHidden(),'Clear photo removes placement controls');
  // A server refusal must leave local work intact and expose the failure.
  await page.route('**/api/studio/designs',route=>route.request().method()==='POST'?route.fulfill({status:507,contentType:'application/json',body:JSON.stringify({error:{code:'fixture_reserve',message:'Manufactured reserve refusal'}})}):route.continue());
  await page.locator('#save').click();
  await page.waitForFunction(()=>document.querySelector('#server-status').textContent.includes('Manufactured reserve refusal'));
  check(await pixel('#artwork')===changed,'Failed save preserves local artwork and displays refusal');
  await page.unroute('**/api/studio/designs');
  return {schema:'visualizeit_studio_browser_edge_checks_v1',timestamp_utc:new Date().toISOString(),checks,mocked_failures:true,physical_phone_verified:false};
}
