// Run with the installed playwright-cli run-code --filename after pairing the local studio.
// Creates one small explicitly named validation design and handoff; no inference.
async page => {
  const checks=[];
  const check=(condition,message)=>{if(!condition)throw new Error(message);checks.push(message);};
  await page.waitForFunction(()=>document.querySelector('#connection-badge').textContent==='Studio paired' && !document.querySelector('#save').disabled);
  check(await page.evaluate(()=>!location.hash && localStorage.length===0 && sessionStorage.length===0),'Pairing cleared fragment and persisted no credentials');
  for(const width of [320,390,1280]){
    await page.setViewportSize({width,height:844});
    check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`No horizontal overflow at ${width}px`);
  }
  const pixel=()=>page.locator('#artwork').evaluate(canvas=>canvas.toDataURL());
  await page.locator('#pattern').selectOption('solid');
  const initial=await pixel();
  await page.locator('#prompt').fill('navy and cream waves rotation 25 text HELLO');
  await page.locator('#prompt-form button').click();
  check(await page.locator('#pattern').inputValue()==='waves','Prompt selected waves');
  check(await page.locator('#design-text').inputValue()==='HELLO','Prompt preserved text');
  const generated=await pixel();
  check(generated!==initial,'Prompt changed artwork pixels');
  await page.locator('#design-name').fill('Browser validation - navy waves');
  const saveResponse=page.waitForResponse(r=>r.url().endsWith('/api/studio/designs')&&r.request().method()==='POST');
  await page.locator('#save').click();
  const saved=await (await saveResponse).json();
  await page.waitForFunction(()=>!document.querySelector('#handoff').disabled);
  check(saved.recipe.text==='HELLO' && saved.recipe.pattern==='waves','Actual backend saved displayed recipe');
  check(await page.evaluate(async ({data,expected})=>{
    const bytes=Uint8Array.from(atob(data.split(',')[1]),c=>c.charCodeAt(0));
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');return digest===expected;
  },{data:generated,expected:saved.png_sha256}),'Saved PNG hash matches browser artwork');
  const handoffResponse=page.waitForResponse(r=>r.url().endsWith('/api/studio/handoffs')&&r.request().method()==='POST');
  await page.locator('#handoff').click();
  const manifest=await (await handoffResponse).json();
  await page.waitForFunction(()=>!document.querySelector('#download-native').disabled);
  check(manifest.design_id===saved.id && manifest.png_sha256===saved.png_sha256 && manifest.tracking_applied===false && manifest.native_import_verified===false,'Real handoff references exact PNG and honest unverified import');
  const downloaded=page.waitForEvent('download');
  await page.locator('#download-native').click();
  const download=await downloaded;
  check((await download.failure())===null,'Authenticated native PNG download completed');
  await page.locator('#design-text').fill('CHANGED');
  check(await page.locator('#handoff').isDisabled(),'Edited artwork cannot reuse stale handoff');
  await page.getByRole('button',{name:'Load Browser validation - navy waves',exact:true}).last().click();
  await page.waitForFunction(()=>document.querySelector('#design-text').value==='HELLO');
  check(await pixel()===generated,'Loading saved design restores exact PNG pixels');
  await page.setViewportSize({width:390,height:844});
  await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:'.cache/phone-studio-mobile.png'});
  return {schema:'visualizeit_studio_browser_checks_v1',timestamp_utc:new Date().toISOString(),checks,saved_design:saved.id,handoff:manifest.id,physical_phone_verified:false};
}
