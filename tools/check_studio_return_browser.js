async page => {
  const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
  const base='http://127.0.0.1:8892';
  const focusComposer=async()=>{if(!(await page.locator('#phone-prompt').isVisible()))await page.evaluate(()=>PhoneCamera.openSurface('generate'));await page.locator('#phone-prompt').focus();};
  const backToStudio=async()=>{if(await page.evaluate(()=>PhoneCamera.surface()!=='camera'))await page.evaluate(()=>PhoneCamera.openSurface('camera'));if(!(await page.locator('#camera-menu').isVisible()))await page.locator('#camera-menu-toggle').click();await page.locator('#back-to-studio').click();};
  const canvasState=async selector=>page.locator(selector).evaluate(async canvas=>{
    const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
    return {width:canvas.width,height:canvas.height,sha256:Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',pixels))).map(v=>v.toString(16).padStart(2,'0')).join('')};
  });
  const mutations=[];page.on('request',r=>{if(r.method()!=='GET')mutations.push({method:r.method(),path:new URL(r.url()).pathname});});
  await page.goto(base+'/camera');
  await focusComposer();
  const initial=await canvasState('#artwork-preview');
  const generated='waves teal cream text "Keep This CASE" scale 1.5 rotation 17 seed 521';
  await page.locator('#phone-prompt').fill(generated);
  await page.locator('#phone-prompt-form').evaluate(form=>form.requestSubmit());
  const before=await canvasState('#artwork-preview');check(before.sha256!==initial.sha256,'Camera generates a distinct design');
  await page.locator('#close-preview').click();await focusComposer();
  await page.locator('#phone-prompt').fill('unsubmitted draft must not replace generated design');
  if(await page.locator('#generate-close').isVisible())await page.locator('#generate-close').click();else await page.locator('#phone-prompt').evaluate(el=>el.blur());
  await backToStudio();await page.waitForURL(base+'/');
  check((await canvasState('#artwork')).sha256===before.sha256,'Camera-to-studio pixels match exactly');
  check(await page.locator('#prompt').inputValue()===generated,'Generated prompt restored; unsubmitted draft excluded');
  check(await page.evaluate(()=>sessionStorage.getItem('visualizeit_studio_return_v1'))===null,'Return payload consumed once');
  check((await page.locator('#saved-status').textContent()).includes('Unsaved'),'Returned artwork is unsaved');
  await page.locator('#open-camera').click();await page.waitForURL(base+'/camera');
  check((await canvasState('#artwork-preview')).sha256===before.sha256,'Studio-to-camera-to-studio artwork roundtrip is exact');
  await focusComposer();
  await page.locator('#phone-prompt').fill('dots red blue seed 42');
  await page.locator('#phone-prompt-form').evaluate(form=>form.requestSubmit());
  await page.evaluate(()=>PhoneCamera.reset());
  await backToStudio();await page.waitForURL(base+'/');
  check((await canvasState('#artwork')).sha256===before.sha256,'Reset discards pending artwork and returns committed pixels');
  check(await page.locator('#prompt').inputValue()===generated,'Reset returns prompt matching committed artwork');
  const invalid=['{broken',JSON.stringify({schema:'wrong',recipe:{},prompt:''}),'x'.repeat(8193),JSON.stringify({schema:'visualizeit_studio_return_v1',recipe:{},prompt:''})];
  for(let i=0;i<invalid.length;i++){
    await page.evaluate(raw=>sessionStorage.setItem('visualizeit_studio_return_v1',raw),invalid[i]);await page.reload();
    check((await page.locator('#local-status').textContent()).includes('Artwork transfer unavailable'),'Invalid transfer '+i+' rejected');
    check(await page.evaluate(()=>sessionStorage.getItem('visualizeit_studio_return_v1'))===null,'Invalid transfer '+i+' consumed');
    check((await canvasState('#artwork')).sha256===initial.sha256,'Invalid transfer '+i+' preserves default artwork');
  }
  await page.goto(base+'/camera');
  await page.evaluate(()=>{Storage.prototype.setItem=function(){throw new DOMException('Manufactured storage denial','QuotaExceededError');};});
  await backToStudio();
  check(page.url()===base+'/camera','Storage denial preserves camera page');
  check((await page.locator('#camera-status').textContent()).includes('Could not transfer artwork'),'Storage denial reports an error');
  check(mutations.length===0,'Transfer and failure controls issue no POST or other network mutation');
  await page.reload();
  return {schema:'studio_return_browser_validation_v1',checks,passed:checks.length,requests_with_non_get:mutations,physical_camera_used:false,model_execution:false};
}
