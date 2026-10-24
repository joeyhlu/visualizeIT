// Contrast from browser-resolved CSS, composited against worst-case white footage.
// This checks selected text surfaces, not WCAG conformance or a physical phone.
async page=>{
 await page.goto('http://127.0.0.1:8892/camera');await page.setViewportSize({width:393,height:852});
 await page.locator('#design-button').click();await page.locator('#choose-generate').click();
 const measured=await page.evaluate(()=>{
  const rgba=s=>{const a=s.match(/rgba?\(([^)]+)\)/);if(!a)throw Error('Unparsed CSS color '+s);const v=a[1].split(/[,\s/]+/).filter(Boolean).map(Number);return [v[0],v[1],v[2],v.length>3?v[3]:1];};
  const over=(f,b)=>f.slice(0,3).map((v,i)=>v*f[3]+b[i]*(1-f[3]));
  const l=rgb=>rgb.map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((n,v,i)=>n+v*[.2126,.7152,.0722][i],0);
  const ratio=(a,b)=>{const x=l(a),y=l(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05);};
  const stops=el=>[...getComputedStyle(el).backgroundImage.matchAll(/rgba?\([^)]+\)/g)].map(m=>rgba(m[0]));
  const bases=(el,bg)=>{const css=getComputedStyle(el),base=over(rgba(css.backgroundColor),bg),s=stops(el);return s.length?s.map(c=>over(c,base)):[base];};
  const sheet=document.querySelector('.generate-sheet'),white=[255,255,255],sheetBgs=bases(sheet,white),results=[];
  const add=(label,foreground,bgs,min=4.5)=>{const ratios=bgs.map(bg=>ratio(over(rgba(foreground),bg),bg));results.push({label,minimum:Math.min(...ratios),required:min,pass:Math.min(...ratios)>=min});};
  add('Sheet title over white footage',getComputedStyle(document.querySelector('#generate-title')).color,sheetBgs);
  add('Secondary capability copy over white footage',getComputedStyle(document.querySelector('#generate-capability')).color,sheetBgs);
  const field=document.querySelector('.prompt-field'),fieldBgs=sheetBgs.flatMap(bg=>bases(field,bg));
  add('Prompt placeholder over white footage',getComputedStyle(document.querySelector('#phone-prompt'),'::placeholder').color,fieldBgs);
  const cta=document.querySelector('#generate-artwork');add('Gradient CTA text at all gradient stops',getComputedStyle(cta).color,stops(cta).map(c=>over(c,white)));
  const close=document.querySelector('#generate-close');add('Close icon on sheet',getComputedStyle(close).color,sheetBgs.flatMap(bg=>bases(close,bg)),3);
  return results;
 });
 if(measured.some(r=>!r.pass))throw Error(JSON.stringify(measured));
 await page.keyboard.press('Escape');await page.locator('#ask-button').click();
 const semantics=await page.evaluate(()=>{const d=document.querySelector('#chat-surface'),c=document.querySelector('#chat-close');return d.getAttribute('aria-modal')==='true'&&d.contains(c)&&document.querySelector('.camera-topbar').inert;});
 if(!semantics)throw Error('Modal ownership failure');await page.locator('#chat-close').click();
 if(await page.evaluate(()=>PhoneCamera.surface())!=='camera')throw Error('Internal chat close did not return home');
 return {measured,insideModalCloseVerified:semantics,physicalVoiceOverVerified:false};
}
