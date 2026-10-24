// Layout and material checks only; not physical Safari or tracking accuracy.
async page=>{
  const checks=[];const check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
  await page.goto('http://127.0.0.1:8892/camera');await page.bringToFront();
  await page.evaluate(()=>{document.querySelector('.camera-stage').style.background='linear-gradient(125deg,#e5ded1 0%,#ae8971 35%,#173b34 65%,#070d12 100%)';});
  for(const [width,height,safe] of [[320,568,0],[393,650,0],[390,844,34],[844,390,21]]){
    await page.setViewportSize({width,height});await page.evaluate(bottom=>document.documentElement.style.setProperty('--safe-bottom',bottom+'px'),safe);
    check(await page.evaluate(()=>{
      const ids=['ask-toggle','confirm-object','import-design'],buttons=ids.map(id=>document.getElementById(id));let el=buttons[0].parentElement;
      while(el&&!buttons.every(b=>el.contains(b)))el=el.parentElement;
      while(el&&el.id!=='camera-studio'){
        const s=getComputedStyle(el),p=getComputedStyle(el,'::before'),blur=s.backdropFilter||s.webkitBackdropFilter,pseudoBlur=p.backdropFilter||p.webkitBackdropFilter;
        if(((blur&&blur!=='none')||(pseudoBlur&&pseudoBlur!=='none'))&&parseFloat(s.borderRadius)>=20)return true;el=el.parentElement;
      }return false;
    }),`Actions share one rounded glass surface at ${width}x${height}`);
    check(await page.evaluate(()=>{
      const ids=['ask-toggle','confirm-object','import-design','show-design','camera-status'];const rects=ids.map(id=>({id,el:document.getElementById(id)})).filter(x=>x.el.getClientRects().length&&!x.el.hidden).map(x=>({...x,rect:x.el.getBoundingClientRect()}));
      for(let i=0;i<rects.length;i++)for(let j=i+1;j<rects.length;j++){const a=rects[i].rect,b=rects[j].rect;if(a.left<b.right&&a.right>b.left&&a.top<b.bottom&&a.bottom>b.top)return false;}return true;
    }),`Caption and primary actions do not overlap at ${width}x${height}`);
    check(await page.evaluate(()=>['ask-toggle','confirm-object','import-design'].every(id=>{const r=document.getElementById(id).getBoundingClientRect();return r.width>=44&&r.height>=44&&r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight;})),`Primary controls are reachable 44px targets at ${width}x${height}`);
  }
  await page.setViewportSize({width:393,height:650});await page.evaluate(()=>document.documentElement.style.setProperty('--safe-bottom','0px'));
  await page.locator('#ask-toggle').click();
  check(await page.locator('#phone-prompt').isVisible(),'Glass composer remains reachable');
  await page.locator('#phone-prompt').fill('green and cream stripes');await page.locator('#generate-artwork').click();
  await page.waitForTimeout(1100);check(await page.locator('#artwork-inspector').isVisible(),'Artwork persists after bottom-origin entrance');
  await page.locator('#close-preview').click();
  check(await page.locator('#ask-toggle').isVisible(),'Closing artwork returns to the camera dock');
  await page.screenshot({path:'.cache/phone-glass-dock-393.png'});
  return {checks:checks.length,names:checks,physicalPhoneVerified:false};
}
