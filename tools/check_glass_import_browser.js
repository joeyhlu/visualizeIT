// Actual browser image encoding/decoding; no camera or model inference.
async page=>{
  await page.goto('http://127.0.0.1:8892/camera');
  const checks=await page.evaluate(async()=>{
    const checks=[];const check=(ok,label)=>{if(!ok)throw Error(label);checks.push(label);};
    const c=document.createElement('canvas');c.width=80;c.height=40;const x=c.getContext('2d');x.fillStyle='#e22233';x.fillRect(0,0,80,40);
    for(const [type,name] of [['image/jpeg','design.jpg'],['image/webp','design.webp']]){
      const blob=await new Promise(resolve=>c.toBlob(resolve,type,.9));check(blob.type===type,`${name} fixture uses requested encoder`);
      check(await PhoneCamera.importFile(new File([blob],name,{type})),`${name} decodes and opens inspection`);
    }
    const jpeg=new Uint8Array(await (await new Promise(resolve=>c.toBlob(resolve,'image/jpeg',.9))).arrayBuffer());
    const app=Uint8Array.from([255,225,0,34,69,120,105,102,0,0,73,73,42,0,8,0,0,0,1,0,18,1,3,0,1,0,0,0,6,0,0,0,0,0,0,0]);
    const sof=jpeg.findIndex((v,i)=>v===255&&jpeg[i+1]===192);check(sof>0,'Actual JPEG fixture contains SOF segment');
    const offset=sof+2+(jpeg[sof+2]<<8)+jpeg[sof+3],oriented=new Uint8Array(jpeg.length+app.length);
    oriented.set(jpeg.subarray(0,offset));oriented.set(app,offset);oriented.set(jpeg.subarray(offset),offset+app.length);
    check(await PhoneCamera.importFile(new File([oriented],'portrait-exif.jpg',{type:'image/jpeg'})),'JPEG with EXIF after SOF imports');
    const canvas=document.querySelector('#artwork-preview'),pixels=canvas.getContext('2d').getImageData(0,0,1024,1024).data;
    let left=1024,right=-1,top=1024,bottom=-1;for(let y=0;y<1024;y++)for(let x=0;x<1024;x++)if(pixels[(y*1024+x)*4+3]){left=Math.min(left,x);right=Math.max(right,x);top=Math.min(top,y);bottom=Math.max(bottom,y);}
    check(bottom-top+1===80&&right-left+1===40,'EXIF orientation preserves portrait dimensions without distortion');
    return checks;
  });
  await page.locator('#close-preview').click();
  const downloadPromise=page.waitForEvent('download');await page.locator('#camera-menu-toggle').click();await page.locator('#back-to-studio').click();await downloadPromise;
  if(page.url()!=='http://127.0.0.1:8892/camera')throw Error('Raster must remain in camera instead of transferring fake recipe');
  checks.push('Imported raster studio action exports PNG and preserves camera page');
  if(await page.evaluate(()=>sessionStorage.getItem('visualizeit_studio_return_v1'))!==null)throw Error('Raster leaked fake procedural recipe');
  checks.push('Raster exports without fake procedural handoff');
  return {schema:'visualizeit_glass_import_browser_v1',timestamp:new Date().toISOString(),checks,physical_phone_verified:false};
}
