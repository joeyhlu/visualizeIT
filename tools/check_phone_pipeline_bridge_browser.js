// Phone dialog software checks with a named fake upstream. No camera and no neural inference.
async page=>{
 const checks=[],check=(ok,name)=>{if(!ok)throw Error(name);checks.push(name);};
 const pin='27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311',requests=[],payloads=[];
 const status={schema:'visualizeit_phone_tracking_status_v1',state:'reachable_blocked',service_reachable:true,requested_head_sha256:pin,reported_export_sha256:pin,installed_head_sha256:null,model_match:true,recorded_admitted:false,head_configured:true,metric_bundle_configured:false,registered_clip_count:9,blockers:['neural_blocked','admission_not_admitted','metric_bundle_not_configured'],upstream_blockers:['torch_runtime','serving_admission'],recorded_supported:false,live_supported:false,live_reason:'live_neural_contract_unsupported',quality_claim:false,message:'Actual execution is blocked.'};
 let hold=false,pending=null,deny=false;
 await page.route('**/api/studio/**',async route=>{
  const r=route.request(),path=new URL(r.url()).pathname;requests.push({method:r.method(),path});
  if(hold&&path.endsWith('/tracking-status')){pending=route;return;}
  const reply=body=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(body)});
  if(deny)return route.fulfill({status:401,contentType:'application/json',body:JSON.stringify({error:{code:'unauthorized'}})});
  if(path.endsWith('/tracking-status'))return reply(status);
  if(path.endsWith('/clips'))return reply({state:'connected',clips:[{clip_id:'fixture-mug',label:'Manufactured clip'}]});
  if(path.endsWith('/designs')&&r.method()==='POST'){const body=r.postDataJSON();payloads.push(body);return reply({id:'a'.repeat(32)});}
  if(path.endsWith('/handoffs')&&r.method()==='POST'){const body=r.postDataJSON();return reply({schema:'visualizeit_artwork_handoff_v1',design_id:body.design_id,clip_id:body.clip_id,tracking_applied:false,native_import_verified:false});}
  return route.fulfill({status:404,body:'{}'});
 });
 await page.goto('http://127.0.0.1:8892/camera');await page.bringToFront();await page.setViewportSize({width:393,height:852});
 check(requests.length===0,'No control-plane requests or uploads on camera home');
 const open=async()=>{await page.locator('#camera-menu-toggle').click();await page.locator('#open-pipeline').click();};
 await open();check(await page.locator('#pipeline-dialog').isVisible(),'Tracking connection opens from Settings');
 check(requests.length===0,'Opening connection does not send artwork or camera data');
 await page.locator('#pipeline-token').fill('manufactured_pairing_token_only');await page.locator('#pipeline-pair').evaluate(f=>f.requestSubmit());
 await page.waitForFunction(()=>PhonePipeline.state()==='reachable_blocked'&&!PhonePipeline.busy());
 check(await page.locator('#pipeline-token').inputValue()==='','Pairing token field is cleared immediately');
 check(/none reported/i.test(await page.locator('#pipeline-installed').textContent()),'Requested hash is never presented as installed weights');
 check(/9/.test(await page.locator('#pipeline-clipcount').textContent()),'Original catalog clip count is displayed');
 check(/torch/i.test(await page.locator('#pipeline-blockers').textContent()),'Blocked runtime requirements remain visible');
 check(requests.every(r=>r.method==='GET'),'Pairing sends only status/catalog GETs');
 await page.locator('#pipeline-clip').selectOption('fixture-mug');await page.locator('#pipeline-save').click();
 await page.waitForFunction(()=>!PhonePipeline.busy()&&/manual-import handoff/i.test(document.querySelector('#pipeline-save-status').textContent));
 check(payloads.length===1&&payloads[0].schema==='visualizeit_design_v1','Explicit save sends an actual design record');
 check(payloads[0].png_base64.startsWith('iVBOR')&&payloads[0].recipe.pattern,'Handoff contains generated PNG and real pattern recipe');
 check(!Object.keys(payloads[0]).some(k=>/camera|frame|mask|pose/i.test(k)),'Artwork upload contains no camera/frame/mask/pose fields');
 check(/not applied/i.test(await page.locator('#pipeline-save-status').textContent()),'Manual handoff does not claim artwork tracking');
 check(requests.filter(r=>r.method==='POST').every(r=>/\/(designs|handoffs)$/.test(r.path)),'Only artwork and manual handoff endpoints receive POSTs');
 // A late status reply cannot restore a disconnected session.
 hold=true;await page.locator('#pipeline-refresh').click();await page.waitForTimeout(100);check(!!pending,'Controlled slow status request is pending');
 await page.locator('#pipeline-disconnect').click();if(pending){try{await pending.fulfill({status:200,contentType:'application/json',body:JSON.stringify(status)});}catch{}}
 await page.waitForTimeout(100);check(!await page.evaluate(()=>PhonePipeline.paired())&&/not paired/i.test(await page.locator('#pipeline-state').textContent()),'Disconnect aborts pending work and rejects stale status');
 hold=false;await page.locator('#pipeline-token').fill('manufactured_pairing_token_only');await page.locator('#pipeline-pair').evaluate(f=>f.requestSubmit());await page.waitForFunction(()=>PhonePipeline.state()==='reachable_blocked'&&!PhonePipeline.busy());
 await page.evaluate(()=>PhoneCamera.close());check(!await page.evaluate(()=>PhonePipeline.paired()),'Camera close API clears pairing and pending connection work');
 await page.locator('#pipeline-close').click();check(await page.locator('#primary-controls').isVisible(),'Closing connection preserves camera home');
 // Raster input must not be represented by a fake procedural recipe.
 await page.evaluate(async()=>{const c=document.createElement('canvas');c.width=20;c.height=20;c.getContext('2d').fillRect(0,0,20,20);const b=await new Promise(r=>c.toBlob(r));await PhoneCamera.importFile(new File([b],'fixture.png',{type:'image/png'}));});await page.locator('#close-preview').click();
 await open();await page.locator('#pipeline-token').fill('manufactured_pairing_token_only');await page.locator('#pipeline-pair').evaluate(f=>f.requestSubmit());await page.waitForFunction(()=>!PhonePipeline.busy());
 const before=payloads.length;if(await page.locator('#pipeline-save').isEnabled())await page.locator('#pipeline-save').click();
 check(payloads.length===before,'Imported raster cannot upload a fabricated recipe');
 await page.locator('#pipeline-close').click();
 check(!requests.some(r=>/frames|v1\//.test(r.path)),'No camera frames or legacy tracking endpoint requests');
 await open();await page.locator('#pipeline-disconnect').click();deny=true;await page.locator('#pipeline-token').fill('manufactured_rejected_token_only');await page.locator('#pipeline-pair').evaluate(f=>f.requestSubmit());await page.waitForTimeout(250);
 check(!await page.evaluate(()=>PhonePipeline.paired())&&await page.locator('#pipeline-pair').isVisible()&&await page.locator('#pipeline-save').isDisabled(),'Rejected token clears pairing and offers direct retry');await page.locator('#pipeline-close').click();
 return {checks:checks.length,names:checks,fakeUpstream:true,physicalPhoneVerified:false,actualNeuralExecution:false};
}
