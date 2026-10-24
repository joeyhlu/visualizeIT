(function(root){
  'use strict';
  // Tracking pipeline dialog: real control-plane status + explicit manual artwork handoff.
  // No camera frames are ever sent. No polling, no workers, no persistent storage.
  // The pairing token lives only in this closure and is cleared on disconnect/stop/hide/pagehide.
  if(typeof document==='undefined')return;
  const $=id=>document.getElementById(id);
  const dialog=$('pipeline-dialog');if(!dialog)return;
  const RESPONSE_LIMIT=262144,REQUEST_LIMIT=3*1024*1024,TIMEOUT_MS=10000;
  const PIN='27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311';
  const HEX64=/^[0-9a-f]{64}$/,ID32=/^[0-9a-f]{32}$/,CODE=/^[a-z][a-z0-9_.:-]{0,63}$/;
  const STATE_TEXT={disconnected:'Workbench not reachable',reachable_blocked:'Reachable · not ready',recorded_configured:'Configured for recorded clips only'};
  const BLOCKER_TEXT={
    workbench_unreachable:'The workbench service is not running or not reachable.',
    invalid_status:'The workbench returned an unreadable status.',
    unexpected_service:'The service is not the VisualizeIT workbench.',
    unexpected_schema:'The workbench status format is not the expected version.',
    model_metadata_missing:'The workbench did not report its model hash.',
    model_mismatch:'The workbench reports a different model than the pinned head.',
    neural_blockers_malformed:'Neural readiness information is malformed.',
    neural_blocked:'The workbench reports neural requirements are blocked.',
    admission_metadata_missing:'Model admission information is missing.',
    admission_not_admitted:'The model has not been admitted for recorded clips.',
    head_metadata_missing:'Head configuration information is missing.',
    head_not_configured:'The trained head is not configured.',
    metric_bundle_metadata_missing:'Metric bundle information is missing.',
    metric_bundle_not_configured:'The metric bundle is not configured.',
    unrecognized_blocker:'An unrecognized requirement is blocked.',
    installed_metadata_invalid:'The installed head information is malformed.',
    pipeline_metadata_missing:'Pipeline availability information is missing.',
    pipeline_disabled:'The workbench pipeline is not enabled.',
  };
  const s={token:'',epoch:0,controllers:new Set(),status:null,clips:[],busy:'',opener:null};

  function text(id,value,error=false){const el=$(id);el.textContent=value;el.classList.toggle('error',error);}
  function abortAll(){for(const c of s.controllers)c.abort();s.controllers.clear();}
  function invalidate(){s.epoch++;abortAll();s.busy='';}

  async function readLimited(response,limit){
    const declared=Number(response.headers.get('Content-Length'));
    if(Number.isFinite(declared)&&declared>limit){try{await response.body?.cancel();}catch{}throw Error('Workbench response is too large.');}
    if(!response.body){const buf=await response.arrayBuffer();if(buf.byteLength>limit)throw Error('Workbench response is too large.');return new Uint8Array(buf);}
    const reader=response.body.getReader(),chunks=[];let total=0;
    try{
      for(;;){const {done,value}=await reader.read();if(done)break;total+=value.byteLength;if(total>limit){try{await reader.cancel();}catch{}throw Error('Workbench response is too large.');}chunks.push(value);}
    }finally{try{reader.releaseLock();}catch{}}
    const out=new Uint8Array(total);let offset=0;for(const c of chunks){out.set(c,offset);offset+=c.byteLength;}return out;
  }
  async function api(path,{method='GET',body}={}){
    if(!s.token)throw Error('Pair with the startup token first.');
    let payload;
    if(body!==undefined){payload=JSON.stringify(body);if(new TextEncoder().encode(payload).byteLength>REQUEST_LIMIT)throw Error('Artwork request exceeds the 3 MiB limit.');}
    const epoch=s.epoch,token=s.token,controller=new AbortController();s.controllers.add(controller);
    const timer=setTimeout(()=>controller.abort(),TIMEOUT_MS);
    try{
      const response=await fetch(`/api/studio/${path}`,{method,headers:{Authorization:`Bearer ${token}`,Accept:'application/json',...(payload?{'Content-Type':'application/json'}:{})},...(payload?{body:payload}:{}),signal:controller.signal,credentials:'same-origin',cache:'no-store',redirect:'error',referrerPolicy:'no-referrer'});
      const bytes=await readLimited(response,RESPONSE_LIMIT);
      if(epoch!==s.epoch)throw Object.assign(Error('stale'),{stale:true});
      let data=null;try{data=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes));}catch{}
      if(!response.ok){const code=data?.error?.code;throw Object.assign(Error(response.status===401?'Token was not accepted. Pair again.':`Request failed${typeof code==='string'&&CODE.test(code)?` (${code})`:` (${response.status})`}.`),{status:response.status,unauthorized:response.status===401});}
      if(data===null||typeof data!=='object')throw Error('Workbench returned an unreadable response.');
      return data;
    }catch(error){
      if(epoch!==s.epoch||error.stale)throw Object.assign(Error('stale'),{stale:true});
      if(error.name==='AbortError')throw Error('Request timed out. Nothing was changed on the phone.');
      throw error;
    }finally{clearTimeout(timer);s.controllers.delete(controller);}
  }

  function validStatus(v){
    return !!v&&v.schema==='visualizeit_phone_tracking_status_v1'&&Object.hasOwn(STATE_TEXT,v.state)&&v.requested_head_sha256===PIN&&v.live_supported===false&&v.quality_claim===false&&Array.isArray(v.blockers);
  }
  function hashText(v){return typeof v==='string'&&HEX64.test(v)?`${v.slice(0,12)}…${v.slice(-6)}`:'none reported';}
  function renderStatus(){
    const v=s.status,paired=!!s.token;
    $('pipeline-paired').hidden=!paired;$('pipeline-pair').hidden=paired;
    text('pipeline-state',!paired?'Not paired':v?STATE_TEXT[v.state]:'Status unavailable');
    $('pipeline-state').dataset.state=v?.state||'none';
    text('pipeline-pin',hashText(PIN));
    text('pipeline-reported',v?hashText(v.reported_export_sha256):'—');
    text('pipeline-installed',v?hashText(v.installed_head_sha256):'—');
    text('pipeline-clipcount',v&&Number.isInteger(v.registered_clip_count)?String(v.registered_clip_count):'unknown');
    const list=$('pipeline-blockers');list.replaceChildren();
    const codes=v?[...v.blockers,...(Array.isArray(v.upstream_blockers)?v.upstream_blockers:[])]:[];
    for(const code of codes.slice(0,24)){if(typeof code!=='string'||!CODE.test(code))continue;const li=document.createElement('li');li.textContent=BLOCKER_TEXT[code]||`Workbench requirement: ${code}`;list.append(li);}
    $('pipeline-blockers-wrap').hidden=!list.children.length;
    const select=$('pipeline-clip'),previous=select.value;select.replaceChildren(new Option('No clip · artwork only',''));
    for(const clip of s.clips)select.append(new Option(clip.label,clip.clip_id));
    select.value=s.clips.some(c=>c.clip_id===previous)?previous:'';
    syncButtons();
  }
  function rasterArtwork(){return root.PhoneCamera?.previewState?.().artworkSource==='raster';}
  function syncButtons(){
    const paired=!!s.token,busy=!!s.busy,raster=rasterArtwork();
    $('pipeline-refresh').disabled=!paired||busy;$('pipeline-disconnect').disabled=!paired;
    $('pipeline-save').disabled=!paired||busy||raster;$('pipeline-clip').disabled=!paired||busy;
    $('pipeline-save').textContent=s.busy==='save'?'Saving…':'Save artwork for workbench';
    $('pipeline-pair-submit').disabled=busy;
    if(paired&&raster&&!busy)text('pipeline-save-status','Uploaded images can’t be saved here. Use Save artwork PNG in Artwork options.');
  }
  // A rejected token is dropped immediately so the Pair form is shown for a direct retry.
  function unauthorized(){disconnect('');text('pipeline-message','Token was not accepted. Paste the startup token and pair again.',true);$('pipeline-token').focus({preventScroll:true});}
  function sanitizeClips(result){
    if(!result||result.state!=='connected'||!Array.isArray(result.clips))return [];
    const out=[];for(const c of result.clips.slice(0,1000))if(c&&typeof c.clip_id==='string'&&/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(c.clip_id)&&typeof c.label==='string')out.push({clip_id:c.clip_id,label:c.label.slice(0,160)});
    return out;
  }
  async function refresh(){
    if(!s.token)return;const epoch=s.epoch;s.busy='refresh';syncButtons();text('pipeline-message','Checking the workbench…');
    try{
      const status=await api('tracking-status');if(!validStatus(status))throw Error('Unexpected tracking status format.');
      const clips=await api('clips');
      if(epoch!==s.epoch)return;
      s.status=status;s.clips=sanitizeClips(clips);renderStatus();
      text('pipeline-message',status.state==='disconnected'?'Paired with this studio, but the workbench is not reachable.':'Status refreshed. Live phone tracking is unavailable.');
    }catch(error){if(error.stale||epoch!==s.epoch)return;if(error.unauthorized){unauthorized();return;}s.status=null;renderStatus();text('pipeline-message',error.message,true);}
    finally{if(epoch===s.epoch){s.busy='';syncButtons();}}
  }
  function disconnect(note){
    invalidate();s.token='';s.status=null;s.clips=[];$('pipeline-token').value='';renderStatus();
    if(note)text('pipeline-message',note);text('pipeline-save-status','');
  }
  async function save(){
    if(!s.token||s.busy||rasterArtwork())return;
    const camera=root.PhoneCamera;if(!camera?.workbenchArtwork){text('pipeline-save-status','Camera artwork is unavailable.',true);return;}
    const name=$('pipeline-name').value.trim();
    if(!name||Array.from(name).length>80){text('pipeline-save-status','Give the artwork a name of 1 to 80 characters.',true);$('pipeline-name').focus();return;}
    let art;
    try{art=camera.workbenchArtwork();}catch(error){text('pipeline-save-status',error.message,true);return;}
    const clipId=$('pipeline-clip').value||null,epoch=s.epoch;
    s.busy='save';syncButtons();text('pipeline-save-status','Saving the current generated artwork…');
    try{
      const design=await api('designs',{method:'POST',body:{schema:'visualizeit_design_v1',name,prompt:art.prompt,recipe:art.recipe,png_base64:art.png_base64}});
      art.png_base64='';
      if(!design||typeof design.id!=='string'||!ID32.test(design.id))throw Error('Server returned an invalid design identifier.');
      const manifest=await api('handoffs',{method:'POST',body:{design_id:design.id,clip_id:clipId}});
      if(epoch!==s.epoch)return;
      if(manifest?.schema!=='visualizeit_artwork_handoff_v1'||manifest.design_id!==design.id||manifest.clip_id!==clipId||manifest.tracking_applied!==false||manifest.native_import_verified!==false)throw Error('Server returned an unexpected handoff format.');
      const current=camera.previewState?.().pendingRevision===art.pendingRevision&&camera.previewState?.().artworkSource==='recipe';
      text('pipeline-save-status',current
        ?`Saved “${name}” with a manual-import handoff${clipId?' for the chosen clip':''}. Not applied to tracking.`
        :`Saved an earlier version of the artwork. Your current artwork has newer, unsaved changes. Not applied to tracking.`);
    }catch(error){if(error.stale||epoch!==s.epoch)return;if(error.unauthorized){unauthorized();return;}text('pipeline-save-status',`${error.message} Nothing was applied to tracking.`,true);}
    finally{if(epoch===s.epoch){s.busy='';syncButtons();}}
  }

  $('open-pipeline').addEventListener('click',()=>{s.opener=$('camera-menu-toggle');renderStatus();if(!dialog.open)dialog.showModal();(s.token?$('pipeline-refresh'):$('pipeline-token')).focus({preventScroll:true});});
  $('pipeline-close').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('close',()=>{invalidate();syncButtons();if(s.busy==='')text('pipeline-message',s.token?'Paired. Refresh to check the workbench.':'');s.opener?.focus?.({preventScroll:true});s.opener=null;});
  $('pipeline-pair').addEventListener('submit',event=>{
    event.preventDefault();const input=$('pipeline-token'),token=input.value.trim();input.value='';
    if(!/^[A-Za-z0-9_-]{16,128}$/.test(token)){text('pipeline-message','Paste the startup token printed by the studio server.',true);input.focus();return;}
    invalidate();s.token=token;s.status=null;s.clips=[];renderStatus();refresh();
  });
  $('pipeline-refresh').addEventListener('click',refresh);
  $('pipeline-disconnect').addEventListener('click',()=>{disconnect('Disconnected. Token cleared from memory.');$('pipeline-token').focus({preventScroll:true});});
  $('pipeline-save').addEventListener('click',save);
  $('pipeline-clip').addEventListener('change',()=>text('pipeline-save-status',''));
  // Stop camera, hiding or leaving the page clears the token and cancels requests.
  $('stop')?.addEventListener('click',()=>{if(s.token||s.controllers.size)disconnect('Camera stopped. Token cleared.');});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){if(dialog.open)dialog.close();disconnect('');}});
  root.addEventListener('pagehide',()=>disconnect(''));
  renderStatus();
  root.PhonePipeline=Object.freeze({paired:()=>!!s.token,state:()=>s.status?.state??null,busy:()=>s.busy,disconnect:()=>disconnect('Disconnected. Token cleared from memory.')});
})(typeof globalThis!=='undefined'?globalThis:this);
