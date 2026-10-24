(function () {
  'use strict';
  const E=window.DesignEngine,$=id=>document.getElementById(id);
  const art=$('artwork'),preview=$('preview');
  const state={recipe:E.normalizeRecipe({...E.DEFAULT,colors:[...E.DEFAULT.colors]}),prompt:$('prompt').value,revision:0,token:'',epoch:0,connected:false,saved:null,manifest:null,designs:[],busy:new Set(),loadSequence:0,photoSequence:0,photo:null,placement:{x:.5,y:.5,size:.55,rotation:0,opacity:.9},promptDraft:false,controlsValid:true};
  function message(id,text,error=false){const el=$(id);el.textContent=text;el.classList.toggle('error',error);}
  function touch(){state.revision++;state.manifest=null;message('handoff-status','');updateActions();}
  function updateActions(){
    const paired=state.connected,valid=state.controlsValid&&!state.promptDraft;
    $('save').disabled=!paired||!valid||state.busy.has('save')||state.busy.has('load')||state.busy.has('handoff');
    $('save').textContent=state.busy.has('save')?'Saving…':'Save design';
    $('refresh').disabled=!paired||state.busy.has('refresh');
    $('disconnect').disabled=!state.token;
    const current=paired&&valid&&state.saved&&state.saved.revision===state.revision;
    $('handoff').disabled=!current||state.busy.has('handoff')||state.busy.has('save')||state.busy.has('load');
    $('handoff').textContent=state.busy.has('handoff')?'Saving handoff…':'Save handoff';
    $('download-manifest').disabled=!current||!state.manifest;
    $('download-native').disabled=!current||!state.manifest||state.busy.has('native');
    $('clip').disabled=!paired||state.busy.has('handoff');
    $('export').disabled=!state.controlsValid;
    $('open-camera').disabled=!valid;
    if(!state.saved)message('saved-status','Save an immutable copy to the local studio server.');
    else if(state.saved.revision!==state.revision)message('saved-status','You have unsaved changes. The saved copy has been kept.');
    else message('saved-status',`Saved: ${state.saved.metadata.name}`);
    if(!state.manifest&&!state.busy.has('handoff')&&!$('handoff-status').classList.contains('error'))message('handoff-status',current?'Current artwork is saved. You can prepare a manual import.':'Save the current design to prepare a handoff.');
    for(const button of $('design-list').querySelectorAll('button'))button.disabled=!paired||state.busy.has('load');
  }
  function recipeDescription(){const r=state.recipe;return `${r.pattern} · ${r.colors.join(' / ')} · scale ${r.scale.toFixed(2)} · rotation ${r.rotation_deg}° · seed ${r.seed}${r.text?` · text “${r.text}”`:''}`;}
  function syncControls(){
    const r=state.recipe;$('pattern').value=r.pattern;$('design-text').value=r.text;$('scale').value=r.scale;$('rotation').value=r.rotation_deg;$('seed').value=r.seed;$('extra-colors').value=r.colors.slice(2).join(', ');
    const container=$('color-inputs');container.replaceChildren();
    r.colors.slice(0,2).forEach((color,index)=>{const label=document.createElement('label');label.className='color-control';const input=document.createElement('input');input.type='color';input.value=color;input.id=`color-${index}`;input.setAttribute('aria-label',`Palette color ${index+1}`);const code=document.createElement('span');code.textContent=color;input.addEventListener('input',()=>{code.textContent=input.value;controlsChanged();});label.append(input,code);container.append(label);});
    updateRecipeLabels();
  }
  function updateRecipeLabels(){const r=state.recipe;$('scale-value').textContent=`${r.scale.toFixed(2)}×`;$('rotation-value').textContent=`${r.rotation_deg}°`;$('recipe-summary').textContent=`Current recipe: ${recipeDescription()}`;$('artwork-summary').textContent=`${r.pattern[0].toUpperCase()+r.pattern.slice(1)} / ${r.colors.length} colors`;$('preview-mode').textContent=state.photo?'Manual 2D placement':`${r.width} × ${r.height} · PNG`;preview.setAttribute('aria-label',state.photo?'Manual artwork placement over your local photo. Use arrow keys to move artwork.':`Artwork: ${recipeDescription()}`);}
  function drawPreview(){
    const ctx=preview.getContext('2d');preview.width=1024;preview.height=1024;ctx.clearRect(0,0,1024,1024);
    if(!state.photo){const fit=Math.min(1024/art.width,1024/art.height),w=art.width*fit,h=art.height*fit;ctx.fillStyle='#e4e6dc';ctx.fillRect(0,0,1024,1024);ctx.drawImage(art,(1024-w)/2,(1024-h)/2,w,h);return;}
    ctx.fillStyle='#e4e6dc';ctx.fillRect(0,0,1024,1024);
    const photo=state.photo,fit=Math.min(1024/photo.width,1024/photo.height),w=photo.width*fit,h=photo.height*fit;
    ctx.drawImage(photo,(1024-w)/2,(1024-h)/2,w,h);
    const p=state.placement,width=p.size*1024,height=width*art.height/art.width;
    ctx.save();ctx.translate(p.x*1024,p.y*1024);ctx.rotate(p.rotation*Math.PI/180);ctx.globalAlpha=p.opacity;ctx.drawImage(art,-width/2,-height/2,width,height);ctx.restore();
  }
  function render(){E.draw(art,state.recipe);drawPreview();updateRecipeLabels();}
  function parse(){
    try{
      const parsed=E.parsePrompt($('prompt').value,state.recipe);state.recipe=parsed.recipe;state.prompt=$('prompt').value;state.promptDraft=false;state.controlsValid=true;
      message('control-error','');$('parse-feedback').replaceChildren();
      for(const warning of parsed.warnings){const p=document.createElement('p');p.textContent=warning;$('parse-feedback').append(p);}
      syncControls();render();touch();message('local-status',parsed.warnings.length?'Artwork generated. Review the recipe notes below.':'Artwork generated locally. Ready to refine.');
    }catch(error){message('local-status',error.message,true);}
  }
  function controlsChanged(){
    try{
      const extra=$('extra-colors').value.trim();const colors=[$('color-0').value,$('color-1').value,...(extra?extra.split(',').map(c=>c.trim()):[])];
      const next=E.normalizeRecipe({...state.recipe,pattern:$('pattern').value,colors,text:$('design-text').value,scale:Number($('scale').value),rotation_deg:Number($('rotation').value),seed:$('seed').value.trim()===''?NaN:Number($('seed').value)});
      state.recipe=next;state.controlsValid=true;message('control-error','');render();touch();message('local-status','Artwork updated.');
    }catch(error){state.controlsValid=false;touch();message('control-error',`${error.message} Preview keeps the last valid artwork.`,true);updateActions();}
  }
  $('prompt-form').addEventListener('submit',event=>{event.preventDefault();parse();});
  $('prompt').addEventListener('input',()=>{state.promptDraft=$('prompt').value!==state.prompt;touch();message('local-status',state.promptDraft?'Prompt changed. Generate to apply it to the artwork.':'Current prompt matches the artwork.');});
  for(const id of ['pattern','design-text','scale','rotation','seed','extra-colors'])$(id).addEventListener('input',controlsChanged);
  $('new-seed').addEventListener('click',()=>{const data=new Uint32Array(1);if(window.crypto&&crypto.getRandomValues)crypto.getRandomValues(data);else data[0]=Math.floor(Math.random()*4294967296);$('seed').value=data[0];controlsChanged();});
  $('design-name').addEventListener('input',touch);
  function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();window.setTimeout(()=>URL.revokeObjectURL(url),30000);}
  function safeName(name){return name.replace(/[^\p{L}\p{N}_-]+/gu,'-').slice(0,70)||'artwork';}
  function artworkBlob(){return new Promise((resolve,reject)=>art.toBlob(blob=>blob?resolve(blob):reject(new Error('PNG export failed.')),'image/png'));}
  $('export').addEventListener('click',async()=>{try{download(await artworkBlob(),`${safeName($('design-name').value)}.png`);message('local-status','Artwork PNG download prepared. Your photo is excluded.');}catch(error){message('local-status',error.message,true);}});
  $('open-camera').addEventListener('click',()=>{
    try{
      if(!state.controlsValid||state.promptDraft)throw new Error('Generate the current prompt and correct invalid controls first.');
      sessionStorage.setItem('visualizeit_camera_recipe_v1',JSON.stringify({recipe:E.normalizeRecipe(state.recipe),prompt:state.prompt}));
      location.assign('/camera');
    }catch(error){message('local-status',`Camera handoff failed: ${error.message}`,true);}
  });
  async function api(path,{method='GET',body,blob=false}={}){
    if(!state.token)throw new Error('Pair with your local studio server first.');
    const epoch=state.epoch,token=state.token,controller=new AbortController(),timeout=window.setTimeout(()=>controller.abort(),15000);
    try{
      const response=await fetch(`/api/studio/${path}`,{method,headers:{Authorization:`Bearer ${token}`,...(body?{'Content-Type':'application/json'}:{})},...(body?{body:JSON.stringify(body)}:{}),signal:controller.signal,credentials:'same-origin',cache:'no-store',redirect:'error'});
      if(epoch!==state.epoch)throw new Error('Pairing changed. Try again with the current connection.');
      if(!response.ok){let detail;try{detail=await response.json();}catch{}throw new Error(detail?.error?.message||`Studio request failed (${response.status}).`);}
      const result=blob?await response.blob():await response.json();
      if(epoch!==state.epoch)throw new Error('Pairing changed. Try again with the current connection.');
      return result;
    }catch(error){if(error.name==='AbortError')throw new Error('Studio server timed out. Your local edits are kept.');throw error;}finally{window.clearTimeout(timeout);}
  }
  function validId(id){if(typeof id!=='string'||!/^[A-Za-z0-9_-]{1,128}$/.test(id))throw new Error('Server returned an invalid design identifier.');return encodeURIComponent(id);}
  function renderLibrary(){
    const list=$('design-list');list.replaceChildren();
    if(!state.designs.length){const li=document.createElement('li');li.className='empty';li.textContent='No saved designs yet. Save your first study.';list.append(li);return;}
    for(const design of state.designs){const li=document.createElement('li');li.className='design-item';const info=document.createElement('div'),title=document.createElement('strong'),date=document.createElement('small'),button=document.createElement('button');title.textContent=design.name;const timestamp=new Date(design.created_at);date.textContent=Number.isNaN(timestamp.valueOf())?'Saved design':timestamp.toLocaleString();info.append(title,date);button.type='button';button.textContent='Load';button.setAttribute('aria-label',`Load ${design.name}`);button.addEventListener('click',()=>loadDesign(design.id));li.append(info,button);list.append(li);}
    updateActions();
  }
  function showClips(result){
    const select=$('clip'),previous=select.value;
    const clips=result.state==='connected'?(result.clips||[]):[];
    const next=clips.some(clip=>clip.clip_id===previous)?previous:'';
    select.replaceChildren(new Option('No clip · artwork only',''));
    for(const clip of clips)select.append(new Option(clip.label,clip.clip_id));
    select.value=next;
    if(next!==previous){state.manifest=null;message('handoff-status','');}
    message('workbench-status',`Workbench ${result.state==='connected'?'connected':'disconnected'}: ${result.message||'No further information.'} Custom artwork tracking is unsupported.`);
    updateActions();
  }
  async function refresh(){
    const epoch=state.epoch;state.busy.add('refresh');updateActions();message('server-status','Refreshing saved designs and workbench clips…');
    try{const [library,clips]=await Promise.all([api('designs'),api('clips')]);if(epoch!==state.epoch)return;state.designs=library.designs;renderLibrary();showClips(clips);message('server-status',`Studio connected. ${state.designs.length} saved design${state.designs.length===1?'':'s'}.`);}catch(error){if(epoch===state.epoch)message('server-status',`${error.message} Current artwork is kept.`,true);}finally{if(epoch===state.epoch){state.busy.delete('refresh');updateActions();}}
  }
  async function connect(token){
    state.epoch++;state.busy.clear();const epoch=state.epoch;state.token=token.trim();state.connected=false;state.saved=null;state.manifest=null;state.designs=[];$('token').value='';$('connection-badge').textContent='Connecting…';updateActions();message('server-status','Connecting to your local studio server…');
    try{const status=await api('status');if(epoch!==state.epoch)return;if(status.schema!=='visualizeit_studio_status_v1')throw new Error('This server does not provide the expected studio API.');state.connected=true;$('connection-badge').textContent='Studio paired';updateActions();await refresh();}catch(error){if(epoch!==state.epoch)return;state.connected=false;$('connection-badge').textContent='Local mode';message('server-status',`${error.message} You can still generate and download locally.`,true);updateActions();}
  }
  $('pair-form').addEventListener('submit',event=>{event.preventDefault();const token=$('token').value.trim();if(!token){message('server-status','Paste the startup token printed by your local server.',true);$('token').focus();return;}connect(token);});
  $('disconnect').addEventListener('click',()=>{state.epoch++;state.busy.clear();state.token='';state.connected=false;state.saved=null;state.manifest=null;state.designs=[];$('token').value='';$('connection-badge').textContent='Local mode';$('design-list').replaceChildren();const li=document.createElement('li');li.className='empty';li.textContent='Pair with the server to see saved designs.';$('design-list').append(li);showClips({state:'disconnected',message:'Studio is not paired.'});message('server-status','Disconnected. Token cleared from memory; your artwork is kept.');updateActions();});
  $('refresh').addEventListener('click',refresh);
  $('save').addEventListener('click',async()=>{
    const epoch=state.epoch,revision=state.revision;state.busy.add('save');updateActions();message('server-status','Saving the current artwork…');
    try{
      const name=$('design-name').value.trim();if(!name||Array.from(name).length>80)throw new Error('Give your design a name between 1 and 80 characters.');
      const body={schema:'visualizeit_design_v1',name,prompt:state.prompt,recipe:E.normalizeRecipe(state.recipe),png_base64:art.toDataURL('image/png').split(',')[1]};
      const metadata=await api('designs',{method:'POST',body});if(epoch!==state.epoch)return;validId(metadata.id);state.saved={metadata,revision};state.manifest=null;state.designs=[metadata,...state.designs.filter(d=>d.id!==metadata.id)];renderLibrary();message('server-status',revision===state.revision?`Saved “${metadata.name}” to the studio server.`:`Saved “${metadata.name}”. Changes made during saving are still unsaved.`);
    }catch(error){if(epoch===state.epoch)message('server-status',`${error.message} Your artwork is kept.`,true);}finally{if(epoch===state.epoch){state.busy.delete('save');updateActions();}}
  });
  async function decodeImage(blob,maxPixels=24_000_000){
    const dimensions=await imageDimensions(blob);
    if(dimensions.width<1||dimensions.height<1||dimensions.width*dimensions.height>maxPixels)throw new Error(`Image is too large. Maximum ${Math.round(maxPixels/1_000_000)} megapixels.`);
    const url=URL.createObjectURL(blob);
    try{const image=new Image();image.src=url;await image.decode();return image;}finally{URL.revokeObjectURL(url);}
  }
  async function imageDimensions(blob){
    // Inspect dimensions before decoding to avoid allocating an unbounded image.
    const bytes=new Uint8Array(await blob.slice(0,262144).arrayBuffer()),view=new DataView(bytes.buffer);
    if(bytes.length>=24&&bytes[0]===137&&bytes[1]===80&&bytes[2]===78&&bytes[3]===71)return {width:view.getUint32(16),height:view.getUint32(20)};
    if(bytes.length>=12&&bytes[0]===255&&bytes[1]===216){let i=2;while(i+9<bytes.length){if(bytes[i]!==255){i++;continue;}const marker=bytes[i+1];if(marker===0xd8||marker===0xd9){i+=2;continue;}const length=view.getUint16(i+2);if(length<2)break;if([0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf].includes(marker))return {height:view.getUint16(i+5),width:view.getUint16(i+7)};i+=2+length;}}
    if(bytes.length>=30&&String.fromCharCode(...bytes.slice(0,4))==='RIFF'&&String.fromCharCode(...bytes.slice(8,12))==='WEBP'){
      const kind=String.fromCharCode(...bytes.slice(12,16));
      if(kind==='VP8X')return {width:1+bytes[24]+(bytes[25]<<8)+(bytes[26]<<16),height:1+bytes[27]+(bytes[28]<<8)+(bytes[29]<<16)};
      if(kind==='VP8 '&&bytes[23]===157&&bytes[24]===1&&bytes[25]===42)return {width:view.getUint16(26,true)&16383,height:view.getUint16(28,true)&16383};
      if(kind==='VP8L'&&bytes[20]===47)return {width:1+bytes[21]+((bytes[22]&63)<<8),height:1+(bytes[22]>>6)+(bytes[23]<<2)+((bytes[24]&15)<<10)};
    }
    throw new Error('Could not read image dimensions safely. Choose a standard JPEG, PNG or WebP image.');
  }
  async function loadDesign(id){
    const epoch=state.epoch,revision=state.revision,sequence=++state.loadSequence;state.busy.add('load');updateActions();message('server-status','Loading the saved artwork…');
    try{const metadata=await api(`designs/${validId(id)}`),recipe=E.normalizeRecipe(metadata.recipe);const blob=await api(`designs/${validId(id)}/image.png`,{blob:true});if(blob.size>2097152)throw new Error('Saved PNG exceeds the studio size limit.');const image=await decodeImage(blob,1048576);if(epoch!==state.epoch||sequence!==state.loadSequence)return;if(revision!==state.revision)throw new Error('Artwork changed while loading. Click Load again when ready to replace it.');if(image.width!==recipe.width||image.height!==recipe.height)throw new Error('Saved image dimensions do not match the recipe.');state.recipe=recipe;state.prompt=metadata.prompt;state.promptDraft=false;state.controlsValid=true;$('prompt').value=metadata.prompt;$('design-name').value=metadata.name;$('parse-feedback').replaceChildren();message('control-error','');art.width=recipe.width;art.height=recipe.height;art.getContext('2d').drawImage(image,0,0);state.revision++;state.saved={metadata,revision:state.revision};state.manifest=null;syncControls();drawPreview();message('server-status',`Loaded “${metadata.name}” and its original saved PNG.`);message('local-status','Saved artwork loaded. Further control edits redraw it locally.');}catch(error){if(epoch===state.epoch)message('server-status',`${error.message} Current artwork is kept.`,true);}finally{if(epoch===state.epoch&&sequence===state.loadSequence){state.busy.delete('load');updateActions();}}
  }
  $('clip').addEventListener('change',()=>{state.manifest=null;updateActions();});
  $('handoff').addEventListener('click',async()=>{
    const epoch=state.epoch,revision=state.revision,saved=state.saved,clipId=$('clip').value||null;if(!saved||saved.revision!==revision)return;state.busy.add('handoff');updateActions();message('handoff-status','Saving a manual-import handoff…');
    try{const manifest=await api('handoffs',{method:'POST',body:{design_id:saved.metadata.id,clip_id:clipId}});if(epoch!==state.epoch)return;if(manifest.schema!=='visualizeit_artwork_handoff_v1'||manifest.design_id!==saved.metadata.id||manifest.clip_id!==clipId||manifest.tracking_applied!==false||manifest.native_import_verified!==false)throw new Error('Server returned an unexpected handoff format.');if(revision!==state.revision||state.saved?.metadata.id!==saved.metadata.id){message('server-status','Handoff saved for the earlier copy. Save the current edits to prepare a new handoff.');return;}if(($('clip').value||null)!==clipId){message('server-status','Handoff saved for the earlier clip selection. Prepare a new handoff for the current selection.');return;}state.manifest=manifest;message('handoff-status','Handoff saved. Download the PNG for manual import. Native import remains unverified; tracking has not been applied.');}catch(error){if(epoch===state.epoch)message('handoff-status',error.message,true);}finally{if(epoch===state.epoch){state.busy.delete('handoff');updateActions();}}
  });
  $('download-manifest').addEventListener('click',()=>{if(!state.manifest)return;download(new Blob([JSON.stringify(state.manifest,null,2)],{type:'application/json'}),`handoff-${safeName(state.manifest.id)}.json`);});
  $('download-native').addEventListener('click',async()=>{const saved=state.saved,epoch=state.epoch;if(!saved||!state.manifest)return;state.busy.add('native');updateActions();try{const blob=await api(`designs/${validId(saved.metadata.id)}/image.png`,{blob:true});if(epoch!==state.epoch)return;download(blob,`${safeName(saved.metadata.name)}-native.png`);message('handoff-status','Saved artwork PNG download prepared. Choose it in the native image picker. Native import remains unverified.');}catch(error){if(epoch===state.epoch)message('handoff-status',error.message,true);}finally{if(epoch===state.epoch){state.busy.delete('native');updateActions();}}});
  function placementUI(){const p=state.placement;$('placement-size').value=Math.round(p.size*100);$('placement-opacity').value=Math.round(p.opacity*100);$('placement-rotation').value=p.rotation;$('placement-size-value').textContent=`${Math.round(p.size*100)}%`;$('placement-opacity-value').textContent=`${Math.round(p.opacity*100)}%`;$('placement-rotation-value').textContent=`${p.rotation}°`;drawPreview();}
  function move(dx,dy){state.placement.x=Math.max(0,Math.min(1,state.placement.x+dx));state.placement.y=Math.max(0,Math.min(1,state.placement.y+dy));drawPreview();}
  for(const button of document.querySelectorAll('[data-move]'))button.addEventListener('click',()=>move(...button.dataset.move.split(',').map(Number)));
  $('center-placement').addEventListener('click',()=>{state.placement.x=.5;state.placement.y=.5;drawPreview();});
  for(const [id,key,factor]of[['placement-size','size',.01],['placement-opacity','opacity',.01],['placement-rotation','rotation',1]])$(id).addEventListener('input',()=>{state.placement[key]=Number($(id).value)*factor;placementUI();});
  function photoState(){const active=!!state.photo;$('placement-controls').hidden=!active;preview.classList.toggle('placement-active',active);if(active){preview.tabIndex=0;preview.setAttribute('aria-describedby','placement-help');}else{preview.removeAttribute('tabindex');preview.removeAttribute('aria-describedby');}updateRecipeLabels();drawPreview();}
  $('photo').addEventListener('change',async()=>{
    const file=$('photo').files[0],sequence=++state.photoSequence;$('photo').value='';if(!file)return;
    try{if(!['image/jpeg','image/png','image/webp'].includes(file.type))throw new Error('Choose a JPEG, PNG or WebP photo.');if(file.size>12*1024*1024)throw new Error('Photo is too large. Choose a file up to 12 MB.');message('local-status','Opening your photo locally…');const image=await decodeImage(file);if(sequence!==state.photoSequence)return;state.photo=image;state.placement={x:.5,y:.5,size:.55,rotation:0,opacity:.9};placementUI();photoState();message('local-status','Photo opened in memory. Placement is manual, flat 2D; no tracking is running.');}catch(error){if(sequence===state.photoSequence)message('local-status',error.message,true);}
  });
  $('clear-photo').addEventListener('click',()=>{state.photoSequence++;state.photo=null;photoState();message('local-status','Photo cleared from this studio. The artwork is kept.');});
  let drag=null;
  preview.addEventListener('pointerdown',event=>{if(!state.photo||event.button!==0)return;const box=preview.getBoundingClientRect();drag={id:event.pointerId,x:event.clientX,y:event.clientY,px:state.placement.x,py:state.placement.y,width:box.width,height:box.height};preview.setPointerCapture(event.pointerId);preview.focus();});
  preview.addEventListener('pointermove',event=>{if(!drag||drag.id!==event.pointerId)return;state.placement.x=Math.max(0,Math.min(1,drag.px+(event.clientX-drag.x)/drag.width));state.placement.y=Math.max(0,Math.min(1,drag.py+(event.clientY-drag.y)/drag.height));drawPreview();});
  for(const name of ['pointerup','pointercancel','lostpointercapture'])preview.addEventListener(name,()=>{drag=null;});
  preview.addEventListener('keydown',event=>{if(!state.photo)return;const step=event.shiftKey?.05:.01,directions={ArrowUp:[0,-step],ArrowDown:[0,step],ArrowLeft:[-step,0],ArrowRight:[step,0]};if(directions[event.key]){event.preventDefault();move(...directions[event.key]);}});
  syncControls();render();updateActions();
  // Consume once, before parsing. Only artwork recipes cross pages; pairing,
  // camera images, photos and placement state are never part of this transfer.
  try{
    const raw=sessionStorage.getItem('visualizeit_studio_return_v1');
    if(raw!==null){
      sessionStorage.removeItem('visualizeit_studio_return_v1');
      if(raw.length>8192)throw new Error('Artwork transfer is too large.');
      const transfer=JSON.parse(raw);
      if(!transfer||Array.isArray(transfer)||transfer.schema!=='visualizeit_studio_return_v1'||Object.keys(transfer).length!==3||typeof transfer.prompt!=='string'||Array.from(transfer.prompt).length>500)throw new Error('Artwork transfer format is invalid.');
      const restored=E.normalizeRecipe(transfer.recipe);
      state.recipe=restored;state.prompt=transfer.prompt;state.promptDraft=false;state.controlsValid=true;
      $('prompt').value=state.prompt;syncControls();render();touch();
      message('local-status','Camera artwork restored. Download its PNG, or pair this studio to save it.');
      message('saved-status','Unsaved artwork from the camera. Save a copy to keep it.');
    }
  }catch(error){message('local-status',`Artwork transfer unavailable: ${error.message}`,true);}
  // Fragment credentials never reach the API URL, storage, image src or DOM text.
  const fragment=new URLSearchParams(location.hash.slice(1)),pairToken=fragment.get('token');
  if(pairToken!==null){try{history.replaceState(null,'',location.pathname+location.search);}catch{}if(pairToken.trim())connect(pairToken);else message('server-status','The pairing URL has an empty token. Paste the server token below.',true);}
})();
