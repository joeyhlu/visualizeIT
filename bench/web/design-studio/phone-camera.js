(function(root){
  'use strict';
  // ---------------------------------------------------------------- worker client (ABI unchanged)
  class PhoneSegmenterClient {
    constructor(){this.worker=null;this.pending=null;this.serial=0;this.ready=false;}
    initialize(epoch){this.close();this.worker=new Worker('/phone-segment-worker.js');const worker=this.worker;worker.onmessage=({data})=>{if(this.worker!==worker)return;const pending=this.pending;if(!pending||data.id!==pending.id||data.epoch!==pending.epoch)return;clearTimeout(pending.timer);this.pending=null;if(data.type==='error')pending.reject(Error(data.error));else{if(data.type==='ready')this.ready=true;pending.resolve(data);}};worker.onerror=()=>{if(this.worker===worker)this.fail(Error('The local selection worker could not run.'));};return this.request('initialize',{epoch},60000);}
    request(type,payload,timeout=20000,transfer=[]){if(!this.worker)return Promise.reject(Error('Automatic selection is unavailable.'));if(this.pending)return Promise.reject(Error('A selection request is already running.'));return new Promise((resolve,reject)=>{const id=++this.serial;const timer=setTimeout(()=>this.fail(Error('Local selection timed out. Clear the selection to retry.')),timeout);this.pending={id,epoch:payload.epoch,timer,resolve,reject};try{this.worker.postMessage({id,type,...payload},transfer);}catch(error){this.fail(error);}});}
    process(image,tag,strokes){return this.request('process',{image,tag,epoch:tag.epoch,strokes},20000,[image]);}
    reset(epoch){this.close();return this.initialize(epoch);}
    fail(error){const pending=this.pending;this.pending=null;if(pending){clearTimeout(pending.timer);pending.reject(error);}this.worker?.terminate();this.worker=null;this.ready=false;}
    close(){this.fail(Error('Selection cancelled.'));}
  }
  root.PhoneSegmenterClient=PhoneSegmenterClient;
  if(typeof module==='object'&&module.exports)module.exports={PhoneSegmenterClient};
  if(typeof document==='undefined')return;

  // ---------------------------------------------------------------- page state
  const G=root.VisualizeITGlass;
  const $=id=>document.getElementById(id),body=document.body,state=new PhoneSelection.Session(),client=new PhoneSegmenterClient();
  const video=$('camera'),scene=$('scene'),ctx=scene.getContext('2d'),artwork=$('artwork-preview'),stage=document.querySelector('.camera-stage');
  let source=document.createElement('canvas'),staging=document.createElement('canvas');
  const snapshot=document.createElement('canvas'),tracking=document.createElement('canvas'),overlay=document.createElement('canvas'),maskCanvas=document.createElement('canvas'),candidate=document.createElement('canvas'),outline=document.createElement('canvas'),appliedArtwork=document.createElement('canvas');
  const trackingContext=tracking.getContext('2d',{willReadFrequently:true});
  const tracker=root.PhoneTracker?.Tracker?new root.PhoneTracker.Tracker():null;
  const FRESH_MS=250,HISTORY_LIMIT=6,TRACE_LIMIT=32;
  // Copies of the unchanged gates (controller + phone-tracker.js), reported with diagnostics only.
  const QUALITY_GATES=Object.freeze({freshMs:FRESH_MS,trackerFrameGapMs:750,sourceMaxEdge:640,trackerMaxEdge:192,minFeatures:8,minSurvivors:8,
    inlierFractionOfMasked:.5,minInliers:8,inlierRadiusPx:1.5,minOccupiedQuadrants:3,minSpatialSpanFraction:.25,
    featureVarianceMin:100,featureEigenMin:25,matchSsdMax:225,matchRatioMax:.75,matchMarginMin:25,forwardBackwardMaxPx:1.5,anchorMaxPx:2,
    startupWatchdogMs:1500,tickIntervalMs:66});
  let stream=null,animation=null,busy=false,facing='environment',runtime='idle',hover={x:.5,y:.5},sourceTimestamp=0,sourceIdentity=null,lastTick=0,sourceOrientation='',trackingResult=null,trackingLost=false,overlayReady=false,candidateReady=false,cameraStalled=false;
  let decodedCallback=null,decodedSerial=0,decodedArrival=0,useDecodedCallback=false,counterSupported=false;
  let cameraGeneration=0;
  let renderMode='raw',cameraStarted=0,acceptedFrames=0,rejectedFrame='not-started',captureError=null,selectionError=null;
  let revisionSerial=0,pendingRevision=0,appliedRevision=0,appliedRecipe=null,recipePrompt='',appliedPrompt='';
  // Selection: an observed candidate mask is shown as an outline only; artwork needs explicit "Use object" + Apply.
  let confirmed=false,appliedByUser=false,menuOpen=false;
  // Artwork source of the pending (`artworkSource`) and applied (`appliedSource`) canvases.
  let artworkSource='recipe',appliedSource='recipe',rasterName='',appliedRasterName='';
  // Persistent selection evidence: survives frame resumption and artwork preview until a new tap or reset.
  let issue=null,failureSnapshot=null,statusText='',statusError=false;
  // Segmentation: idle | pending | nonempty | rejected. Tap marker acknowledges a tap; it is not an outline.
  const segmentation={state:'idle',reason:null,startedAt:0,latencyMs:null,pixels:0,requests:0};
  let tapMarker=null;
  const timings={captureMs:null,readbackMs:null,trackerMs:null,renderMs:null,segmentationMs:null};
  // In-memory only, numeric/short-state entries. No images, masks, device IDs, storage or network.
  const traceLog=[];
  const inspector={open:false,mode:'rotate',rotationX:0,rotationY:0,scale:1,panX:0,panY:0},pointers=new Map();
  let recipe={...DesignEngine.DEFAULT,colors:[...DesignEngine.DEFAULT.colors]},placement={x:0,y:0,scale:100,rotation:0,opacity:80};
  const imports=new G.ImportGate(new G.WorkToken(),{document});
  const animator=new G.EntranceAnimator($('artwork-entrance'),{window:root,onSettled:()=>ui()});
  let releaseKeyboard=G.bindKeyboardInset(root,document.documentElement);

  // ---------------------------------------------------------------- surfaces (exactly one at a time)
  // camera | chooser | chat | generate | result. Analyze "held" is a temporary flag, not a surface.
  const SURFACES=Object.freeze({chooser:'chooser-sheet',chat:'chat-surface',generate:'generate-surface',result:'artwork-inspector'});
  const FOCUSABLE='button,[href],textarea,input:not([type="hidden"]),select,[tabindex]:not([tabindex="-1"])';
  let surface='camera',surfaceOrigin=null,generateReturn='camera',dragged=false;
  let analyzeHeld=false,analyzeSource=null,analyzePointer=null,analyzeKey=null,lastAnalyzePress=0;

  // Detailed wording for the unchanged tracker's codes. Shown only in diagnostics, never on the camera.
  const TRACKER_REASONS={
    'insufficient-features':'too little visible texture to follow (clear, glossy or plain surfaces often fail)',
    'insufficient-survivors':'the tracked points were lost (fast motion, blur, or a plain or see-through surface)',
    'insufficient-mask-support':'too few trackable points inside the outline (little texture, reflections or transparency)',
    'insufficient-inliers':'points on the object moved inconsistently (reflections, transparency or motion)',
    'insufficient-spatial-support':'trackable points cover only part of the object',
    'frame-gap':'camera frames paused for too long',
    'nonincreasing-time':'camera frames arrived out of order',
    'dimension-change':'the camera size or orientation changed',
    'invalid-input':'a camera frame could not be read',
    'empty-mask':'the selection outline was empty',
    'invalid-mask':'the selection outline was malformed',
    'mask-not-registered':'the outline did not belong to the current tap frame',
    'not-started':'tracking had not started',
    'tracker-start-failed':'tracking could not start on the tap frame',
  };
  const humanReason=reason=>TRACKER_REASONS[reason]||`tracker reported “${reason||'unknown'}”`;
  const ms=value=>Math.round(value*100)/100;
  // Short on-camera copy. The raw reason and detail live in diagnostics.
  const SHORT={tracking:'Tracking lost. Analyze again.',frames:'Camera paused. Analyze again.',
    segmentation:'No outline found. Analyze again.',initialization:'Selection unavailable. See settings.'};

  function trace(event,data){
    const entry={t:cameraStarted?Math.round(performance.now()-cameraStarted):null,e:event};
    if(data)for(const [key,value] of Object.entries(data))if(value===null||typeof value==='number'||typeof value==='boolean'||(typeof value==='string'&&value.length<=48))entry[key]=value;
    traceLog.push(entry);if(traceLog.length>TRACE_LIMIT)traceLog.shift();
  }
  function trackerNumbers(){
    const d=trackingResult?.diagnostics;if(!d)return null;
    const out={};for(const [key,value] of Object.entries(d))if(value===null||typeof value==='number')out[key]=value;return out;
  }
  function captureFailure(kind,reason){
    // Freeze root evidence of the FIRST failure of this selection; later frames cannot erase it.
    if(failureSnapshot)return;
    const now=performance.now();
    failureSnapshot=JSON.parse(JSON.stringify({at:new Date().toISOString(),kind,reason,renderMode,runtime,
      frameAgeMs:sourceTimestamp?Math.round(now-sourceTimestamp):null,acceptedFrames,rejectedFrame,
      selectionFrame:state.frame?{frameId:state.frame.frameId,width:state.frame.width,height:state.frame.height}:null,
      segmentation:{state:segmentation.state,reason:segmentation.reason,latencyMs:segmentation.latencyMs,
        elapsedMs:segmentation.startedAt?Math.round(now-segmentation.startedAt):null,pixels:segmentation.pixels},
      tracker:trackingResult?{valid:!!trackingResult.valid,reason:trackingResult.reason??null,diagnostics:trackerNumbers()}:null,
      timings:{...timings},gates:QUALITY_GATES,trace:traceLog.slice()}));
  }

  function renderStatus(){const el=$('camera-status');el.textContent=issue?issue.short:statusText;el.classList.toggle('error',!!issue||statusError);}
  function status(text,error=false){statusText=text;statusError=error;renderStatus();}
  function setRuntime(text){$('runtime-status').textContent=text;}
  function setIssue(kind,reason,detail,short){
    issue={kind,reason,short:short||SHORT[kind]||'Something went wrong. See settings.',detail:detail||humanReason(reason),at:new Date().toISOString()};
    trace('issue',{kind,reason:String(reason||'').slice(0,48)});captureFailure(kind,reason);renderStatus();
  }
  function clearIssue(){issue=null;renderStatus();}
  function orientation(){return `${screen.orientation?.angle??root.orientation??0}:${video.videoWidth}x${video.videoHeight}`;}
  function freshResult(){
    return renderMode==='synchronized'&&!!state.mask&&!!trackingResult?.valid&&trackingResult.timestamp===sourceTimestamp&&performance.now()-sourceTimestamp<=FRESH_MS&&Number.isFinite(trackingResult.dx)&&Number.isFinite(trackingResult.dy)&&!trackingLost;
  }
  function correspondence(){
    if(!state.frame||!trackingResult)return 'none';
    if(freshResult())return 'live';
    if(trackingResult.reason==='awaiting-mask'&&!trackingLost)return 'awaiting';
    if(trackingLost||!trackingResult.valid)return 'lost';
    return 'stale';
  }
  function outlineVisible(){return candidateReady&&freshResult();}
  function canConfirm(){return !busy&&!confirmed&&surface==='camera'&&freshResult();}
  function canApply(){return confirmed&&freshResult()&&!(appliedByUser&&pendingRevision===appliedRevision);}
  // "Use object" exists only while a fresh, valid candidate exists: no dead control before selection.
  function syncConfirm(){
    const button=$('confirm-object'),usable=canConfirm();
    if(button.hidden===usable)button.hidden=!usable;if(button.disabled===usable)button.disabled=!usable;
    const apply=!canApply();if($('apply-preview').disabled!==apply)$('apply-preview').disabled=apply;
  }
  function phaseText(live,starting){
    if(starting)return 'Opening…';if(!live)return 'Camera off';if(cameraStalled)return 'Paused';if(busy)return 'Analyzing…';
    if(issue)return {tracking:'Lost',frames:'Paused',segmentation:'No outline',initialization:'Unavailable'}[issue.kind]||'Check settings';
    if(confirmed&&freshResult())return appliedByUser?'Design on object':'Selected';
    if(outlineVisible())return 'Outline found';
    if(runtime==='loading')return 'Loading';
    return 'Live';
  }
  function syncLive(){
    syncConfirm();
    if(tapMarker&&outlineVisible()){tapMarker=null;$('reticle').hidden=true;}
    const live=state.phase==='live',text=phaseText(live,state.phase==='starting');
    if($('phase').textContent!==text)$('phase').textContent=text;
  }
  function placeReticle(live){
    const reticle=$('reticle'),show=live&&!!tapMarker&&surface==='camera';reticle.hidden=!show;if(!show)return;
    reticle.dataset.state=tapMarker.state;
    $('reticle-label').textContent={pending:'Analyze point',waiting:'Analyze point · loading',rejected:'Analyze point',aim:'Analyze point'}[tapMarker.state]||'Analyze point';
    const synced=renderMode==='synchronized'||!video.videoWidth,w=synced?scene.width:video.videoWidth,h=synced?scene.height:video.videoHeight;
    const rect=stage.getBoundingClientRect(),scale=Math.max(rect.width/w,rect.height/h);
    reticle.style.left=`${hover.x*w*scale-(w*scale-rect.width)/2}px`;
    reticle.style.top=`${hover.y*h*scale-(h*scale-rect.height)/2}px`;
  }
  function ui(){
    const live=state.phase==='live',starting=state.phase==='starting';
    body.dataset.phase=state.phase;body.dataset.confirmed=String(confirmed);body.dataset.surface=surface;body.dataset.inspector=inspector.open?'open':'closed';
    body.dataset.analyze=analyzeHeld?'held':'idle';body.dataset.issue=issue?issue.kind:'none';
    $('start').disabled=starting;$('start').textContent=starting?'Opening camera…':'Open camera';
    $('stop').disabled=!live&&!starting;$('switch').disabled=!live;$('retake').disabled=!live||(!state.frame&&!issue);
    $('empty-stage').hidden=live||surface!=='camera';$('placement').disabled=!freshResult();
    $('phase').textContent=phaseText(live,starting);
    $('phase').dataset.active=String(live&&!cameraStalled);scene.setAttribute('aria-busy',String(busy));
    scene.setAttribute('aria-label',live?'Live camera. Hold Analyze to outline the object in the center.':'Camera off. Open the camera to begin.');
    placeReticle(live);
    $('camera-menu').hidden=!menuOpen;$('camera-menu-toggle').setAttribute('aria-expanded',String(menuOpen));
    $('ask-open').setAttribute('aria-expanded',String(surface==='chat'));
    $('ask-open').setAttribute('aria-label',surface==='chat'?'Close idea helper':'Open idea helper');
    $('ask-button').setAttribute('aria-expanded',String(surface==='chat'));
    $('design-button').setAttribute('aria-expanded',String(surface==='chooser'));
    $('analyze').setAttribute('aria-pressed',String(analyzeHeld));
    $('show-design').hidden=surface==='result'||!pendingRevision;
    $('artwork-source-note').textContent=artworkSource==='raster'?`Imported image${rasterName?` · ${rasterName}`:''} · decoded on this device.`:'Made locally from a pattern recipe.';
    renderStatus();syncConfirm();
    if(surface==='result'){
      $('result-copy').textContent=artworkSource==='raster'?'Uploaded image · move, resize, or upload another.':'Local pattern preview · move, resize, or generate a new design.';
      $('apply-status').textContent=appliedByUser&&pendingRevision===appliedRevision?'Applied to the selected object.':canApply()?'Ready to apply to the selected object.':confirmed?'Hold the object in view to apply.':'Analyze an object, then Use object, to apply.';
    }
  }

  // ---------------------------------------------------------------- surface management
  function surfaceEl(name=surface){return SURFACES[name]?$(SURFACES[name]):null;}
  function allowedRoots(){
    // Every modal (chat included) owns all of its controls; the top bar is inert behind it.
    return [surfaceEl()].filter(Boolean);
  }
  // Background siblings of the active surface (never its ancestor) are inert while it is open.
  function applyInert(){
    const active=surfaceEl(),keep=new Set(allowedRoots());
    for(const el of $('camera-studio').children)el.inert=!!active&&!keep.has(el);
    const skip=document.querySelector('.skip');if(skip)skip.inert=!!active;
  }
  function focusable(el){return !!el&&el.isConnected&&!el.disabled&&!el.hidden&&!el.closest('[hidden],[inert]')&&el.getClientRects().length>0;}
  function surfaceControls(){return allowedRoots().flatMap(r=>[...r.querySelectorAll(FOCUSABLE)]).filter(focusable);}
  function restoreSurfaceFocus(){
    // Never refocus a text field: that would raise the keyboard over the camera.
    const preferred=surfaceOrigin&&surfaceOrigin.tagName!=='TEXTAREA'?surfaceOrigin:null;surfaceOrigin=null;
    for(const el of [preferred,$('design-button'),$('ask-button'),scene])if(focusable(el)){el.focus({preventScroll:true});return;}
  }
  function setSurface(next,focusTarget){
    if(next!=='camera'&&!SURFACES[next])throw Error('Unknown surface.');
    if(next===surface){if(focusTarget)focusTarget.focus({preventScroll:true});return;}
    if(surface==='camera'){const active=document.activeElement;surfaceOrigin=active&&active!==body?active:null;}
    if(surface==='result'){animator.cancel();releasePointers();}
    releaseAnalyze();
    surface=next;inspector.open=next==='result';menuOpen=false;
    for(const [name,id] of Object.entries(SURFACES))$(id).hidden=name!==next;
    trace('surface',{name:next});applyInert();ui();
    if(next==='camera')restoreSurfaceFocus();else if(focusTarget)focusTarget.focus({preventScroll:true});
  }
  function trapTab(event){
    const controls=surfaceControls();if(!controls.length)return;event.preventDefault();
    const index=controls.indexOf(document.activeElement);
    const next=index<0?(event.shiftKey?controls.length-1:0):(index+(event.shiftKey?-1:1)+controls.length)%controls.length;
    controls[next].focus({preventScroll:true});
  }
  document.addEventListener('focusin',event=>{
    if(surface==='camera'||event.target.closest?.('dialog'))return;
    if(!allowedRoots().some(r=>r.contains(event.target)))surfaceControls()[0]?.focus({preventScroll:true});
  });
  document.addEventListener('keydown',event=>{
    if(surface==='camera'||surface==='result'||document.querySelector('dialog[open]'))return;
    if(event.key==='Tab')trapTab(event);
    else if(event.key==='Escape'){event.preventDefault();closeSurface();}
  });
  function closeSurface(){
    if(surface==='result'){closeInspector();return;}
    if(surface==='generate'&&generateReturn==='result'){generateReturn='camera';revealArtwork('',{animate:false});return;}
    generateReturn='camera';setSurface('camera');
  }
  function openSurface(name){
    if(name==='camera'){closeSurface();return;}
    if(name==='chooser')setSurface('chooser',$('choose-generate'));
    else if(name==='chat')openChat();
    else if(name==='generate')openGenerate();
    else if(name==='result')revealArtwork('',{animate:false});
    else throw Error('Unknown surface.');
  }

  // ---------------------------------------------------------------- resources
  function setRenderMode(mode){renderMode=mode;body.dataset.renderMode=mode;scene.hidden=mode!=='synchronized';}
  function diagnostics(){
    const now=performance.now(),track=stream?.getVideoTracks()[0];
    return {renderMode,phase:state.phase,runtime,workerReady:client.ready,workerPending:!!client.pending,surface,analyzeHeld,
      standalone:!!root.navigator.standalone||!!root.matchMedia?.('(display-mode: standalone)').matches,
      video:{readyState:video.readyState,width:video.videoWidth,height:video.videoHeight,paused:video.paused},
      track:track?{readyState:track.readyState,muted:track.muted}:null,
      callbacks:decodedSerial,callbackAgeMs:decodedArrival?Math.round(now-decodedArrival):null,
      acceptedFrames,frameAgeMs:sourceTimestamp?Math.round(now-sourceTimestamp):null,
      geometry:sourceOrientation,rejectedFrame,captureError,selectionError,
      selection:{segmentation:segmentation.state,segmentationReason:segmentation.reason,maskPixels:segmentation.pixels,segmentationRequests:segmentation.requests,
        correspondence:correspondence(),confirmed,candidate:!!state.mask&&!confirmed,outlineVisible:outlineVisible()},
      trackerReason:trackingResult?.reason??null,trackerExplanation:trackingResult&&!trackingResult.valid?humanReason(trackingResult.reason):null,
      trackerDiagnostics:trackingResult?.diagnostics??null,issue,timings:{...timings},
      gates:QUALITY_GATES,failureSnapshot,trace:traceLog.slice()};
  }
  function refreshDiagnostics(){if($('camera-diagnostics'))$('camera-diagnostics').textContent=JSON.stringify(diagnostics(),null,2);}
  async function copyDiagnostics(){
    refreshDiagnostics();
    const text=$('camera-diagnostics').textContent,note=$('copy-status');
    try{
      if(!root.isSecureContext||!navigator.clipboard?.writeText)throw Error('clipboard-unavailable');
      await navigator.clipboard.writeText(text);note.textContent='Diagnostics copied.';trace('diagnostics-copied');
    }catch(error){
      // Fallback: reveal the JSON and select it so the user can copy it manually.
      const details=$('camera-diagnostics').closest('details');if(details)details.open=true;
      try{const range=document.createRange();range.selectNodeContents($('camera-diagnostics'));const selection=root.getSelection();selection.removeAllRanges();selection.addRange(range);}catch(ignored){}
      note.textContent='Copy blocked. Select the text below and copy it.';trace('diagnostics-copy-failed');
    }
  }
  // Frames stopped verifying: suppress every overlay, keep the evidence, never silently reset.
  function previewOnly(reason,detail,short){
    setRenderMode('raw');trackingResult=null;
    if(!cameraStalled){cameraStalled=true;trace('preview-only',{reason});if(state.frame){trackingLost=true;confirmed=false;}setIssue('frames',reason,detail,short);status('Camera preview only.');ui();}
  }
  function tracksOff(expected){
    if(expected&&expected!==stream){expected.getTracks().forEach(track=>track.stop());return;}
    stream?.getTracks().forEach(track=>track.stop());stream=null;video.pause();video.srcObject=null;
  }
  function releaseImages(){for(const canvas of [source,staging,snapshot,tracking,overlay,maskCanvas,candidate,outline]){canvas.width=1;canvas.height=1;}sourceTimestamp=0;sourceIdentity=null;sourceOrientation='';overlayReady=false;candidateReady=false;cameraStalled=false;decodedSerial=0;decodedArrival=0;cameraStarted=0;acceptedFrames=0;rejectedFrame='not-started';captureError=null;setRenderMode('raw');}
  function resetSegmentation(){Object.assign(segmentation,{state:'idle',reason:null,startedAt:0,latencyMs:null,pixels:0});}
  function releasePointers(){const surfaceGesture=$('preview-gesture');for(const id of pointers.keys())if(surfaceGesture.hasPointerCapture(id))surfaceGesture.releasePointerCapture(id);pointers.clear();}
  function resetPreview(){releasePointers();$('preview-card').classList.remove('springing');Object.assign(inspector,{mode:'rotate',rotationX:0,rotationY:0,scale:1,panX:0,panY:0});paintPreview();}
  // Rubber band: after a drag the floating preview springs back to centered, front-facing rest. Zoom is kept.
  function springHome(){
    const card=$('preview-card');card.classList.add('springing');
    Object.assign(inspector,{rotationX:0,rotationY:0,panX:0,panY:0});paintPreview();
  }
  $('preview-card').addEventListener('transitionend',event=>{if(event.propertyName==='transform')$('preview-card').classList.remove('springing');});
  function closeInspector(discard=false){
    const wasOpen=surface==='result';
    animator.cancel();releasePointers();resetPreview();
    if(discard&&appliedRevision){
      artwork.width=appliedArtwork.width;artwork.height=appliedArtwork.height;const a=artwork.getContext('2d');a.clearRect(0,0,artwork.width,artwork.height);a.drawImage(appliedArtwork,0,0);
      pendingRevision=appliedRevision;artworkSource=appliedSource;rasterName=appliedRasterName;if(appliedRecipe)recipe=JSON.parse(JSON.stringify(appliedRecipe));recipePrompt=appliedPrompt;
      for(const id of ['artwork-reveal','artwork-back']){$(id).width=1;$(id).height=1;}
    }
    if(wasOpen){generateReturn='camera';setSurface('camera');}else ui();
  }
  function cancelReveal(){imports.cancel();closeInspector(true);}
  function commitArtwork(){appliedArtwork.width=artwork.width;appliedArtwork.height=artwork.height;const a=appliedArtwork.getContext('2d');a.clearRect(0,0,appliedArtwork.width,appliedArtwork.height);a.drawImage(artwork,0,0);appliedRevision=pendingRevision;appliedSource=artworkSource;appliedRasterName=rasterName;appliedRecipe=JSON.parse(JSON.stringify(recipe));appliedPrompt=recipePrompt;cacheArtwork();render();ui();}
  function paintPreview(){
    inspector.scale=Math.max(.5,Math.min(2,inspector.scale));inspector.rotationX=Math.max(-60,Math.min(60,inspector.rotationX));
    // Keep yaw in -180..180 so the spring back to front-facing takes the short way.
    inspector.rotationY=((inspector.rotationY+180)%360+360)%360-180;
    inspector.panX=Math.max(-140,Math.min(140,inspector.panX));inspector.panY=Math.max(-110,Math.min(110,inspector.panY));
    $('preview-card').style.transform=`translate(${inspector.panX}px,${inspector.panY}px) rotateX(${inspector.rotationX}deg) rotateY(${inspector.rotationY}deg) scale(${inspector.scale})`;
    $('rotate-preview').setAttribute('aria-pressed',String(inspector.mode==='rotate'));$('pan-preview').setAttribute('aria-pressed',String(inspector.mode==='pan'));
  }
  function watchDecodedFrames(media){
    useDecodedCallback=typeof video.requestVideoFrameCallback==='function';counterSupported=useDecodedCallback||typeof video.getVideoPlaybackQuality==='function';
    if(!useDecodedCallback)return;
    const next=(now)=>{if(stream!==media)return;decodedSerial++;decodedArrival=now;decodedCallback=video.requestVideoFrameCallback(next);};
    decodedCallback=video.requestVideoFrameCallback(next);
  }
  function loseTracking(reason,detail){
    trackingLost=true;confirmed=false;const raw=reason||trackingResult?.reason||'unknown';
    setIssue('tracking',raw,detail||humanReason(raw));ui();
  }
  function cancel(phase='idle',discardPreview=true){
    root.PhonePipeline?.disconnect();
    cameraGeneration++;trace('camera-stop');releaseAnalyze();
    if(animation!==null)cancelAnimationFrame(animation);animation=null;if(decodedCallback!==null)video.cancelVideoFrameCallback?.(decodedCallback);decodedCallback=null;
    if(discardPreview)cancelReveal();else releasePointers();client.close();tracker?.reset();tracksOff();
    state.reset(phase);busy=false;runtime='idle';trackingResult=null;trackingLost=false;confirmed=false;menuOpen=false;tapMarker=null;resetSegmentation();clearIssue();failureSnapshot=null;
    ctx.clearRect(0,0,scene.width,scene.height);releaseImages();ui();
  }
  async function prepare(epoch){
    runtime='loading';setRuntime('Preparing verified local object selection…');ui();
    try{
      await client.initialize(epoch);if(epoch!==state.epoch||state.phase!=='live')return;
      runtime='ready';selectionError=null;setRuntime('Object selection: local CPU/WASM. Tracking: experimental local 2D translation.');trace('worker-ready');
      if(issue?.kind==='initialization')clearIssue();
      status(renderMode==='synchronized'?'Hold Analyze to outline the center object':'Waiting for camera frames');
    }catch(error){
      if(epoch!==state.epoch||state.phase!=='live')return;runtime='unavailable';selectionError=error.message;setRuntime(error.message);trace('worker-failed');
      setIssue('initialization','worker-initialize-failed',`Object selection could not start: ${error.message}`);
    }
    ui();
  }
  function clearSelection(reload=true,discardPreview=true){
    if(state.phase!=='live')return;if(discardPreview)cancelReveal();client.close();tracker?.reset();state.reset('live');busy=false;trackingResult=null;trackingLost=false;confirmed=false;overlayReady=false;candidateReady=false;tapMarker=null;resetSegmentation();
    for(const canvas of [snapshot,maskCanvas,overlay,candidate,outline]){canvas.width=1;canvas.height=1;}render();ui();
    if(reload)prepare(state.epoch);
  }

  // ---------------------------------------------------------------- frames, tracking, rendering
  function readFrame(timestamp){
    if(!video.videoWidth||!video.videoHeight||video.readyState<2){rejectedFrame='video-not-ready';return false;}
    const size=PhoneSelection.boundedSize(video.videoWidth,video.videoHeight),angle=orientation();
    // A playback clock is not proof of newly decoded pixels. Prefer callbacks
    // from the video compositor; the fallback counts decoded video frames.
    let identity=null,arrival=timestamp;
    if(useDecodedCallback){if(!decodedSerial||timestamp-decodedArrival>FRESH_MS){rejectedFrame='no-fresh-video-callback';return false;}identity=`callback:${decodedSerial}`;arrival=decodedArrival;}
    else if(counterSupported){const quality=video.getVideoPlaybackQuality(),total=quality.totalVideoFrames,dropped=quality.droppedVideoFrames;if(!Number.isFinite(total)||!Number.isFinite(dropped)||dropped<0||total<dropped){rejectedFrame='invalid-decoded-counter';return false;}const presented=total-dropped;if(presented<1){rejectedFrame='no-decoded-frames';return false;}identity=`decoded:${presented}`;}
    if(identity===null){rejectedFrame='decoded-frame-verification-unavailable';return false;}
    if(sourceIdentity===identity){rejectedFrame='duplicate-frame';return false;}
    // Copy first: rejection or drawing failure must never erase the last committed source.
    if(staging.width!==size.width)staging.width=size.width;if(staging.height!==size.height)staging.height=size.height;
    const copyStart=performance.now();
    try{staging.getContext('2d',{willReadFrequently:true}).drawImage(video,0,0,size.width,size.height);}
    catch(error){captureError={stage:'camera-copy',name:error.name};rejectedFrame='camera-copy-failed';throw error;}
    timings.captureMs=ms(performance.now()-copyStart);
    const changed=source.width!==size.width||source.height!==size.height||sourceOrientation!==angle;
    const hadSelection=changed&&!!state.frame;
    if(hadSelection)clearSelection(true,false);
    const previous=source;source=staging;staging=previous;
    sourceTimestamp=Math.min(timestamp,arrival);sourceIdentity=identity;sourceOrientation=angle;
    if(changed){scene.width=size.width;scene.height=size.height;const trackSize=PhoneSelection.boundedSize(size.width,size.height,192);tracking.width=trackSize.width;tracking.height=trackSize.height;}
    if(hadSelection)setIssue('frames','geometry-changed','The camera size or orientation changed, so the selection was cleared.');
    acceptedFrames++;rejectedFrame=null;setRenderMode('synchronized');return true;
  }
  function trackingFrame(){const start=performance.now();trackingContext.drawImage(source,0,0,tracking.width,tracking.height);const rgba=trackingContext.getImageData(0,0,tracking.width,tracking.height).data;timings.readbackMs=ms(performance.now()-start);return {rgba,width:tracking.width,height:tracking.height,timestamp:sourceTimestamp};}
  function render(){
    if(renderMode!=='synchronized')return;
    const start=performance.now();
    ctx.clearRect(0,0,scene.width,scene.height);if(!sourceTimestamp)return;ctx.drawImage(source,0,0);
    // Only a verified frame with a valid tracked translation draws anything: no stale mask or artwork.
    if(freshResult()){
      const dx=trackingResult.dx*source.width/tracking.width,dy=trackingResult.dy*source.height/tracking.height;
      if(confirmed&&appliedByUser&&overlayReady){
        ctx.save();ctx.globalAlpha=placement.opacity/100;ctx.drawImage(overlay,dx,dy);ctx.restore();
        if(candidateReady)ctx.drawImage(outline,dx,dy); // observed boundary stays visible over the design
      }else if(candidateReady)ctx.drawImage(candidate,dx,dy);
    }
    timings.renderMs=ms(performance.now()-start);
  }
  // Two-tone observed boundary: dark halo (radius 3 source px) under a light mint contour
  // (radius 1.5), plus a faint inner tint. Built once per registered mask; nothing is invented.
  function buildCandidate(mask,width,height){
    const level=new Uint8Array(width*height),HALO=9,LINE=2.25;
    for(let i=0;i<level.length;i++)if(mask[i])level[i]=1;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const i=y*width+x;if(!mask[i])continue;
      if(!(x===0||y===0||x===width-1||y===height-1||!mask[i-1]||!mask[i+1]||!mask[i-width]||!mask[i+width]))continue;
      for(let dy=-3;dy<=3;dy++){const yy=y+dy;if(yy<0||yy>=height)continue;for(let dx=-3;dx<=3;dx++){const xx=x+dx,d2=dx*dx+dy*dy;if(xx<0||xx>=width||d2>HALO)continue;const k=yy*width+xx,v=d2<=LINE?3:2;if(level[k]<v)level[k]=v;}}
    }
    const paint=(canvas,withTint)=>{
      canvas.width=width;canvas.height=height;const c=canvas.getContext('2d'),image=c.createImageData(width,height),d=image.data;
      for(let i=0;i<level.length;i++){const j=i*4,v=level[i];
        if(v===3){d[j]=176;d[j+1]=248;d[j+2]=214;d[j+3]=255;}
        else if(v===2){d[j]=6;d[j+1]=14;d[j+2]=12;d[j+3]=150;}
        else if(v===1&&withTint){d[j]=214;d[j+1]=246;d[j+2]=228;d[j+3]=34;}}
      c.putImageData(image,0,0);
    };
    paint(candidate,true);paint(outline,false);candidateReady=true;
  }
  function cacheArtwork(){
    overlayReady=false;if(!state.frame||!state.mask)return;
    const {width,height}=state.frame;overlay.width=width;overlay.height=height;maskCanvas.width=width;maskCanvas.height=height;
    const m=maskCanvas.getContext('2d'),image=m.createImageData(width,height);
    for(let i=0;i<state.mask.length;i++){const j=i*4;image.data[j]=255;image.data[j+1]=255;image.data[j+2]=255;image.data[j+3]=state.mask[i];}m.putImageData(image,0,0);
    const o=overlay.getContext('2d'),size=Math.max(width,height)*placement.scale/100;
    o.save();o.translate(width*(.5+placement.x/100),height*(.5+placement.y/100));o.rotate(placement.rotation*Math.PI/180);o.drawImage(appliedArtwork,-size/2,-size/2,size,size);o.restore();
    o.globalCompositeOperation='destination-in';o.drawImage(maskCanvas,0,0);o.globalCompositeOperation='source-over';overlayReady=true;
  }
  function tick(timestamp){
    if(state.phase!=='live')return;animation=requestAnimationFrame(tick);
    // Canvas composition contains matching pixels and overlays. Raw fallback has no overlay.
    if(timestamp-lastTick<66)return;lastTick=timestamp;
    try{
      if(!readFrame(performance.now())){
        const now=performance.now();
        if(sourceTimestamp&&now-sourceTimestamp>FRESH_MS)previewOnly('frames-stalled','Camera frames stopped verifying, so the outline and design are hidden.');
        else if(!sourceTimestamp&&cameraStarted&&now-cameraStarted>1500)previewOnly('no-verified-frames','This browser is not delivering verifiable camera frames, so object selection is unavailable.','Selection unavailable on this camera feed.');
        render();syncLive();return;
      }
      if(cameraStalled){
        cameraStalled=false;trace('frames-resumed');
        status(state.frame?'Camera resumed':runtime==='ready'?'Hold Analyze to outline the center object':'Preparing selection…');
        if(runtime==='idle')prepare(state.epoch);ui();
      }else if(runtime==='ready'&&!issue&&statusText==='Waiting for camera frames'){
        status('Hold Analyze to outline the center object');
      }
      let tracked=false;
      if(state.frame&&tracker){
        const frame=trackingFrame(),start=performance.now();
        tracker.update(frame);trackingResult=tracker.result();timings.trackerMs=ms(performance.now()-start);tracked=true;
        // Preserve the first correspondence failure even while segmentation is
        // pending; waiting for its reply can otherwise erase the causal timing.
        if(!trackingResult.valid&&trackingResult.reason!=='awaiting-mask'&&!failureSnapshot){
          trace('tracking-failed',{reason:trackingResult.reason});captureFailure('tracking',trackingResult.reason);
        }
        if(state.mask&&!trackingResult.valid&&!trackingLost)loseTracking(trackingResult.reason);
      }
      render();syncLive();
      if(tracked){
        const d=trackingResult.diagnostics||{};
        trace('frame',{age:Math.round(performance.now()-sourceTimestamp),cap:timings.captureMs,rb:timings.readbackMs,trk:timings.trackerMs,rnd:timings.renderMs,
          init:d.initialFeatures??null,surv:d.survivingFeatures??null,masked:d.maskedInitialFeatures??null,inl:d.inliers??null,quad:d.occupiedQuadrants??null,
          valid:!!trackingResult.valid,reason:trackingResult.reason});
      }
    }catch(error){captureError={stage:rejectedFrame==='camera-copy-failed'?'camera-copy':'tracking-or-render',name:error.name};previewOnly('frame-processing-failed','Camera frame processing failed, so selection is paused.');syncLive();}
  }

  // ---------------------------------------------------------------- camera lifecycle
  async function start(){
    cancel('idle',false);const epoch=state.begin(),generation=cameraGeneration;ui();status('Allow camera access');
    if(!root.isSecureContext||!navigator.mediaDevices?.getUserMedia){cancel('error');status('Camera needs HTTPS or localhost.',true);return;}
    try{
      const media=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:facing},width:{ideal:1280},height:{ideal:720}},audio:false});
      if(generation!==cameraGeneration||epoch!==state.epoch||document.hidden){media.getTracks().forEach(track=>track.stop());return;}
      stream=media;video.srcObject=media;cameraStarted=performance.now();traceLog.length=0;trace('camera-start');setRenderMode('raw');watchDecodedFrames(media);
      for(const track of media.getVideoTracks())track.addEventListener('ended',()=>{if(stream===media){cancel();status('Camera disconnected. Open it again.',true);}},{once:true});
      if(!state.permission(epoch)){tracksOff(media);return;}
      // Watch for missing frames even if Safari's play promise does not settle.
      // All camera copies (including the first) run inside the guarded loop.
      ui();animation=requestAnimationFrame(tick);
      await video.play();if(generation!==cameraGeneration||stream!==media||document.hidden){tracksOff(media);return;}
      trace('play');
      if(!tracker){runtime='unavailable';setIssue('initialization','tracker-unavailable','Local tracking did not load.');ui();return;}
      if(!counterSupported){runtime='unavailable';setIssue('initialization','frame-verification-unavailable','This browser cannot verify camera frames.');ui();return;}
      if(runtime==='idle'){status('Preparing selection…');prepare(state.epoch);}
    }catch(error){if(generation!==cameraGeneration)return;cancel('error');status(error.name==='NotAllowedError'?'Camera permission denied.':'Camera could not open.',true);captureError={stage:'open',name:error.name};}
  }
  function rejectTap(kind,reason,detail){segmentation.state='rejected';segmentation.reason=reason;if(tapMarker)tapMarker.state='rejected';setIssue(kind,reason,detail);ui();}
  async function select(point){
    if(state.phase!=='live'||surface!=='camera')return;
    // Acknowledge the request immediately with a labelled marker (never an object outline).
    // A new selection starts fresh evidence: the old issue and frozen failure snapshot are cleared.
    hover=point;tapMarker={state:'pending'};clearIssue();failureSnapshot=null;traceLog.length=0;trace('tap');
    if(runtime==='loading'){tapMarker.state='waiting';status('Selection is loading. Try again in a moment.');ui();return;}
    if(runtime!=='ready'){rejectTap('initialization',selectionError?'worker-initialize-failed':'selection-unavailable',selectionError?`Object selection is unavailable: ${selectionError}`:'Object selection is unavailable.');return;}
    if(!tracker){rejectTap('initialization','tracker-unavailable','Local tracking did not load.');return;}
    if(!counterSupported){rejectTap('initialization','frame-verification-unavailable','This browser cannot verify camera frames.');return;}
    if(renderMode!=='synchronized'){rejectTap('frames','no-verified-frames','No verified camera frames at tap time.');return;}
    // Retapping clears any confirmation. A ready, idle worker is reused through its reset
    // protocol; a busy one is terminated. Nothing is queued.
    confirmed=false;
    const reuse=client.ready&&!client.pending;
    if(!reuse)client.close();
    if(!sourceTimestamp||performance.now()-sourceTimestamp>FRESH_MS){if(!readFrame(performance.now())){rejectTap('frames',rejectedFrame||'no-fresh-frame','No fresh camera frame for this tap.');return;}}
    cameraStalled=false;
    const tag=state.select(point,source.width,source.height,sourceOrientation,sourceTimestamp);if(!tag)return;
    snapshot.width=source.width;snapshot.height=source.height;snapshot.getContext('2d').drawImage(source,0,0);
    trackingResult=null;trackingLost=false;overlayReady=false;candidateReady=false;busy=true;
    Object.assign(segmentation,{state:'pending',reason:null,startedAt:performance.now(),latencyMs:null,pixels:0});segmentation.requests++;
    try{tracker.start({...trackingFrame(),epoch:tag.epoch,frameId:tag.frameId});trackingResult=tracker.result();const d=trackingResult.diagnostics||{};trace('tracker-start',{valid:!!trackingResult.valid,reason:trackingResult.reason,init:d.initialFeatures??null,eligible:d.eligibleCandidates??null});}
    catch(error){busy=false;segmentation.state='rejected';segmentation.reason='tracker-start-failed';if(tapMarker)tapMarker.state='rejected';loseTracking('tracker-start-failed');return;}
    render();ui();status('Outlining…');
    let image=null;
    try{
      if(reuse)await client.request('reset',{epoch:tag.epoch});else await client.initialize(tag.epoch);
      if(!state.accepts(tag))return;
      image=await createImageBitmap(snapshot);if(!state.accepts(tag)){image.close();image=null;return;}
      const result=await client.process(image,tag,state.strokes);image=null;
      if(!state.accepts(tag)||!PhoneSelection.sameFrame(result.tag,tag))return;
      const mask=new Uint8Array(result.mask);
      if(mask.length!==tag.width*tag.height)throw Error('The local mask dimensions are invalid.');
      let pixels=0;for(let i=0;i<mask.length;i++)if(mask[i])pixels++;
      segmentation.latencyMs=timings.segmentationMs=ms(performance.now()-segmentation.startedAt);segmentation.pixels=pixels;
      trace('mask',{ms:segmentation.latencyMs,px:pixels});
      if(!pixels){rejectTap('segmentation','empty-mask','The tap produced an empty outline.');return;}
      // Announce nothing until the tracker has actually registered this mask for the tap frame.
      const registered=tracker.setMask({mask,width:tag.width,height:tag.height,epoch:tag.epoch,frameId:tag.frameId});trackingResult=tracker.result();
      trace('set-mask',{registered,valid:!!trackingResult.valid,reason:trackingResult.reason});
      if(!registered){
        const reason=trackingResult.reason;
        // A tracker that already failed on the tap frame (e.g. too little texture) refuses the mask.
        if(reason&&!['awaiting-mask','tracking','not-started'].includes(reason)){
          segmentation.state='nonempty';segmentation.reason=`tracker-rejected:${reason}`;if(tapMarker)tapMarker.state='rejected';
          loseTracking(reason,`An outline was found, but live 2D tracking cannot use it: ${humanReason(reason)}.`);return;
        }
        rejectTap('segmentation','mask-not-registered','The outline did not match the current tap frame and was discarded.');return;
      }
      if(!state.accept(tag,mask))return;
      segmentation.state='nonempty';buildCandidate(mask,tag.width,tag.height);cacheArtwork();
      // Success requires the same gate that draws the outline and offers "Use object".
      if(!freshResult()){
        if(tapMarker)tapMarker.state='rejected';
        if(issue?.kind==='frames'){
          // Keep the earlier frame evidence; trackingLost and the mask stay as they are.
          setIssue('frames',issue.reason,`${issue.detail} A later outline belongs to that lost selection.`,issue.short);ui();
        }else{
          const reason=!trackingResult.valid?trackingResult.reason:trackingLost?'tracking-lost-before-outline':renderMode!=='synchronized'?'no-verified-frames':'stale-after-segmentation';
          const why=!trackingResult.valid?humanReason(reason):trackingLost?'tracking was already lost while outlining':renderMode!=='synchronized'?'camera frames were not verified when the outline arrived':'the frame went stale while outlining';
          loseTracking(reason,`An outline was found, but live 2D tracking cannot hold it: ${why}.`);
        }
      }else{tapMarker=null;status('Outline found');}
      render();
    }catch(error){
      if(state.accepts(tag)){
        selectionError=error.message;trackingResult=null;overlayReady=false;candidateReady=false;
        if(!client.ready)runtime='unavailable';
        rejectTap('segmentation','segmentation-failed',`No outline: ${error.message}`);
      }
    }
    finally{image?.close();if(state.accepts(tag)){busy=false;ui();}}
  }
  function confirmSelection(){
    if(confirmed&&freshResult())return true;
    if(!canConfirm()){status(state.mask?'Hold the object in view.':'Analyze an object first.');ui();return false;}
    confirmed=true;trace('confirm');cacheArtwork();render();status('Selected. Add a design.');ui();return true;
  }

  // ---------------------------------------------------------------- Analyze hold (deliberate, once per hold)
  // The glow is a press affordance only; it says nothing about tracking quality.
  function pressAnalyze(origin){
    if(analyzeHeld||surface!=='camera')return false;
    analyzeHeld=true;analyzeSource=origin;lastAnalyzePress=performance.now();trace('analyze-hold',{origin:String(origin).slice(0,12)});ui();
    if(state.phase!=='live'){status('Open the camera to analyze.');return true;}
    if(busy||client.pending){trace('analyze-skipped',{reason:'busy'});return true;}
    select({x:.5,y:.5});return true;
  }
  function releaseAnalyze(){
    if(!analyzeHeld)return;
    analyzeHeld=false;analyzeSource=null;analyzeKey=null;
    const button=$('analyze');if(analyzePointer!==null&&button.hasPointerCapture?.(analyzePointer))button.releasePointerCapture(analyzePointer);analyzePointer=null;
    trace('analyze-release');ui();
  }

  // ---------------------------------------------------------------- local idea helper (deterministic, offline)
  // Not an AI model: it collects object/shape/material/features and the colors/pattern the local
  // generator can actually draw, then carries the brief into Generate.
  const IDEA_OBJECTS=['desk lamp','floor lamp','lamp','armchair','chair','stool','bench','table','desk','shelf','vase','planter','plant','mug','cup','bottle','bowl','kettle','pair of sneakers','sneakers','sneaker','shoes','boots','headphones','earbuds','speaker','phone case','watch','backpack','bag','helmet','bicycle','bike','skateboard','car model','car','toy','figurine','sculpture','clock','keyboard','mouse','guitar','sunglasses','jacket','hoodie','t-shirt','cap'];
  const IDEA_SHAPES=['cylindrical','spherical','rectangular','geometric','faceted','tapered','organic','rounded','round','square','angular','curved','slim','tall','low','flat','sleek','compact','oversized','chunky'];
  const IDEA_MATERIALS=['carbon fiber','walnut','oak','bamboo','wood','marble','stone','concrete','porcelain','ceramic','clay','glass','aluminium','aluminum','steel','brass','copper','metal','leather','fabric','wool','linen','plastic','rubber','silicone','paper','cork'];
  const IDEA_FEATURES=['soft warm light','warm light','soft light','minimalist','modern','premium','playful','matte','glossy','textured','sporty','retro','industrial','ergonomic','foldable','wireless','handmade','bold','calm','warm','soft','lightweight','durable'];
  const PATTERN_WORDS=/\b(solid|stripes?|striped|checker|checkerboard|checkered|dots?|dotted|waves?|wavy|grid)\b/gi;
  const PATTERN_ALIASES={stripe:'stripes',striped:'stripes',checkerboard:'checker',checkered:'checker',dot:'dots',dotted:'dots',wave:'waves',wavy:'waves'};
  const COLOR_WORDS=new RegExp(`#[0-9a-f]{6}\\b|\\b(?:${Object.keys(DesignEngine.NAMED_COLORS).join('|')})\\b`,'gi');
  const brief={object:'',shapes:[],materials:[],features:[],colors:[],pattern:''};
  const chatMessages=[];let chatStarted=false;
  const escapeRe=text=>text.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  function findTerms(text,list,limit){
    const found=[];let rest=` ${text.toLowerCase()} `;
    for(const term of list){const re=new RegExp(`\\b${escapeRe(term)}\\b`);if(re.test(rest)){found.push(term);rest=rest.replace(re,' ');if(found.length>=limit)break;}}
    return found;
  }
  function addUnique(list,items,limit){for(const item of items)if(!list.includes(item)&&list.length<limit)list.push(item);}
  function briefText(){
    const parts=[brief.object,brief.shapes.join(' '),brief.materials.join(' and '),brief.features.join(', ')].filter(Boolean);
    const look=[brief.colors.join(' and '),brief.pattern].filter(Boolean).join(' ');
    return [parts.join(', '),look].filter(Boolean).join('; ').slice(0,500);
  }
  function briefHasContent(){return !!(brief.object||brief.shapes.length||brief.materials.length||brief.features.length||brief.colors.length||brief.pattern);}
  function ideaReply(text){
    const captured=[];
    const object=findTerms(text,IDEA_OBJECTS,1)[0];if(object){brief.object=object.startsWith('pair')?`a ${object}`:object;captured.push(`object: ${brief.object}`);}
    const shapes=findTerms(text,IDEA_SHAPES,3);if(shapes.length){addUnique(brief.shapes,shapes,3);captured.push(`shape: ${shapes.join(', ')}`);}
    const materials=findTerms(text,IDEA_MATERIALS,3);if(materials.length){addUnique(brief.materials,materials,3);captured.push(`materials: ${materials.join(', ')}`);}
    const features=findTerms(text,IDEA_FEATURES,3);if(features.length){addUnique(brief.features,features,4);captured.push(`feel: ${features.join(', ')}`);}
    const colors=[...new Set((text.match(COLOR_WORDS)||[]).map(c=>c.toLowerCase()))];if(colors.length){brief.colors=colors.slice(0,5);captured.push(`colors: ${brief.colors.join(', ')}`);}
    const patterns=[...(text.match(PATTERN_WORDS)||[])].map(p=>PATTERN_ALIASES[p.toLowerCase()]||p.toLowerCase());if(patterns.length){brief.pattern=patterns[0];captured.push(`pattern: ${brief.pattern}`);}
    const lead=captured.length?`Noted ${captured.join('; ')}.`:'I didn’t catch a specific detail.';
    let next;
    if(!brief.object)next='What product or object is it? For example a chair, a vase or headphones.';
    else if(!brief.shapes.length)next='What shape or silhouette should it have: round, slim, angular, organic?';
    else if(!brief.materials.length)next='Which materials or finish? I’ll keep them in your brief.';
    else if(!brief.features.length)next='Any key features or feel, like matte, warm light or ergonomic?';
    else if(!brief.colors.length&&!brief.pattern)next='Last step for the local generator: which colors and pattern? It can draw stripes, checker, dots, waves, grid or solid.';
    else next='Your brief is ready. Create a design from it below. The local generator draws the pattern and colors; object, shape and material words stay as notes.';
    if(captured.length&&(materials.length||shapes.length)&&!(brief.colors.length||brief.pattern))next+=' Note: the generator can’t render materials or shapes, only patterns and colors.';
    return {title:captured.length?'Got it.':'Tell me a little more.',text:`${lead} ${next}`};
  }
  function renderChat(){
    const log=$('chat-log');log.replaceChildren();
    for(const message of chatMessages){
      const li=document.createElement('li');li.className=`msg ${message.role}`;
      if(message.role==='assistant'){const avatar=document.createElement('span');avatar.className='msg-avatar';avatar.setAttribute('aria-hidden','true');avatar.innerHTML='<svg viewBox="0 0 24 24"><path d="M12 3c.6 4.6 3.4 7.4 9 9-5.6 1.6-8.4 4.4-9 9-.6-4.6-3.4-7.4-9-9 5.6-1.6 8.4-4.4 9-9z"/></svg>';li.append(avatar);}
      const bubble=document.createElement('div');bubble.className='msg-bubble';
      if(message.title){const strong=document.createElement('strong');strong.textContent=message.title;bubble.append(strong);}
      const p=document.createElement('p');p.textContent=message.text;bubble.append(p);
      const label=document.createElement('span');label.className='sr-only';label.textContent=message.role==='assistant'?'Idea helper: ':'You: ';bubble.prepend(label);
      li.append(bubble);log.append(li);
    }
    $('chat-use-brief').hidden=!briefHasContent();
    log.scrollTop=log.scrollHeight;
  }
  function pushChat(message){chatMessages.push(message);while(chatMessages.length>40)chatMessages.shift();}
  function openChat(){
    if(!chatStarted){chatStarted=true;pushChat({role:'assistant',title:'Hi! I’m the VisualizeIT idea helper.',text:'Tell me about the product you want to create. I’ll help refine its shape, materials and features, then carry the brief into Generate.'});}
    renderChat();setSurface('chat',$('chat-input'));
  }
  function respond(text){
    const value=String(text||'').trim().slice(0,300);if(!value)return null;
    pushChat({role:'user',text:value});const reply=ideaReply(value);pushChat({role:'assistant',...reply});renderChat();return reply;
  }

  // ---------------------------------------------------------------- generate validation (honest local capability)
  function hasSupportedTerms(prompt){
    COLOR_WORDS.lastIndex=0;PATTERN_WORDS.lastIndex=0;
    const ok=PATTERN_WORDS.test(prompt)||COLOR_WORDS.test(prompt)||/"[^"\n]+"|'[^'\n]+'/.test(prompt)||/\b(?:text|say|saying)\s*:?\s+\S/i.test(prompt)||/\b(?:scale|rotation|rotate|seed)\s*:?\s*[+-]?\d/i.test(prompt);
    COLOR_WORDS.lastIndex=0;PATTERN_WORDS.lastIndex=0;return ok;
  }
  function promptProblem(prompt){
    const value=prompt.trim();
    if(!value)return 'Describe a pattern and colors, e.g. “blue and white stripes”.';
    if(!hasSupportedTerms(value))return `This local generator draws patterns, colors and quoted text, not objects. Try “${value.replace(/[,;.\s]+$/,'').slice(0,60)}, blue and white stripes”.`;
    return '';
  }
  function promptStatus(text,error=false){const el=$('prompt-status');el.textContent=text;el.classList.toggle('error',error);}
  function updateCount(){const value=$('phone-prompt').value;$('prompt-count').textContent=`${Array.from(value).length}/500`;
    for(const chip of document.querySelectorAll('#prompt-ideas .idea-chip'))chip.setAttribute('aria-pressed',String(value.toLowerCase().startsWith(chip.dataset.idea.toLowerCase())));}
  function openGenerate(origin='camera'){
    generateReturn=origin;
    if(!$('phone-prompt').value.trim()&&recipePrompt)$('phone-prompt').value=recipePrompt;
    updateCount();promptStatus('');setSurface('generate',$('phone-prompt'));
  }

  // ---------------------------------------------------------------- menu and history
  function setMenu(open){if(open&&surface!=='camera')return;menuOpen=open;ui();if(open){refreshDiagnostics();$('copy-status').textContent='';$('camera-menu').querySelector('button:not(:disabled),a')?.focus({preventScroll:true});}}
  function addHistory(prompt,result){
    const list=$('ask-history');
    for(const [text,cls] of [[prompt,'prompt'],[result,'result']]){if(!text)continue;const li=document.createElement('li');li.className=cls;li.textContent=text;list.append(li);}
    while(list.children.length>HISTORY_LIMIT*2)list.firstElementChild.remove();
  }

  // ---------------------------------------------------------------- result preview, generation, import
  // The result is an independent flat 2D card floating in front of the camera, never attached to the scene.
  function revealArtwork(note='',{title,animate=true}={}){
    releasePointers();
    for(const id of ['artwork-reveal','artwork-back']){const canvas=$(id);canvas.width=512;canvas.height=512;const c=canvas.getContext('2d');c.clearRect(0,0,512,512);c.drawImage(artwork,0,0,512,512);}
    $('result-title').textContent=title||(artworkSource==='raster'?'Design uploaded':'Design generated');
    setSurface('result');resetPreview();ui();if(animate)animator.play();
    $('preview-status').textContent=note;$('preview-status').classList.remove('error');
    $('preview-gesture').focus({preventScroll:true});
  }
  function generate(event,animate=true){
    event?.preventDefault();
    const prompt=$('phone-prompt').value;
    if(animate){const problem=promptProblem(prompt);if(problem){promptStatus(problem,true);trace('generate-rejected');return;}}
    try{
      const parsed=DesignEngine.parsePrompt(prompt,recipe);
      imports.cancel();recipe=parsed.recipe;recipePrompt=prompt;DesignEngine.draw(artwork,recipe);artworkSource='recipe';rasterName='';pendingRevision=++revisionSerial;
      if(animate){
        addHistory(prompt.trim(),`${recipe.pattern} · ${recipe.colors.join(' / ')}${parsed.warnings.length?' · see notes':''}`);
        const unsupported=parsed.unsupportedWords;
        const note=unsupported.length?`Not drawn: ${unsupported.slice(0,6).join(', ')}${unsupported.length>6?'…':''}. Only pattern, colors and text are drawn.`:parsed.warnings[0]||'';
        promptStatus('');$('phone-prompt').blur();generateReturn='camera';trace('generated');
        revealArtwork(note,{title:'Design generated'});
      }else{promptStatus(parsed.warnings.join(' '));cacheArtwork();render();ui();}
    }catch(error){promptStatus(error.message,true);}
  }
  async function importFile(file){
    try{
      status('Opening image…');
      const result=await imports.submit(file);
      if(!result.ok){
        if(result.code!=='superseded'){
          const message=`Upload failed: ${result.message}`;status(message,true);
          if(surface==='result'){$('preview-status').textContent=message;$('preview-status').classList.add('error');}
        }
        return false;
      }
      const raster=result.raster;
      try{artwork.width=raster.canvas.width;artwork.height=raster.canvas.height;const a=artwork.getContext('2d');a.clearRect(0,0,artwork.width,artwork.height);a.drawImage(raster.canvas,0,0);}
      finally{raster.canvas.width=1;raster.canvas.height=1;}
      artworkSource='raster';rasterName=raster.name;pendingRevision=++revisionSerial;
      addHistory(`Imported ${raster.name}`,`${raster.sourceWidth}×${raster.sourceHeight} fitted to 1024 px, transparency kept`);
      status('Image opened');generateReturn='camera';revealArtwork('',{title:'Design uploaded'});return true;
    }catch(error){status(`Upload failed: ${error.message}`,true);return false;}
  }
  function chooseUpload(){if(surface==='chooser')setSurface('camera');$('design-file').click();}
  // Explicit workbench save only (called by phone-pipeline.js). Returns the CURRENT generated
  // artwork: its normalized recipe, prompt, revision and PNG of #artwork-preview. Never the scene,
  // camera source or video. Imported rasters have no recipe and are refused, never paired with one.
  function workbenchArtwork(){
    if(artworkSource!=='recipe')throw Error('Uploaded images can’t be saved to the workbench because the studio stores pattern recipes. Use Save artwork PNG in Artwork options instead.');
    const normalized=DesignEngine.normalizeRecipe(recipe);
    if(artwork.width!==normalized.width||artwork.height!==normalized.height)throw Error('The current artwork does not match its recipe. Generate it again before saving.');
    const revision=pendingRevision,prompt=String(recipePrompt||'').slice(0,500);
    const url=artwork.toDataURL('image/png'),prefix='data:image/png;base64,';
    if(!url.startsWith(prefix)||url.length<=prefix.length)throw Error('PNG export is unavailable.');
    return {recipe:JSON.parse(JSON.stringify(normalized)),prompt,pendingRevision:revision,png_base64:url.slice(prefix.length)};
  }

  // ---------------------------------------------------------------- events
  function pointerPoint(event){return PhoneSelection.coverPointInCanvas(event.clientX,event.clientY,scene.getBoundingClientRect(),scene.width,scene.height);}
  scene.addEventListener('pointerdown',event=>{if(event.button!==0||surface!=='camera')return;const point=pointerPoint(event);if(point)select(point);});
  // Raw fallback: the tap is acknowledged and its real reason (no verified frames) is recorded.
  video.addEventListener('pointerdown',event=>{
    if(event.button!==0||state.phase!=='live'||surface!=='camera')return;
    const point=PhoneSelection.coverPointInCanvas(event.clientX,event.clientY,video.getBoundingClientRect(),video.videoWidth||scene.width,video.videoHeight||scene.height);
    if(point)select(point);
  });
  scene.addEventListener('keydown',event=>{
    if(surface!=='camera')return;
    const step=event.shiftKey?.05:.01,delta={ArrowLeft:[-step,0],ArrowRight:[step,0],ArrowUp:[0,-step],ArrowDown:[0,step]}[event.key];
    if(delta){event.preventDefault();hover={x:Math.max(0,Math.min(1,hover.x+delta[0])),y:Math.max(0,Math.min(1,hover.y+delta[1]))};if(!tapMarker||tapMarker.state!=='pending')tapMarker={state:'aim'};ui();}
    else if(event.key==='Enter'){event.preventDefault();select(hover);}
  });
  // Analyze: pointer hold, keyboard Space/Enter hold (repeats ignored), and a single assistive-tech click.
  const analyze=$('analyze');
  analyze.addEventListener('pointerdown',event=>{if(event.button!==0)return;event.preventDefault();if(pressAnalyze('pointer')){analyzePointer=event.pointerId;try{analyze.setPointerCapture(event.pointerId);}catch(ignored){}}});
  for(const type of ['pointerup','pointercancel','lostpointercapture'])analyze.addEventListener(type,event=>{if(analyzeSource==='pointer'&&(analyzePointer===null||event.pointerId===analyzePointer))releaseAnalyze();});
  analyze.addEventListener('keydown',event=>{if(event.key!==' '&&event.key!=='Enter')return;event.preventDefault();if(event.repeat||analyzeHeld)return;if(pressAnalyze('key'))analyzeKey=event.key;});
  analyze.addEventListener('keyup',event=>{if(event.key!==' '&&event.key!=='Enter')return;event.preventDefault();if(analyzeSource==='key'&&event.key===analyzeKey)releaseAnalyze();});
  analyze.addEventListener('blur',()=>{if(analyzeSource==='key')releaseAnalyze();});
  analyze.addEventListener('contextmenu',event=>event.preventDefault());
  analyze.addEventListener('click',event=>{
    // VoiceOver/Switch Control activation without a press: one bounded analyze, then release.
    if(event.detail===0&&!analyzeHeld&&performance.now()-lastAnalyzePress>600&&pressAnalyze('assistive'))setTimeout(()=>{if(analyzeSource==='assistive')releaseAnalyze();},450);
  });
  root.addEventListener('blur',releaseAnalyze);
  $('ask-button').onclick=()=>openChat();
  $('ask-open').onclick=()=>surface==='chat'?closeSurface():openChat();
  $('design-button').onclick=()=>setSurface('chooser',$('choose-generate'));
  $('chooser-close').onclick=()=>closeSurface();
  $('chat-close').onclick=()=>closeSurface();
  $('chooser-scrim').onclick=()=>closeSurface();
  $('choose-generate').onclick=()=>openGenerate('camera');
  $('import-design').onclick=chooseUpload;
  $('result-upload').onclick=()=>$('design-file').click();
  $('result-regenerate').onclick=()=>{$('phone-prompt').value=recipePrompt||$('phone-prompt').value;openGenerate('result');};
  $('generate-close').onclick=()=>closeSurface();
  $('chat-form').addEventListener('submit',event=>{event.preventDefault();const input=$('chat-input');if(respond(input.value))input.value='';});
  $('chat-input').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('chat-form').requestSubmit();}});
  $('chat-use-brief').onclick=()=>{$('phone-prompt').value=briefText();openGenerate('camera');};
  for(const chip of document.querySelectorAll('#prompt-ideas .idea-chip'))chip.addEventListener('click',()=>{
    const prompt=$('phone-prompt'),ideas=[...document.querySelectorAll('#prompt-ideas .idea-chip')].map(c=>c.dataset.idea);
    let rest=prompt.value.trim();for(const idea of ideas)if(rest.toLowerCase().startsWith(idea.toLowerCase())){rest=rest.slice(idea.length).replace(/^[,;\s]+/,'');break;}
    prompt.value=`${chip.dataset.idea}, ${rest}`.slice(0,500);updateCount();promptStatus('');
    prompt.focus({preventScroll:true});prompt.setSelectionRange(prompt.value.length,prompt.value.length);
  });
  $('phone-prompt').addEventListener('input',()=>{updateCount();if($('prompt-status').classList.contains('error'))promptStatus('');});
  $('phone-prompt').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('phone-prompt-form').requestSubmit();}});
  $('start').onclick=start;
  $('stop').onclick=()=>{cancel();status('Camera stopped.');setRuntime('Local object selection loads when the camera opens.');};
  $('switch').onclick=()=>{menuOpen=false;facing=facing==='environment'?'user':'environment';start();};
  $('retake').onclick=()=>{menuOpen=false;traceLog.length=0;trace('reset');clearIssue();failureSnapshot=null;clearSelection();status('Selection cleared');};
  $('refresh-app').onclick=()=>{cancel();location.reload();};
  $('refresh-diagnostics').onclick=refreshDiagnostics;
  $('copy-diagnostics').onclick=copyDiagnostics;
  $('camera-menu-toggle').onclick=()=>setMenu(!menuOpen);
  $('camera-menu').addEventListener('click',event=>{const button=event.target.closest('button');if(button&&!['refresh-diagnostics','copy-diagnostics'].includes(button.id))setMenu(false);});
  $('camera-menu').addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();event.stopPropagation();setMenu(false);$('camera-menu-toggle').focus();}});
  // A tap outside the open menu only closes it; it never also selects an object.
  document.addEventListener('pointerdown',event=>{if(menuOpen&&!event.target.closest('#camera-menu,#camera-menu-toggle')){event.stopPropagation();event.preventDefault();setMenu(false);}},true);
  $('confirm-object').onclick=()=>confirmSelection();
  $('design-file').addEventListener('change',()=>{const file=$('design-file').files?.[0];$('design-file').value='';if(file)importFile(file);});
  $('show-design').onclick=()=>revealArtwork('',{animate:false});
  const gesture=$('preview-gesture');
  function pointerDistance(){const values=[...pointers.values()];return values.length<2?0:Math.hypot(values[0].x-values[1].x,values[0].y-values[1].y);}
  gesture.addEventListener('pointerdown',event=>{event.stopPropagation();if(event.button!==0)return;event.preventDefault();$('preview-card').classList.remove('springing');dragged=false;pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});gesture.setPointerCapture(event.pointerId);gesture.focus({preventScroll:true});});
  gesture.addEventListener('pointermove',event=>{event.stopPropagation();const previous=pointers.get(event.pointerId);if(!previous)return;event.preventDefault();const before=pointerDistance(),dx=event.clientX-previous.x,dy=event.clientY-previous.y;pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});if(pointers.size>1){const after=pointerDistance();if(before>0&&after>0)inspector.scale*=after/before;}else{dragged=true;if(inspector.mode==='pan'){inspector.panX+=dx;inspector.panY+=dy;}else{inspector.rotationY+=dx*.6;inspector.rotationX-=dy*.4;}}paintPreview();});
  function endPointer(event){
    event.stopPropagation();const had=pointers.delete(event.pointerId);if(gesture.hasPointerCapture(event.pointerId))gesture.releasePointerCapture(event.pointerId);
    if(had&&pointers.size===0&&dragged){dragged=false;springHome();}
  }
  gesture.addEventListener('pointerup',endPointer);gesture.addEventListener('pointercancel',endPointer);gesture.addEventListener('lostpointercapture',endPointer);
  gesture.addEventListener('wheel',event=>{event.preventDefault();event.stopPropagation();inspector.scale*=Math.exp(-event.deltaY*.0015);paintPreview();},{passive:false});
  $('artwork-inspector').addEventListener('pointerdown',event=>event.stopPropagation());
  $('artwork-inspector').addEventListener('keydown',event=>{event.stopPropagation();if(event.key==='Tab'){trapTab(event);return;}if(event.key==='Escape'){event.preventDefault();closeInspector();return;}if(event.target!==gesture)return;const delta={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[event.key];if(delta){event.preventDefault();$('preview-card').classList.remove('springing');if(inspector.mode==='pan'){inspector.panX+=delta[0]*10;inspector.panY+=delta[1]*10;}else{inspector.rotationY+=delta[0]*10;inspector.rotationX-=delta[1]*10;}}else if(['+','=','-','_'].includes(event.key)){event.preventDefault();inspector.scale*=event.key==='-'||event.key==='_'?.9:1.1;}else return;paintPreview();});
  $('close-preview').onclick=()=>closeInspector();$('reset-preview').onclick=resetPreview;
  $('view-options-toggle').onclick=()=>{const panel=$('view-options'),open=panel.hidden;panel.hidden=!open;$('view-options-toggle').setAttribute('aria-expanded',String(open));ui();};
  $('rotate-preview').onclick=()=>{releasePointers();inspector.mode='rotate';paintPreview();};$('pan-preview').onclick=()=>{releasePointers();inspector.mode='pan';paintPreview();};
  $('apply-preview').onclick=()=>{
    if(!confirmed||!freshResult()){status(confirmed?'Hold the object in view.':'Analyze an object, then Use object.');ui();return;}
    commitArtwork();appliedByUser=true;trace('apply');cacheArtwork();render();ui();$('preview-status').textContent='Applied';status('Design applied');
  };
  function placementControls(){for(const key of ['x','y','scale','rotation','opacity']){$(`placement-${key}`).value=placement[key];$(`placement-${key}-value`).textContent=`${placement[key]}${key==='rotation'?'°':'%'}`;}cacheArtwork();render();}
  for(const key of ['x','y','scale','rotation','opacity'])$(`placement-${key}`).oninput=event=>{placement[key]=Number(event.target.value);placementControls();};
  $('center-artwork').onclick=()=>{placement={x:0,y:0,scale:100,rotation:0,opacity:80};placementControls();};
  function download(canvas,name){canvas.toBlob(blob=>{if(!blob){status('PNG export is unavailable.',true);return;}const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);},'image/png');}
  $('export-artwork').onclick=()=>download(artwork,'visualizeit-artwork.png');
  $('back-to-studio').onclick=event=>{
    event.preventDefault();
    // The studio only edits procedural recipes. An imported raster is never paired
    // with an unrelated recipe; its pixels are offered as a PNG instead.
    if(artworkSource==='raster'){status('Imported images save as PNG; studio editing is for patterns.',true);download(artwork,'visualizeit-imported-artwork.png');return;}
    try{
      const transfer={schema:'visualizeit_studio_return_v1',recipe:DesignEngine.normalizeRecipe(recipe),prompt:recipePrompt};
      sessionStorage.setItem('visualizeit_studio_return_v1',JSON.stringify(transfer));
      location.assign('/');
    }catch(error){status('Could not transfer artwork. Save its PNG instead.',true);}
  };
  $('export-mockup').onclick=()=>{if(!freshResult()){status('Reselect an object before saving a preview.',true);return;}render();const exported=document.createElement('canvas');exported.width=scene.width;exported.height=scene.height;exported.getContext('2d').drawImage(scene,0,0);download(exported,'visualizeit-camera-2d-preview.png');};
  $('phone-prompt-form').onsubmit=generate;
  const sheet=$('artwork-options');$('open-settings').onclick=()=>{menuOpen=false;ui();sheet.showModal();};$('close-settings').onclick=()=>sheet.close();
  sheet.addEventListener('click',event=>{if(event.target===sheet){const rect=sheet.getBoundingClientRect();if(event.clientY<rect.top||event.clientX<rect.left||event.clientX>rect.right)sheet.close();}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){releaseAnalyze();cancel();status('Camera stopped while hidden.');}});
  root.addEventListener('pagehide',()=>{cancel();releaseKeyboard();releaseKeyboard=()=>{};});
  root.addEventListener('pageshow',event=>{if(event.persisted){releaseKeyboard();releaseKeyboard=G.bindKeyboardInset(root,document.documentElement);}});
  root.addEventListener('resize',ui);
  if(G.shouldShowHomeScreenHint(root))$('home-screen-help').hidden=false;

  // Only an explicitly transferred artwork recipe crosses pages. Camera pixels,
  // masks and imported images are kept in memory, never in storage or requests.
  let imported=false;
  try{const handoff=sessionStorage.getItem('visualizeit_camera_recipe_v1');if(handoff){sessionStorage.removeItem('visualizeit_camera_recipe_v1');const transfer=JSON.parse(handoff);recipe=DesignEngine.normalizeRecipe(transfer.recipe||transfer);DesignEngine.draw(artwork,recipe);imported=true;$('phone-prompt').value=typeof transfer.prompt==='string'&&transfer.prompt.length<=500?transfer.prompt:`${recipe.pattern} ${recipe.colors.join(' and ')} scale ${recipe.scale} rotation ${recipe.rotation_deg} seed ${recipe.seed}${recipe.text?` text ${recipe.text}`:''}`;recipePrompt=$('phone-prompt').value;promptStatus('Studio artwork transferred on this device.');}}catch(error){promptStatus(`Artwork transfer unavailable: ${error.message}`,true);}
  root.PhoneCamera={initialize:start,process:select,select,confirmSelection,importFile,reset:cancel,close:cancel,selection:state,tracker,diagnostics,copyDiagnostics,
    openSurface,closeSurface,surface:()=>surface,workbenchArtwork,
    analyze:Object.freeze({press:()=>pressAnalyze('api'),release:releaseAnalyze,held:()=>analyzeHeld}),
    ideaHelper:Object.freeze({respond,brief:()=>JSON.parse(JSON.stringify(brief)),briefText,messages:()=>chatMessages.map(m=>({...m}))}),
    selectionState:()=>Object.freeze({segmentation:segmentation.state,segmentationReason:segmentation.reason,maskPixels:segmentation.pixels,
      correspondence:correspondence(),outlineVisible:outlineVisible(),tapMarker:tapMarker?tapMarker.state:null,issue:issue?{...issue}:null,
      trackerReason:trackingResult?.reason??null,timings:{...timings},useObjectVisible:canConfirm(),failureSnapshot,traceLength:traceLog.length}),
    previewState:()=>Object.freeze({mode:inspector.mode,rotation:{x:inspector.rotationX,y:inspector.rotationY},scale:inspector.scale,pan:{x:inspector.panX,y:inspector.panY},open:inspector.open,appliedRevision,pendingRevision,
      confirmed,candidate:!!state.mask&&!confirmed,artworkSource,appliedSource,rasterName,appliedRasterName,applied:appliedByUser,entering:animator.entering,
      surface,analyzeHeld,composerFocused:surface==='generate',askOpen:surface==='chat',outlineVisible:outlineVisible()})};
  if(!imported){generate(null,false);promptStatus('');}else pendingRevision=++revisionSerial;
  // Baseline for discard only; nothing is drawn on an object until Use object and Apply.
  commitArtwork();paintPreview();updateCount();ui();status('Open the camera to begin');
})(typeof globalThis!=='undefined'?globalThis:this);
