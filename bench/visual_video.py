"""Create a small synchronized mask/mapping player from existing native videos.

No new video encoding or inference. Masks are held-out dataset annotations.
"""
import base64
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisualizeIt — mask and surface mapping</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101b21;color:#e6eff4;font:16px system-ui}
main{max-width:1600px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 8px}
p{line-height:1.5;color:#b6c8d4;max-width:1100px}canvas{width:100%;background:#080f14;border-radius:12px;display:block}
.controls,.modes{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:15px 0}
button{background:#223943;color:white;border:1px solid #497080;border-radius:8px;padding:11px 16px;font:inherit;cursor:pointer}
button[aria-pressed=true]{background:#14736d;border-color:#5de2cb}input{flex:1;min-width:180px;accent-color:#5de2cb}
#status{font-variant-numeric:tabular-nums;color:#70ebd2}a{color:#70ebd2}video{position:fixed;bottom:0;left:0;width:1px;height:1px;opacity:.01;pointer-events:none}.note{font-size:14px}
</style><main><h1>Watch the mask and mapped design move</h1>
<p>Real recorded footage of a pudding box. The cyan mask is a dataset reference annotation. The checker design uses the object mesh and fixed surface coordinates.</p>
<div class="modes"><button id="control" aria-pressed="true">Mapping control · supplied poses</button><button id="tracker" aria-pressed="false">Our tracker · estimated poses</button><span id="status">Loading videos and masks…</span></div>
<canvas id="view" width="1920" height="540" aria-label="Synchronized original footage, reference mask, and mapped surface"></canvas>
<div class="controls"><button id="play">Play</button><button id="restart">Restart</button><input id="seek" type="range" min="0" max="5.6" step="0.001" value="0" aria-label="Playback position"><span id="time">0.0 / 5.6 s</span><button id="fullscreen">Full screen</button></div>
<p id="explanation">Supplied poses isolate the mapping: perspective, curvature, texture attachment, and visible seams. These poses come from the dataset.</p>
<p class="note">Switch to our tracker to see actual saved estimates. The design disappears when tracking is lost; accepted estimates can still be wrong. Automatic object-mask prediction is not implemented here. This view shows the native-resolution tracker, not the faster reduced-resolution candidate.</p>
<p class="note">1920 × 1440 source frames, cropped for inspection. 56 sampled frames replayed at 10 FPS for viewing; original capture FPS is unknown. Playback speed is not a live tracking performance result. Neutral diagnostic rendering; lighting and external occlusion remain unvalidated.</p>
<p><a href="index.html">Full-resolution videos</a> · <a href="progress.html">Speed experiments and actual input frames</a> · <a href="performance-higher-fps/index.html">Higher FPS experiment</a> · <a href="performance-object-region/index.html">Object-region speed test</a> · <a href="performance-pyramid/index.html">Feature-pyramid speed test</a></p>
<p class="note">NVIDIA HOPE / BOP; Stephen Tyree et al. Source licence: CC BY-NC-SA 4.0, per bundled dataset metadata. Local research preview.</p>
<video id="original" muted playsinline preload="auto" src="object-000005/original.webm"></video>
<video id="mapped" muted playsinline preload="auto" src="object-000005/supplied-pose.webm"></video>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.querySelector('#data').textContent);
const original=document.querySelector('#original'),mapped=document.querySelector('#mapped');
const view=document.querySelector('#view'),ctx=view.getContext('2d');
const play=document.querySelector('#play'),seek=document.querySelector('#seek'),status=document.querySelector('#status');
let mode='supplied',ready=false,changing=false,error=null,originalFrameTime=0,mappedFrameTime=0;
function observeFrames(v,set){if(v.requestVideoFrameCallback){v.requestVideoFrameCallback((_,meta)=>{set(meta.mediaTime);observeFrames(v,set);});}else{v.addEventListener('timeupdate',()=>set(v.currentTime),{once:true});}}
observeFrames(original,t=>originalFrameTime=t);observeFrames(mapped,t=>mappedFrameTime=t);
const crop=[512,384,896,672],masks=[];
const imageLoaded=src=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=src;});
function mediaReady(v){return new Promise((resolve,reject)=>{if(v.readyState>=2)return resolve();v.addEventListener('loadeddata',resolve,{once:true});v.addEventListener('error',()=>reject(new Error('Video failed to load.')),{once:true});});}
function tintedMask(im){const c=document.createElement('canvas');c.width=640;c.height=480;const x=c.getContext('2d',{willReadFrequently:true});x.drawImage(im,...crop,0,0,640,480);const pixels=x.getImageData(0,0,640,480),a=pixels.data;for(let i=0;i<a.length;i+=4){const inside=a[i]>127;a[i]=0;a[i+1]=230;a[i+2]=235;a[i+3]=inside?105:0;}x.putImageData(pixels,0,0);return c;}
function pause(){mapped.pause();original.pause();play.textContent='Play';}
async function start(){if(!ready||changing)return;try{original.currentTime=mapped.currentTime;await mapped.play();original.play().catch(()=>{});play.textContent='Pause';}catch(e){pause();status.textContent='Press Play to begin.';}}
function jump(t){const end=Number.isFinite(mapped.duration)?mapped.duration:5.6;mapped.currentTime=Math.max(0,Math.min(t,end));original.currentTime=mapped.currentTime;}
async function switchMode(next){if(next===mode||changing||!ready)return;changing=true;const t=mapped.currentTime,wasPlaying=!mapped.paused;pause();mode=next;ready=false;mapped.src='object-000005/'+(mode==='supplied'?'supplied-pose':'estimated-pose')+'.webm';mapped.load();document.querySelector('#control').setAttribute('aria-pressed',String(mode==='supplied'));document.querySelector('#tracker').setAttribute('aria-pressed',String(mode==='estimated'));document.querySelector('#explanation').textContent=mode==='supplied'?'Supplied poses isolate the mapping: perspective, curvature, texture attachment, and visible seams. These poses come from the dataset.':'Actual native-resolution tracker estimates: observe recovery failures, false accepted poses, and the design hiding when tracking is lost.';try{await mediaReady(mapped);ready=true;changing=false;jump(Math.min(t,mapped.duration));if(wasPlaying)await start();}catch(e){error=e.message;changing=false;}}
function draw(){ctx.fillStyle='#080f14';ctx.fillRect(0,0,1920,540);ctx.font='23px system-ui';ctx.fillStyle='#e6eff4';['Original footage','Reference object mask',mode==='supplied'?'Mapped design · supplied poses':'Mapped design · our tracker'].forEach((label,i)=>ctx.fillText(label,i*640+16,34));if(ready&&!changing){const i=Math.min(data.frames.length-1,Math.max(0,Math.floor(mappedFrameTime*data.fps+0.0001)));if(original.readyState>=2){ctx.drawImage(original,...crop,0,52,640,480);ctx.drawImage(original,...crop,640,52,640,480);}const mi=Math.min(data.frames.length-1,Math.max(0,Math.floor(originalFrameTime*data.fps+0.0001)));if(masks[mi])ctx.drawImage(masks[mi],640,52);if(mapped.readyState>=2)ctx.drawImage(mapped,...crop,1280,52,640,480);const f=data.frames[i];status.textContent='Source frame '+f.id+' · '+(mode==='supplied'?'supplied pose':f.state);seek.max=mapped.duration||5.6;seek.value=mapped.currentTime;document.querySelector('#time').textContent=mapped.currentTime.toFixed(1)+' / '+(mapped.duration||5.6).toFixed(1)+' s';if(!mapped.paused&&!original.seeking&&Math.abs(original.currentTime-mapped.currentTime)>.08)original.currentTime=mapped.currentTime;}else{ctx.fillText(error||'Loading synchronized video…',20,110);}requestAnimationFrame(draw);}
play.onclick=()=>mapped.paused?start():pause();document.querySelector('#restart').onclick=()=>{if(ready){jump(0);start();}};seek.oninput=()=>{if(ready)jump(Number(seek.value));};document.querySelector('#control').onclick=()=>switchMode('supplied');document.querySelector('#tracker').onclick=()=>switchMode('estimated');document.querySelector('#fullscreen').onclick=()=>view.requestFullscreen?.();mapped.onended=()=>{pause();jump(0);start();};
Promise.all([mediaReady(original),mediaReady(mapped),...data.frames.map(async(f,i)=>masks[i]=tintedMask(await imageLoaded(f.mask)))]).then(()=>{ready=true;start();}).catch(e=>{error=e.message||'Could not load mask images.';status.textContent=error;});draw();
</script></main></html>'''


def run(root=ROOT):
    cache=root/'.cache'/'hope-hd'
    output=root/'artifacts'/'video-hd'
    manifest=json.loads((cache/'object-000005.json').read_text())
    report=json.loads((output/'object-000005'/'report.json').read_text())
    for name in ('original','supplied-pose','estimated-pose'):
        if (output/'object-000005'/f'{name}.webm').stat().st_size<1024:
            raise ValueError('A completed native video is required.')
    records=manifest['frames'][10:]
    indices=sorted(set(range(0,len(records),report['presentation']['renderStride']))|{len(records)-1})
    frames=[]
    for i in indices:
        record=records[i]
        payload=(cache/record['evaluationMask']['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=record['evaluationMask']['sha256']:
            raise ValueError('Reference mask hash mismatch.')
        if report['frames'][i]['frameId']!=record['frameId']:
            raise ValueError('Saved trace and masks are not synchronized.')
        frames.append(dict(id=record['frameId'],state=report['frames'][i]['state'],
                           mask='data:image/png;base64,'+base64.b64encode(payload).decode('ascii')))
    content=HTML.replace('__DATA__',json.dumps(dict(fps=report['presentation']['fps'],frames=frames),separators=(',',':'))).encode('utf-8')
    # Bounded thin viewer only; existing large-export storage guards stay intact.
    if len(content)>2*1024*1024:
        raise ValueError('Viewer exceeds its 2 MiB limit.')
    if shutil.disk_usage(output).free<len(content)+1024*1024:
        raise OSError('Viewer requires its size plus 1 MiB free disk space.')
    target=output/'visual.html'
    temporary=target.with_suffix('.html.tmp')
    temporary.write_bytes(content)
    temporary.replace(target)
    print(f'Created {target}: {len(frames)} synchronized masks; {len(content)} bytes. No new encoding.')


if __name__=='__main__':
    run()
