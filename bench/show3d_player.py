"""A single decoded video frame drives all three 60 FPS diagnostic views."""
import json
from .storage import write_artifact

HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisualizeIt · real objects at 60 FPS</title>
<style>*{box-sizing:border-box}body{background:#101b21;color:#e6eff4;font:16px system-ui;margin:0}main{max-width:1320px;padding:24px;margin:auto}h1{font-size:28px;margin:0 0 12px}p{max-width:1120px;line-height:1.5;color:#bdd0da}button,select{background:#203c47;color:white;border:1px solid #52818d;border-radius:7px;padding:9px 13px;font:inherit;cursor:pointer}button[aria-pressed=true]{background:#18766d}.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:12px 0}canvas{width:100%;max-height:65vh;object-fit:contain;background:#080f14;border-radius:8px}input{flex:1;min-width:160px;accent-color:#5de2cb}video{position:fixed;bottom:0;left:0;width:1px;height:1px;opacity:.01;pointer-events:none}a{color:#70ebd2}td,th{padding:10px;text-align:left;border-bottom:1px solid #44616f}table{border-collapse:collapse}.note{font-size:14px}#status{color:#70ebd2}#table{overflow-x:auto}</style>
<main><h1>Real moving objects · 60 FPS recordings</h1><p>A keyboard, a white mug and a dressing bottle, held and moved by people. Every consecutive source frame is retained; no interpolation or speed-up.</p>
<div class="controls" id="cases"></div><div class="controls"><label>Mapping mode <select id="mode"><option value="reference">Reference-pose control</option><option value="1">Our tracker · faster candidate</option><option value="0">Our tracker · baseline</option></select></label><span id="status">Loading…</span></div>
<canvas id="view" width="960" height="436" aria-label="Original footage, reference mesh silhouette and mapped surface"></canvas>
<div class="controls"><button id="play">Play</button><button id="restart">Restart</button><input type="range" id="seek" min="0" max="4" step=".001" value="0" aria-label="Playback position"><span id="time">0 / 4 s</span><button id="full">Full screen</button></div>
<p class="note">The cyan view is the <b>projected reference mesh silhouette</b>, not automatic segmentation. Reference poses are dataset estimates with confidence filtering, not human ground truth. Our tracker modes use saved independent estimates and hide the design after a loss. Hand occlusion, lighting and photorealistic appearance are not implemented in this diagnostic renderer.</p>
<p class="note">60 FPS describes the recorded video. Browser drawing rate and measured CPU vision latency are separate. Source: 1024 × 1280 monochrome; preview: 640 × 800; tracking: 288 × 360. Camera acquisition, decoding and rendering are excluded from CPU vision timings. No phone or sustained live FPS claim.</p>
<div id="table">__TABLE__</div><p><a href="report.json">All traces, source hashes and measurement scope</a> · <a href="../video-hd/visual.html">Earlier box experiment</a> · <a href="https://huggingface.co/datasets/facebook/show3d-dataset">SHOW3D source</a></p>
<p class="note">SHOW3D: Rim et al., CVPR 2026; CC BY-NC 4.0. Canonical models: HOT3D / Meta, dataset terms retained. Local research preview.</p>
<video id="video" muted loop playsinline preload="auto"></video><script id="caseData" type="application/json">__CASES__</script>
<script>
const cases=JSON.parse(document.querySelector('#caseData').textContent),video=document.querySelector('#video'),view=document.querySelector('#view'),ctx=view.getContext('2d');
const mode=document.querySelector('#mode'),play=document.querySelector('#play'),seek=document.querySelector('#seek'),status=document.querySelector('#status');
let data=null,frameIndex=0,ready=false,loadNumber=0,paintCount=0,lastWall=performance.now(),drawRate=0;
function renderer(){const canvas=document.createElement('canvas');canvas.width=320;canvas.height=400;const gl=canvas.getContext('webgl2',{alpha:true,premultipliedAlpha:true});if(!gl)throw Error('WebGL 2 unavailable.');
 const vertex=`#version 300 es
 in vec3 aP;in vec3 aN;in vec2 aUV;uniform mat4 pose;uniform vec4 intrinsics;uniform vec2 size;out vec2 uv;out vec3 normal;
 void main(){vec3 p=(pose*vec4(aP,1.)).xyz;float A=(20.+.03)/(20.-.03),B=2.*20.*.03/(20.-.03);gl_Position=vec4(2.*intrinsics.x*p.x/size.x+(2.*intrinsics.z/size.x-1.)*p.z,(1.-2.*intrinsics.w/size.y)*p.z-2.*intrinsics.y*p.y/size.y,A*p.z-B,p.z);uv=aUV;normal=mat3(pose)*aN;}`;
 const fragment=`#version 300 es
 precision highp float;in vec2 uv;in vec3 normal;uniform bool silhouette;out vec4 color;
 void main(){if(silhouette){color=vec4(0.,.85,.9,.45);}else{float c=mod(floor(uv.x*32.)+floor(uv.y*32.),2.);vec3 base=mix(vec3(.90,.94,.91),vec3(.06,.45,.37),c);float light=.65+.35*max(0.,dot(normalize(normal),normalize(vec3(-.3,-.4,-1.))));color=vec4(base*light,1.);}}`;
 function shader(kind,source){const s=gl.createShader(kind);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
 const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
 const buffer=gl.createBuffer(),elements=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);for(const [name,count,offset] of [['aP',3,0],['aN',3,12],['aUV',2,24]]){const a=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,count,gl.FLOAT,false,32,offset);}gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,elements);
 gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);let indices=0;
 return {canvas,load(bytes,d){gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(bytes,0,d.vertices*8),gl.STATIC_DRAW);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,elements);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint32Array(bytes,d.vertices*32,d.indices),gl.STATIC_DRAW);indices=d.indices;gl.uniform4f(gl.getUniformLocation(program,'intrinsics'),d.k[0],d.k[4],d.k[2],d.k[5]);gl.uniform2f(gl.getUniformLocation(program,'size'),d.width,d.height);},draw(matrix,silhouette){gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);if(matrix){gl.uniformMatrix4fv(gl.getUniformLocation(program,'pose'),false,new Float32Array([0,1,2,3].flatMap(c=>matrix.map(row=>row[c]))));gl.uniform1i(gl.getUniformLocation(program,'silhouette'),silhouette?1:0);gl.drawElements(gl.TRIANGLES,indices,gl.UNSIGNED_INT,0);}}};}
let maskRenderer,mappingRenderer;try{maskRenderer=renderer();mappingRenderer=renderer();}catch(e){status.textContent=e.message;}
function draw(){ctx.fillStyle='#080f14';ctx.fillRect(0,0,960,436);ctx.fillStyle='#e6eff4';ctx.font='14px system-ui';['Original footage','Reference mesh silhouette','Mapped design · '+(mode.value==='reference'?'reference pose':mode.value==='1'?'fast tracker':'baseline')].forEach((label,i)=>ctx.fillText(label,i*320+8,24));if(!ready||video.readyState<2)return;
 const f=data.frames[Math.min(data.frames.length-1,frameIndex)];for(let i=0;i<3;i++)ctx.drawImage(video,i*320,32,320,400);
 maskRenderer.draw(f.reference,true);ctx.drawImage(maskRenderer.canvas,320,32,320,400);
 mappingRenderer.draw(mode.value==='reference'?f.reference:f.poses[Number(mode.value)],false);ctx.drawImage(mappingRenderer.canvas,640,32,320,400);
 status.textContent='Frame '+f.frameId+' · '+(mode.value==='reference'?'dataset reference':f.states[Number(mode.value)])+' · browser draw '+drawRate.toFixed(0)+' FPS';
 seek.value=video.currentTime;document.querySelector('#time').textContent=video.currentTime.toFixed(2)+' / 4 s';}
function onFrame(_,metadata){frameIndex=Math.round(metadata.mediaTime*60);paintCount++;const now=performance.now();if(now-lastWall>=1000){drawRate=paintCount*1000/(now-lastWall);paintCount=0;lastWall=now;}draw();video.requestVideoFrameCallback(onFrame);}
if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(onFrame);else status.textContent='Decoded-frame callbacks unavailable; use a current Chromium browser.';
async function start(){if(!ready)return;try{await video.play();play.textContent='Pause';}catch(e){play.textContent='Play';}}
async function selectCase(c){const ticket=++loadNumber;video.pause();ready=false;status.textContent='Loading '+c.label+'…';document.querySelectorAll('#cases button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.name===c.name)));
 try{const [d,bytes]=await Promise.all([fetch(c.name+'/player.json').then(r=>{if(!r.ok)throw Error('Missing clip data.');return r.json();}),fetch(c.name+'/mesh.bin').then(r=>{if(!r.ok)throw Error('Missing mesh.');return r.arrayBuffer();})]);if(ticket!==loadNumber)return;data=d;maskRenderer.load(bytes,d);mappingRenderer.load(bytes,d);frameIndex=0;video.src=c.name+'/original.webm';video.load();await new Promise((resolve,reject)=>{video.addEventListener('loadeddata',resolve,{once:true});video.addEventListener('error',()=>reject(Error('Video load failed.')),{once:true});});if(ticket!==loadNumber)return;ready=true;paintCount=0;lastWall=performance.now();draw();start();}catch(e){status.textContent=e.message;}}
for(const c of cases){const button=document.createElement('button');button.textContent=c.label;button.dataset.name=c.name;button.onclick=()=>selectCase(c);document.querySelector('#cases').append(button);}mode.onchange=draw;play.onclick=()=>{if(video.paused)start();else{video.pause();play.textContent='Play';}};document.querySelector('#restart').onclick=()=>{video.currentTime=0;start();};seek.oninput=()=>{if(ready)video.currentTime=Number(seek.value);};document.querySelector('#full').onclick=()=>view.requestFullscreen?.();video.onended=()=>{video.currentTime=0;start();};if(maskRenderer&&mappingRenderer)selectCase(cases[0]);
</script></main></html>'''


def build(output,reports):
    labels={'keyboard':'Keyboard','mug':'White mug','ranch':'Dressing bottle'}
    cases=[dict(name=r['name'],label=labels[r['name']]) for r in reports]
    rows=[]
    for r in reports:
        for v in r['variants']:
            score=v['score'];error=score.get('p95ErrorPixels')
            accepted=v.get('latencyByTrackingState',{}).get('tracking',{})
            accepted_p95=accepted.get('p95Ms');samples=accepted.get('samples',0)
            rows.append(f'<tr><td>{labels[r["name"]]}</td><td>{v["levels"]} levels</td><td>{v["medianMs"]:.2f}</td><td>{v["p95Ms"]:.2f}</td><td>{"—" if accepted_p95 is None else f"{accepted_p95:.2f}"} ({samples})</td><td>{score["allFrameAvailability"]:.1%}</td><td>{"—" if error is None else f"{error:.1f}"}</td><td>{v["accuracyAnd60FpsGatePassed"]}</td></tr>')
    table='<h2>CPU vision results</h2><p class="note">All 240 evaluation frames count toward availability. Most frames lose tracking; fast failure is not usable high FPS. Accepted-pose latency and sample counts are shown separately, with five warmup frames omitted only from timings. Errors compare against estimated dataset references and use 720p-equivalent pixels. The evaluator has no visible-hand segmentation. The provisional 60 FPS vision budget is 10 ms P95 within a 16.67 ms total frame budget. Every combined gate fails.</p><table><tr><th>Object</th><th>Candidate</th><th>All-state median ms</th><th>All-state P95 ms</th><th>Accepted P95 ms (n)</th><th>Availability</th><th>P95 error px</th><th>Accuracy + speed gate</th></tr>'+''.join(rows)+'</table>'
    write_artifact(output/'index.html',HTML.replace('__TABLE__',table).replace('__CASES__',json.dumps(cases)))
