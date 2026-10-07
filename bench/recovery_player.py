"""Publish recorded independent poses and image-mask predictions together."""
import json
from .show3d_player import HTML
from .storage import write_artifact
from .ycb_assets import ROOT

def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('Player template changed: '+old[:70])
    return source.replace(old,new)

def build():
    output=ROOT/'artifacts/video-60';baseline=json.loads((output/'report.json').read_text())['objects']
    cases=[];rows=[];comparison=[]
    labels={'keyboard':'Keyboard','mug':'White mug','ranch':'Dressing bottle'}
    for original in baseline:
        alias=original['name'];variants=original['variants'].copy()
        for trial in ('retry-mask','retry-model-mask','retry-learned-mask'):
            variants+=json.loads((output/trial/alias/'report.json').read_text())['variants']
        player=json.loads((output/alias/'player.json').read_text());learned=json.loads((output/'learned-mask'/f'{alias}-full.json').read_text())
        player['variants']=[dict(label=v.get('label',f'Original {v["levels"]}-level tracker'),height=v.get('trackingHeight',original['trackingHeight'])) for v in variants]
        for i,frame in enumerate(player['frames']):
            frame['poses']=[v['frames'][i]['cvCameraFromSourceObject'] if v['frames'][i]['state']=='tracking' else None for v in variants]
            frame['states']=[v['frames'][i]['state'] for v in variants]
            frame['diagnostics']=[v['frames'][i] for v in variants];frame['learned']=learned['frames'][i]
        write_artifact(output/alias/'player-recovery.json',json.dumps(player,separators=(',',':')))
        cases.append(dict(name=alias,label=labels[alias]))
        comparison.append(dict(name=alias,variants=variants,learnedMask=learned))
        for v in variants:
            score=v['score'];error=score.get('p95ErrorPixels');accepted=v['latencyByTrackingState'].get('tracking',{}).get('p95Ms')
            label=v.get('label',f'Original {v["levels"]}-level tracker')
            total=v.get('maskAndTrackerLatency',{}).get('p95Ms')
            rows.append(f'<tr><td>{labels[alias]}</td><td>{label}</td><td>{score["allFrameAvailability"]:.1%}</td><td>{"—" if error is None else f"{error:.1f}"}</td><td>{"—" if accepted is None else f"{accepted:.0f}"}</td><td>{"—" if total is None else f"{total:.0f}"}</td><td>Failed</td></tr>')
        rows.append(f'<tr><td>{labels[alias]}</td><td>Learned mask only; no new pose</td><td>{learned["maskAvailability"]:.1%} masks</td><td>Not measured</td><td>—</td><td>{learned["p95Ms"]:.0f} (mask calls)</td><td>Unvalidated</td></tr>')
    table='<h2>Quality retries — all pose gates still fail</h2><p>Same 240 consecutive frames per object. Pose errors use estimated dataset references, not human ground truth. Longer visibility can include drift. Learned-mask availability measures returned masks, not correctness. CPU timings are fresh desktop measurements, not paired speed comparisons or phone FPS. Learned mask + tracker totals sum separately measured calls, not a live concurrent implementation.</p><table><tr><th>Object</th><th>Trial</th><th>Pose / mask availability</th><th>P95 attachment px (720p)</th><th>Accepted tracker P95 ms</th><th>Model + tracker P95 ms</th><th>Gate</th></tr>'+''.join(rows)+'</table>'
    html=HTML.replace('__TABLE__',table).replace('__CASES__',json.dumps(cases))
    html=replace_once(html,'<title>VisualizeIt · real objects at 60 FPS</title>','<title>VisualizeIt · mask and recovery diagnostics</title>')
    html=replace_once(html,'<h1>Real moving objects · 60 FPS recordings</h1>','<h1>Object masks and tracking recovery</h1>')
    begin=html.index('<label>Mapping mode');end=html.index('</label>',begin)+len('</label>')
    html=html[:begin]+'<label>Mapping mode <select id="mode"><option value="4">Learned mask + model references retry</option><option value="3">Model references + foreground retry</option><option value="2">Foreground + recovery retry</option><option value="1">Original faster tracker</option><option value="0">Original baseline tracker</option><option value="reference">Reference-pose control</option></select></label> <label>Mask view <select id="maskMode"><option value="learned">Learned object + approximate hand exclusion</option><option value="candidate">Current tracker foreground</option><option value="reference">Reference mesh silhouette</option></select></label>'+html[end:]
    begin=html.index('<p class="note">The cyan view');end=html.index('<div id="table">',begin)
    html=html[:begin]+'''<p class="note">Green: independently predicted object mask. Orange: approximate hand cores from landmarks, not an exact hand mask. Cyan in reference mode: geometric mesh silhouette, which can cover the hand. Reference poses never enter the evaluation tracker or learned mask. The new segmentation has not established accurate 6D tracking. Selecting a predicted mask clips the displayed design; unavailable masks hide it. Baseline poses and retries are saved offline results.</p><p class="note">All source and preview frames remain 60 FPS. Inference runs much slower on this desktop; smooth replay is not live tracking. Retry tracking uses 720-pixel image height. No phone or sustained live FPS result. Failed candidates remain visible for diagnosis.</p>'''+html[end:]
    html=replace_once(html,"fetch(c.name+'/player.json')","fetch(c.name+'/player-recovery.json')")
    html=replace_once(html,"['Original footage','Reference mesh silhouette','Mapped design · '+(mode.value==='reference'?'reference pose':mode.value==='1'?'fast tracker':'baseline')]","['Original footage',maskMode.value==='reference'?'Reference mesh silhouette':'Predicted object mask','Mapped design · '+(mode.value==='reference'?'reference pose':'independent tracker')]")
    old="maskRenderer.draw(f.reference,true);ctx.drawImage(maskRenderer.canvas,320,32,320,400);\n mappingRenderer.draw(mode.value==='reference'?f.reference:f.poses[Number(mode.value)],false);ctx.drawImage(mappingRenderer.canvas,640,32,320,400);"
    new="""const diagnostic=mode.value==='reference'?null:f.diagnostics[Number(mode.value)];
 const maskRecord=maskMode.value==='learned'?f.learned:maskMode.value==='candidate'?diagnostic:null;
 const mask=maskRecord?.visibleMaskContours, height=maskMode.value==='learned'?720:(mode.value==='reference'?720:data.variants[Number(mode.value)].height);
 if(maskMode.value==='reference'){maskRenderer.draw(f.reference,true);ctx.drawImage(maskRenderer.canvas,320,32,320,400);}else{
   if(mask){maskPath(mask,320,height);ctx.fillStyle='#29e8aa88';ctx.fill('evenodd');}
   if(maskMode.value==='learned'&&f.learned.handCoreContours){maskPath(f.learned.handCoreContours,320,720);ctx.fillStyle='#ff9c2844';ctx.fill('evenodd');}
   ctx.fillStyle='#fff';ctx.fillText(maskRecord?.maskReason||(!mask?'Mask unavailable':''),328,420);
 }
 mappingRenderer.draw(mode.value==='reference'?f.reference:f.poses[Number(mode.value)],false);
 if(maskMode.value==='reference'){ctx.drawImage(mappingRenderer.canvas,640,32,320,400);}else if(mask){ctx.save();maskPath(mask,640,height);ctx.clip('evenodd');ctx.drawImage(mappingRenderer.canvas,640,32,320,400);ctx.restore();}
 if(diagnostic){const scale=400/data.variants[Number(mode.value)].height;for(const [key,color] of [['supportPixels','#56ff73'],['rejectedPixels','#ff6e64']]){ctx.fillStyle=color;for(const p of diagnostic[key]||[]){ctx.beginPath();ctx.arc(640+p[0]*scale,32+p[1]*scale,1.5,0,Math.PI*2);ctx.fill();}}} """
    html=replace_once(html,old,new)
    html=replace_once(html,'let maskRenderer,mappingRenderer;',"const maskMode=document.querySelector('#maskMode');function maskPath(paths,x,height){ctx.beginPath();for(const path of paths){const points=path.points;if(!points.length)continue;ctx.moveTo(x+points[0][0]*400/height,32+points[0][1]*400/height);for(const p of points.slice(1))ctx.lineTo(x+p[0]*400/height,32+p[1]*400/height);ctx.closePath();}} let maskRenderer,mappingRenderer;")
    html=replace_once(html,'mode.onchange=draw;','mode.onchange=draw;maskMode.onchange=draw;')
    html=replace_once(html,'video.onended=',"video.addEventListener('seeked',()=>{frameIndex=Math.round(video.currentTime*60);draw();});video.onended=")
    html=replace_once(html,'href="report.json"','href="recovery-report.json"')
    write_artifact(output/'recovery.html',html)
    write_artifact(output/'recovery-report.json',json.dumps(dict(schemaVersion=1,objects=comparison),indent=2))

if __name__=='__main__':build()
