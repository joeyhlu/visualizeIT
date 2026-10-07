"""Show real RGB input, actual working luminance and saved estimated overlays."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .hd_experiment import image_for, tracking_input, overlay, gallery
from .ply_model import read_ply, pose_camera
from .scan_mapping import unwrap
from .mapping_candidates import correct_scale
from .video_experiment import display_mesh
from .storage import save_image, write_artifact
from . import storage
from .ycb_assets import ROOT


def run():
    root=ROOT/'.cache'/'hope-hd';output=ROOT/'artifacts'/'video-hd'
    study=json.loads((output/'performance-360-gray'/'report.json').read_text())
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    title=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',30)
    canvas=Image.new('RGB',(1560,1220),'#eff4f2');draw=ImageDraw.Draw(canvas)
    draw.text((25,18),'Recorded HD tests — actual frames and saved estimates',font=title,fill='#183f37')
    draw.text((25,59),'Source: 1920 × 1440. Fast vision: 480 × 360 grayscale. No live phone camera.',font=font,fill='#183f37')
    y=115
    for report in study['objects']:
        manifest=json.loads((root/f'object-{report["objectId"]:06d}.json').read_text())
        scan,_,transform=read_ply(root/manifest['model']['path'],report['name'])
        mesh,_,_,_,details=unwrap(scan,report['name'],details=True)
        mesh=display_mesh(mesh,correct_scale(mesh,details['metricUV'],details['chartIds']),None,'repeat',.025,0)
        for index in (0,20):
            record=manifest['frames'][10+index];estimate=report['frames'][index];rgb=image_for(root,record)
            crop=(400,300,1520,1140);x0,y0,x1,y1=crop
            source=rgb[y0:y1,x0:x1].copy();working,_=tracking_input(rgb,record['calibration'],360,True)
            working_crop=working[75:285,100:380]
            mapped=source.copy()
            if estimate['state']=='tracking':
                matrix=np.array(estimate['cvCameraFromSourceObject']);k=np.array(record['calibration']['cam_K'],dtype=float).reshape(3,3)
                k[0,2]-=x0;k[1,2]-=y0
                camera=pose_camera(matrix[:3,:3],matrix[:3,3],dict(cam_K=k.ravel().tolist()),transform,x1-x0,y1-y0)
                mapped=overlay(source,mesh,camera)
            label=f'{report["name"]}, frame {record["frameId"]}: {estimate["state"]}'
            draw.text((25,y),label,font=font,fill='#183f37');y+=32
            for column,(label,image) in enumerate((('Original RGB crop',Image.fromarray(source)),
                ('Actual luminance input (enlarged)',Image.fromarray(working_crop).convert('RGB')),
                ('Saved estimate; hidden when lost',Image.fromarray(mapped)))):
                x=25+column*510
                canvas.paste(image.resize((490,250),Image.Resampling.NEAREST if column==1 else Image.Resampling.LANCZOS),(x,y))
                draw.text((x,y+254),label,font=font,fill='#183f37')
            y+=295
    # Each row is a literal image crop. The middle panel makes the reduced input visible.
    canvas=canvas.crop((0,0,1560,y+12));save_image(canvas,output/'progress.jpg',quality=95)
    variants=[]
    for directory in ('performance-480-compact','performance-480-indexed','performance-360-gray'):
        file=output/directory/'report.json'
        if file.exists():
            for item in json.loads(file.read_text())['objects']:
                variants.append(dict(directory=directory,**{key:item[key] for key in ('name','trackingHeight','trackingInputAndVisionMedianMs','trackingInputAndVisionP95Ms','score','qualityAndSpeedGatePassed')}))
    rows=''.join(f'<tr><td>{v["directory"]}</td><td>{v["name"]}</td><td>{v["trackingInputAndVisionMedianMs"]:.1f}</td><td>{v["trackingInputAndVisionP95Ms"]:.1f}</td><td>{v["score"]["allFrameAvailability"]:.1%}</td><td>{v["qualityAndSpeedGatePassed"]}</td></tr>' for v in variants)
    html='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>What the tracker is doing</title><style>body{font:17px system-ui;margin:28px;background:#eff4f2;color:#183f37}img{width:100%;max-width:1500px}td,th{padding:10px;border-bottom:1px solid #abc}p{max-width:1000px}</style><h1>What the tracker is doing</h1><p>Actual recorded frames: original RGB, the reduced luminance image, and the overlay from saved estimated poses. Frame 10 is the first held-out frame; frame 30 is later in the same orbit. Lost/limited estimates hide the design. Accepted estimates can still be inaccurate; all attachment gates below failed.</p><img src="progress.jpg"><h2>30 FPS target</h2><p>33.3 ms total frame budget. Timings below include CPU resize and vision, excluding decoding and rendering. Full phone FPS remains untested. Speed alone does not qualify the tracker.</p><table><tr><th>Candidate</th><th>Object</th><th>Median ms</th><th>P95 ms</th><th>Availability</th><th>Speed + quality passed</th></tr>'+rows+'</table><p><a href="index.html">Native HD footage</a> · <a href="performance-360-gray/index.html">Fast candidate report</a> · <a href="performance-480-indexed/index.html">Indexed candidate report</a></p><p>Intel Core i5-11400F desktop, 16 GB RAM; memory and storage pressure varied during runs. This is replay evidence, not a phone performance result. NVIDIA HOPE/BOP source; Tyree et al.; CC BY-NC-SA 4.0 per bundled metadata.</p>'
    write_artifact(output/'progress.html',html)
    print('Created actual-frame progress.jpg and comparison page.',flush=True)


if __name__=='__main__':
    storage.RESERVE=128*1024*1024
    run()
