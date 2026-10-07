"""Compare unchanged baseline and safeguarded metric mapping candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .mapping_candidates import candidates, design_uv, diagnostic_color, seam_design_quality
from .model import Mesh
from .obj_model import read_obj
from .real_objects import camera_for
from .renderer import render
from .scan_mapping import unwrap
from .ycb_assets import ROOT, OBJECTS, SOURCE
from .storage import save_image, save_arrays, write_artifact


def artwork(size=512):
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(1, 0, size))
    return (diagnostic_color(np.column_stack([x.ravel(), y.ravel()]), False).reshape(size, size, 3)*255).astype(np.uint8)


def run(output, assets, design=None, tile=.04, rotation=0):
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schemaVersion=2, physicalValidationPassed=False, source=SOURCE,
                  scope='Metric chart improvement; synthetic cameras; geometry unchanged', objects=[])
    texture = artwork()
    if design:
        with Image.open(design) as imported:
            if imported.width*imported.height > 16000000: raise ValueError('Design exceeds 16 megapixels.')
            background = Image.new('RGBA', imported.size, (255,255,255,255))
            background.alpha_composite(imported.convert('RGBA'))
            texture = np.asarray(background.convert('RGB'))
    save_image(Image.fromarray(texture), output/'diagnostic-artwork.png')
    x, y = np.meshgrid(np.arange(128), np.arange(128))
    checker = np.where(((x//16+y//16)%2)[..., None], [25,100,92], [230,238,232]).astype(np.uint8)
    stripes = np.where(((x//16)%2)[..., None], [30,70,200], [245,200,40]).astype(np.uint8)
    labelled = Image.fromarray(artwork()); pen = ImageDraw.Draw(labelled)
    pen.text((25,25),'TOP / ONE ARTWORK',fill='white',stroke_width=2,stroke_fill='black')
    pen.line([(256,450),(256,90),(230,120),(256,90),(282,120)],fill='white',width=12)
    patterns = dict(checker=checker, stripes=stripes, labelled_artwork=np.asarray(labelled))
    for key, pixels in patterns.items(): save_image(Image.fromarray(pixels), output/f'{key}.png')
    for identifier, name in OBJECTS.items():
        scan = read_obj(assets/identifier/'textured.obj', name)
        mesh, _, _, _, details = unwrap(scan, name, details=True)
        directory = output/identifier
        directory.mkdir(exist_ok=True)
        visibility = np.zeros(len(mesh.triangles))
        views = []
        for i in range(8):
            camera = camera_for(scan.positions, i*45)
            frame = render(mesh, camera)
            visibility[np.unique(frame.triangle[frame.triangle >= 0])] += 1/8
            views.append(camera)
        item = dict(objectId=identifier, name=name, modes={})
        for repeating, mode in ((True, 'repeat'), (False, 'artwork')):
            alternatives, selected = candidates(mesh, details, visibility, repeating, texture if design else None,tile,rotation)
            for uv, q, _, _ in alternatives:
                q['designSensitivity'] = {key: {k:v for k,v in seam_design_quality(mesh,details['sourceFaces'],details['chartIds'],uv,repeating,pixels,tile,rotation).items() if k != 'seamFaces'} for key,pixels in patterns.items()}
            record = dict(selected=alternatives[selected][1]['candidate'], candidates=[r[1] for r in alternatives], images={},
                          patternSettings=dict(tileWidthMetres=tile,rotationDegrees=rotation,imageAspect=texture.shape[0]/texture.shape[1]))
            record['selectionStatus'] = 'fallback_control' if alternatives[selected][1]['rejectionReasons'] else 'selected_eligible'
            for index, label in ((0, 'baseline'), (selected, 'selected')):
                metric, q, _, trusted = alternatives[index]
                display = Mesh(name, mesh.positions, mesh.normals, design_uv(metric, repeating,tile=tile, aspect=texture.shape[0]/texture.shape[1],rotation_degrees=rotation), mesh.triangles)
                image = render(display, views[0], texture_image=texture, texture_repeat=repeating)
                path = f'{identifier}/{mode}-{label}.png'
                save_image(Image.fromarray(image.color), output/path)
                record['images'][label] = path
                if label == 'baseline':
                    baseline_frames = [Image.fromarray(render(display,view,texture_image=texture,texture_repeat=repeating).color) for view in views]
                    path = f'{identifier}/{mode}-baseline-orbit.gif'
                    save_image(baseline_frames[0],output/path,save_all=True,append_images=baseline_frames[1:],duration=350,loop=0)
                    record['images']['baselineOrbit'] = path
                    save_arrays(directory/f'{mode}-baseline-asset.npz',schemaVersion=2,positions=mesh.positions,normals=mesh.normals,
                                triangles=mesh.triangles,metricUV=metric,sourceVertices=details['sourceVertices'],
                                sourceFaces=details['sourceFaces'],chartIds=details['chartIds'],trustedFace=trusted)
                if label == 'selected':
                    frame_list = []
                    for view in views:
                        frame_list.append(Image.fromarray(render(display, view, texture_image=texture, texture_repeat=repeating).color))
                    path = f'{identifier}/{mode}-orbit.gif'
                    save_image(frame_list[0], output/path, save_all=True, append_images=frame_list[1:], duration=350, loop=0)
                    record['images']['orbit'] = path
                    colors = np.where(trusted[:, None], [30, 150, 126], [235, 120, 45])
                    colors[q['seamFaces']] = [160, 55, 160]
                    path = f'{identifier}/{mode}-seams.png'
                    save_image(Image.fromarray(render(display, views[0], face_colors=colors).color), output/path)
                    record['images']['seams'] = path
                    save_arrays(directory/f'{mode}-asset.npz', schemaVersion=2, positions=mesh.positions,
                                        normals=mesh.normals, triangles=mesh.triangles, metricUV=metric,
                                        sourceVertices=details['sourceVertices'], sourceFaces=details['sourceFaces'],
                                        chartIds=details['chartIds'], trustedFace=trusted)
            item['modes'][mode] = record
            q = alternatives[selected][1]
            print(f"{name} {mode}: selected {record['selected']}, seam improvement {q['meanSeamColorImprovement']*100:.1f}%, gate {q['provisionalImprovementGatePassed']}", flush=True)
        report['objects'].append(item)
        write_artifact(output/'report.json',json.dumps(report, indent=2, allow_nan=False)+'\n')
    sections = []
    for item in report['objects']:
        for mode, record in item['modes'].items():
            q = next(q for q in record['candidates'] if q['selected'])
            images = ''.join(f'<figure><img src="{record["images"][k]}"><figcaption>{k}</figcaption></figure>' for k in ('baseline', 'selected', 'baselineOrbit', 'orbit', 'seams') if k in record['images'])
            rows = ''.join(f'<tr><td>{q1["candidate"]}</td><td>{q1["meanSeamColorError"]:.3f}</td><td>{q1["areaWeightedP95ScaleError"]*100:.2f}%</td><td>{", ".join(q1["rejectionReasons"]) or "eligible"}</td></tr>' for q1 in record['candidates'])
            fallback = ' No eligible candidate; the displayed baseline is a failed comparison control.' if q['rejectionReasons'] else ''
            sections.append(f'<h2>{item["name"]} / {mode}</h2><p>Selected {record["selected"]}; seam improvement {q["meanSeamColorImprovement"]*100:.1f}%; improvement gate {q["provisionalImprovementGatePassed"]}.{fallback}</p><div class="grid">{images}</div><table><tr><th>Candidate</th><th>Seam colour error</th><th>P95 scale error</th><th>Safeguards</th></tr>{rows}</table>')
    html = '<!doctype html><meta charset="utf-8"><title>Mapping comparisons</title><style>body{font:17px system-ui;margin:30px;background:#edf2ef;color:#183f37}.grid{display:grid;grid-template-columns:1fr 1fr}figure{margin:6px}img{width:100%}td,th{padding:8px;text-align:left}table{background:white}</style><h1>Mapping candidates</h1><p>Real YCB scan geometry, synthetic camera views. Purple marks seam-adjacent faces; teal meets local limits; orange fails them. Results do not establish natural AR. Diagnostic artwork is asymmetric to expose discontinuity.</p>'+''.join(sections)+'<p>YCB data: Calli et al., CC BY 4.0. Coordinate placement and texture replaced; geometry retained. <a href="report.json">Raw metrics</a></p>'
    write_artifact(output/'index.html',html)


def rebuild_baselines(output,assets):
    """Upgrade existing reports without repeating measurements or selected renders."""
    report=json.loads((output/'report.json').read_text())
    texture=np.asarray(Image.open(output/'diagnostic-artwork.png').convert('RGB'))
    for item in report['objects']:
        identifier=item['objectId']; scan=read_obj(assets/identifier/'textured.obj',item['name'])
        mesh,_,_,_,details=unwrap(scan,item['name'],details=True)
        for mode,record in item['modes'].items():
            record.setdefault('patternSettings',dict(tileWidthMetres=.04,rotationDegrees=0,imageAspect=texture.shape[0]/texture.shape[1]))
            settings=record['patternSettings']; repeating=mode=='repeat'
            baseline=record['candidates'][0]
            check=seam_design_quality(mesh,details['sourceFaces'],details['chartIds'],details['metricUV'],repeating)
            if abs(check['meanSeamColorError']-baseline['meanSeamColorError'])>1e-8:
                raise ValueError('Regenerated atlas differs from recorded baseline; rerun the full experiment.')
            display=Mesh(mesh.name,mesh.positions,mesh.normals,design_uv(details['metricUV'],repeating,
                settings['tileWidthMetres'],settings['imageAspect'],settings['rotationDegrees']),mesh.triangles)
            frames=[Image.fromarray(render(display,camera_for(scan.positions,i*45),texture_image=texture,texture_repeat=repeating).color) for i in range(8)]
            path=f'{identifier}/{mode}-baseline-orbit.gif'
            save_image(frames[0],output/path,save_all=True,append_images=frames[1:],duration=350,loop=0)
            record['images']['baselineOrbit']=path
            print(f'{item["name"]} {mode}: verified and rendered eight unchanged baseline views',flush=True)
        write_artifact(output/'report.json',json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Insert both orbit views into the already generated, otherwise unchanged gallery.
    html=(output/'index.html').read_text(encoding='utf-8')
    for item in report['objects']:
        for mode,record in item['modes'].items():
            old=f'<figure><img src="{record["images"]["orbit"]}"><figcaption>orbit</figcaption></figure>'
            new=f'<figure><img src="{record["images"]["baselineOrbit"]}"><figcaption>baseline orbit</figcaption></figure>'+old
            html=html.replace(old,new)
    write_artifact(output/'index.html',html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/'mapping-improvements')
    parser.add_argument('--assets', type=Path, default=ROOT/'.cache'/'ycb')
    parser.add_argument('--design', type=Path, help='Optional RGB PNG/JPEG artwork; aspect retained. Alpha is flattened to RGB in this diagnostic bench.')
    parser.add_argument('--tile-metres',type=float,default=.04)
    parser.add_argument('--rotation-degrees',type=float,default=0)
    parser.add_argument('--rebuild-baseline-orbits',action='store_true',help='Add verified baseline orbits to an existing default-design report.')
    args = parser.parse_args()
    if not np.isfinite(args.tile_metres) or args.tile_metres <= 0 or not np.isfinite(args.rotation_degrees): parser.error('Pattern size must be positive and rotation finite.')
    if args.rebuild_baseline_orbits:
        if args.design or args.tile_metres!=.04 or args.rotation_degrees!=0: parser.error('Upgrade supports existing default diagnostic reports only.')
        rebuild_baselines(args.output,args.assets)
    else: run(args.output, args.assets, args.design,args.tile_metres,args.rotation_degrees)
