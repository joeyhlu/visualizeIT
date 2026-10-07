"""Mapping baseline on attributed scans of real objects; no estimated poses or physical AR."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .model import Camera
from .obj_model import read_obj
from .renderer import render
from .scan_mapping import unwrap
from .ycb_assets import ROOT, OBJECTS, SHA256, SOURCE


def camera_for(positions, phase=0):
    bounds = positions.max(axis=0)-positions.min(axis=0)
    target = np.array([0, bounds[1]/2, 0])
    # Frame the unchanged physical geometry; small objects get a closer camera.
    radius = np.linalg.norm(bounds)/2
    distance = max(.23, radius*3.8)
    angle = np.deg2rad(30+phase)
    eye = target+distance*np.array([np.sin(angle), .6, -np.cos(angle)])
    return Camera.look_at(eye, target, width=960, height=540, focal=637.5)


def font(size):
    path = Path('C:/Windows/Fonts/segoeui.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def contact_sheet(output, objects):
    width, row_height = 1320, 318
    image = Image.new('RGB', (width, 170+len(objects)*row_height), '#f0f4f2')
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), 'VisualizeIt: mapping on five real object scans', font=font(30), fill='#173e39')
    draw.text((28, 65), 'YCB scanned geometry | synthetic cameras | no physical tracking validation', font=font(20), fill='#35574f')
    draw.text((28, 92), 'Display crops enlarged for inspection; camera intrinsics and full frames are saved in the report.', font=font(15), fill='#35574f')
    for column, title in enumerate(('Original photographic texture', 'Applied 5 mm checker squares', 'Mapping limits: teal passes / orange fails')):
        draw.text((28+column*432, 119), title, font=font(17), fill='#173e39')
    for row, item in enumerate(objects):
        y = 170+row*row_height
        q = item['quality']
        caption = (f"{item['name']}   |   local scale error: median {q['areaWeightedMedianScaleError']*100:.1f}%, "
                   f"p95 {q['areaWeightedP95ScaleError']*100:.1f}%   |   within strict limits: {q['areaWithinProvisionalMappingLimits']*100:.1f}% of area")
        draw.text((28, y), caption, font=font(19), fill='#173e39')
        original = Image.open(output/item['images']['original']).convert('RGB')
        coordinates = np.argwhere(np.any(np.asarray(original) != [239, 243, 241], axis=2))
        low = np.maximum(coordinates.min(axis=0)-20, 0)
        high = np.minimum(coordinates.max(axis=0)+21, [original.height, original.width])
        crop = (int(low[1]), int(low[0]), int(high[1]), int(high[0]))
        for column, mode in enumerate(('original', 'mapped', 'quality')):
            tile = ImageOps.contain(Image.open(output/item['images'][mode]).convert('RGB').crop(crop), (416, 234))
            image.paste(tile, (28+column*432+(416-tile.width)//2, y+36+(234-tile.height)//2))
        draw.text((28, y+275), f"{q['charts']} charts; {q['chartSeamEdges']} seam edges. Continuity is not solved; none passes the full release gate.", font=font(16), fill='#735038')
    image.save(output/'real-object-results.png')


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(title='Real object scan mapping baseline', datasetSource=SOURCE,
                  cameras='Known synthetic poses, not a tracker', physicalValidationPassed=False,
                  geometry='Original YCB google_16k meshes; no new reconstruction or geometry completion',
                  units='Source OBJ coordinates treated as metres; nominal dimensions not independently measured',
                  illumination='Basic diffuse reference rendering; not Unity PBR or real camera compositing',
                  mapping='xatlas 0.0.11 charts at metric texel density; 4 cm tiles / 5 mm checker squares',
                  limits='Provisional per-face scale error <=2% and anisotropy <=1.05; seamlessness is evaluated separately',
                  objects=[])
    for object_id, name in OBJECTS.items():
        started = time.perf_counter()
        source_dir = args.assets/object_id
        digest = hashlib.sha256((args.assets/f'{object_id}.tgz').read_bytes()).hexdigest()
        if digest != SHA256[object_id]:
            raise ValueError(f'{name}: source archive differs from pinned scan.')
        scan = read_obj(source_dir/'textured.obj', name)
        mapped, quality, errors, trusted = unwrap(scan, name)
        print(f"{name}: unwrapped {len(scan.triangles)} triangles, p95 scale error {quality['areaWeightedP95ScaleError']*100:.2f}%, valid-limit area {quality['areaWithinProvisionalMappingLimits']*100:.1f}%", flush=True)
        directory = args.output/object_id
        directory.mkdir(exist_ok=True)
        texture = np.asarray(Image.open(source_dir/'texture_map.png').convert('RGB'))
        camera = camera_for(scan.positions)
        colors = np.where(trusted[:, None], [29, 150, 126], [235, 123, 48])
        colors[~np.isfinite(errors)] = [171, 57, 123]
        images = {}
        for mode, mesh, kwargs in (('original', scan.original, {'texture_image': texture}),
                                   ('mapped', mapped, {}), ('quality', mapped, {'face_colors': colors})):
            Image.fromarray(render(mesh, camera, **kwargs).color).save(directory/f'{mode}.png')
            images[mode] = f'{object_id}/{mode}.png'
        orbit, cameras = [], []
        for i in range(args.views):
            view = camera_for(scan.positions, i*360/args.views)
            pixels = render(mapped, view).color
            path = f'{object_id}/orbit-{i:02d}.png'
            Image.fromarray(pixels).save(args.output/path)
            orbit.append(path)
            cameras.append(view.manifest())
        frames = [Image.open(args.output/path).convert('RGB') for path in orbit]
        frames[0].save(directory/'orbit.gif', save_all=True, append_images=frames[1:], duration=350, loop=0)
        np.savez_compressed(directory/'mapped-mesh.npz', positions=mapped.positions, normals=mapped.normals,
                            triangles=mapped.triangles, textureUV=mapped.uv, faceScaleError=errors, trustedFace=trusted)
        item = dict(objectId=object_id, name=name, source=json.loads((source_dir/'source.json').read_text()),
                    boundsMetres=(scan.positions.max(axis=0)-scan.positions.min(axis=0)).tolist(),
                    originalVertices=len(scan.positions), repairedNormalVertices=scan.repairedNormalVertices,
                    quality=quality, images=images, orbit=orbit, cameras=cameras,
                    referenceRenderSeconds=round(time.perf_counter()-started, 2))
        (directory/'metrics.json').write_text(json.dumps(item, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        report['objects'].append(item)
        print(f'{name}: rendered {len(orbit)} known-pose views; {item["referenceRenderSeconds"]} s CPU batch (not phone performance)', flush=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    contact_sheet(args.output, report['objects'])
    template = (ROOT/'bench'/'real_objects.html').read_text(encoding='utf-8')
    embedded = json.dumps(report, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    (args.output/'index.html').write_text(template.replace('__REPORT_JSON__', embedded), encoding='utf-8')
    print(f'Wrote {args.output}/index.html and real-object-results.png', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=ROOT/'.cache'/'ycb')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/'real-objects')
    parser.add_argument('--views', type=int, default=8)
    args = parser.parse_args()
    if args.views < 4 or args.views > 36:
        parser.error('Use 4–36 views.')
    run(args)
