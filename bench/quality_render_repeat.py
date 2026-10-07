"""Renderer repeatability diagnostic; setup pose only, no tracking score."""
import argparse
import hashlib
import json
import numpy as np
from .quality_assets import CACHE, save_result
from .quality_gotrack import TexturedRenderer, upstream_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--flat', action='store_true', help='Diagnostic unlit texture rendering, not a tracking default')
    args = parser.parse_args(); upstream_path()
    if args.flat:
        import pyrender
        original_render = pyrender.OffscreenRenderer.render
        def flat_render(renderer, scene, flags=0, seg_node_map=None):
            return original_render(renderer, scene, flags|pyrender.RenderFlags.FLAT, seg_node_map)
        pyrender.OffscreenRenderer.render = flat_render
    from utils import structs, renderer_base
    root = CACHE/'inputs/keyboard'; manifest = json.loads((root/'input.json').read_text())
    pose = np.linalg.inv(np.array(manifest['controlled_initial_pose'])); pose[:3, 3] *= 1000
    k = np.array(manifest['intrinsics'])
    def camera(h):
        w = round(h*manifest['native_resolution'][0]/manifest['native_resolution'][1]); scale = h/manifest['native_resolution'][1]
        return structs.PinholePlaneCameraModel(width=w, height=h,
            f=(k[0, 0]*scale, k[1, 1]*scale), c=(k[0, 2]*scale, k[1, 2]*scale), T_world_from_eye=pose)
    renderers = [TexturedRenderer(root/manifest['asset'], manifest['object_id']) for _ in range(2)]
    rows = []
    try:
        for mode in ('steady', 'shared_context_resize', 'separate_depth_context'):
            for i in range(10):
                if mode != 'steady': renderers[int(mode == 'separate_depth_context')].render_object_model(manifest['object_id'], camera(640))
                output = renderers[0].render_object_model(manifest['object_id'], camera(280))
                rows.append(dict(mode=mode, repetition=i,
                    color_sha256=hashlib.sha256(output[renderer_base.RenderType.COLOR].tobytes()).hexdigest(),
                    depth_sha256=hashlib.sha256(output[renderer_base.RenderType.DEPTH].tobytes()).hexdigest()))
    finally:
        for renderer in renderers: renderer.close()
    report = dict(scope=__doc__, flat_diagnostic=args.flat, frames=rows,
        unique_color_hashes={m:len({r['color_sha256'] for r in rows if r['mode']==m}) for m in {r['mode'] for r in rows}},
        unique_depth_hashes={m:len({r['depth_sha256'] for r in rows if r['mode']==m}) for m in {r['mode'] for r in rows}})
    save_result(CACHE/'diagnostics'/args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k!='frames'}, indent=2), flush=True)


if __name__ == '__main__': main()
