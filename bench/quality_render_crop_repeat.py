"""Repeat the actual GoTrack crop/render path without loading its network."""
import argparse
import hashlib
import json
import numpy as np
from .quality_assets import CACHE, save_result
from .quality_gotrack import TexturedRenderer, upstream_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nearest', action='store_true')
    parser.add_argument('--no-msaa', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); upstream_path()
    import torch
    import pyrender
    if args.no_msaa:
        # Process-local diagnostic only. Preserve installed dependency source.
        import pyrender.renderer as render_module
        original_enable = render_module.glEnable
        def enable(capability):
            if capability == render_module.GL_MULTISAMPLE:
                render_module.glDisable(capability)
            else: original_enable(capability)
        render_module.glEnable = enable
    from utils import data_util, structs
    torch.set_num_threads(1)
    root = CACHE/'inputs/keyboard'; m = json.loads((root/'input.json').read_text())
    renderer = TexturedRenderer(root/m['asset'], m['object_id'], unlit=True)
    if args.nearest:
        for node in renderer.scene.mesh_nodes:
            for primitive in node.mesh.primitives:
                texture = primitive.material.baseColorTexture
                if texture is not None: texture.sampler = pyrender.Sampler(magFilter=9728, minFilter=9728)
    k = np.array(m['intrinsics']); w,h = m['native_resolution']
    camera = structs.PinholePlaneCameraModel(width=w, height=h, f=(k[0,0],k[1,1]), c=(k[0,2],k[1,2]), T_world_from_eye=np.eye(4))
    pose = np.array(m['controlled_initial_pose'], np.float32); pose[:3,3] *= 1000
    rows = []
    try:
        for variant in range(3):
            p = pose.copy(); p[0,3] += variant*4.13; p[1,3] -= variant*2.77
            for repetition in range(10):
                data, cameras, transform = data_util.compute_gotrack_inputs_from_init_poses(
                    input_rgbs=torch.zeros((1,3,h,w)), input_cameras=[camera], init_poses_cam_from_model=torch.from_numpy(p)[None],
                    renderer=renderer, obj_ids=[m['object_id']], object_vertices=[renderer.vertices_m*1000],
                    crop_size=(280,280), crop_rel_pad=.1, cropping_type='perspective_2d_box', ssaa_factor=1., background_type='gray',
                    input_masks=torch.ones((1,h,w)))
                template = data['templates']; color = template.rgbs.numpy(); depth = template.depths.numpy()
                rows.append(dict(variant=variant,repetition=repetition,color_sha256=hashlib.sha256(color.tobytes()).hexdigest(),
                    depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest()))
    finally: renderer.close()
    report = dict(scope=__doc__, nearest_texture=args.nearest, multisampling_disabled=args.no_msaa, frames=rows,
        unique_colors={v:len({r['color_sha256'] for r in rows if r['variant']==v}) for v in range(3)},
        unique_depths={v:len({r['depth_sha256'] for r in rows if r['variant']==v}) for v in range(3)})
    save_result(CACHE/'diagnostics'/args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2),flush=True)


if __name__ == '__main__': main()
