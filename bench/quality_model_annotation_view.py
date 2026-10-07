"""Canonical model-only views for semantic landmark annotation; no video poses loaded."""
import json
import numpy as np
from PIL import Image, ImageDraw
from .quality_assets import CACHE, digest, save_result
from .quality_gotrack import TexturedRenderer


def main():
    renderer=TexturedRenderer(CACHE/'inputs/keyboard/object.glb',28,unlit=True,disable_multisampling=True)
    from utils.structs import PinholePlaneCameraModel
    from utils.renderer_base import RenderType
    out=CACHE/'diagnostics/keyboard-annotation-model';out.mkdir(parents=True,exist_ok=True)
    width,height=1400,500;k=np.array([[1900.,0,700.],[0,1900.,250.],[0,0,1.]])
    try:
        for name,rotation in [('keys',[[1,0,0],[0,0,1],[0,-1,0]]),('underside',[[1,0,0],[0,0,-1],[0,1,0]])]:
            pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=[0,0,.7]
            camera_pose=np.linalg.inv(pose);camera_pose[:3,3]*=1000
            camera=PinholePlaneCameraModel(width=width,height=height,f=(1900,1900),c=(700,250),T_world_from_eye=camera_pose)
            rendered=renderer.render_object_model(28,camera)
            color=np.rint(rendered[RenderType.COLOR]*255).astype(np.uint8)
            depth=rendered[RenderType.DEPTH]*.001
            np.savez_compressed(out/f'{name}.npz',depth_m=depth,camera_from_object=pose,intrinsics=k)
            raw=Image.fromarray(color);raw.save(out/f'{name}.png')
            annotated=raw.copy();draw=ImageDraw.Draw(annotated)
            for x in range(0,width,100):draw.line([(x,0),(x,height)],fill=(40,160,200),width=1);draw.text((x+2,3),str(x),fill='yellow')
            for y in range(0,height,50):draw.line([(0,y),(width,y)],fill=(40,160,200),width=1);draw.text((2,y+2),str(y),fill='yellow')
            annotated.save(out/f'{name}-grid.png')
        save_result(out/'provenance.json',dict(scope=__doc__,asset_sha256=digest(CACHE/'inputs/keyboard/object.glb'),
            reference_poses_loaded=False,predicted_poses_loaded=False,units='metres; OpenCV camera coordinates'))
    finally:renderer.close()
    print(out)


if __name__=='__main__':main()
