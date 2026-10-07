"""Model-versus-image appearance audit; no pose labels or attachment success claim."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from .quality_assets import ROOT,CACHE,save_result


def main():
    root=CACHE/'diagnostics/keyboard-annotation-model';metrics={}
    for name in ('keys','underside'):
        rgb=np.asarray(Image.open(root/f'{name}.png'))
        with np.load(root/f'{name}.npz',allow_pickle=False) as data:mask=data['depth_m']>0
        luma=rgb[mask,:3]@np.array([.2126,.7152,.0722])
        metrics[name]=dict(rendered_foreground_pixels=int(mask.sum()),luminance_mean_255=float(luma.mean()),
            luminance_std_255=float(luma.std()),fraction_black_under_2=float((luma<2).mean()))
    fig=plt.figure(figsize=(10,5.8),layout='constrained');grid=fig.add_gridspec(2,2,width_ratios=[2.4,1.])
    for i,name,title in [(0,'keys','Fixed model: key-side texture'),(1,'underside','Fixed model: underside texture is essentially blank')]:
        ax=fig.add_subplot(grid[i,0]);ax.imshow(Image.open(root/f'{name}.png'));ax.set_title(title,fontsize=11);ax.axis('off')
    ax=fig.add_subplot(grid[:,1]);ax.imshow(Image.open(ROOT/'artifacts/model-quality/annotations/keyboard/1314.jpg'))
    ax.set_xlim(410,740);ax.set_ylim(1010,420);ax.set_title('Actual underside\nfeet and printed markings',fontsize=11);ax.axis('off')
    fig.suptitle('Appearance coverage gap: fixed model and source frame 1314',fontsize=13)
    fig.savefig(CACHE/'diagnostics/keyboard-model-appearance.jpg',dpi=120,pil_kwargs={'quality':82});plt.close(fig)
    save_result(CACHE/'diagnostics/keyboard-model-appearance.json',dict(scope=__doc__,metrics=metrics,
        canonical_views_use_video_poses=False,normal_texture_atlas_visually_uniform=True,
        conclusion='The model-only unlit underside lacks recognizable feet and printed markings present in the source image.',
        causal_tracking_effect_verified=False,independent_accuracy_verified=False))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
