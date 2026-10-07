"""Scientific contact sheet for image-only foreground proposal inspection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from .quality_assets import CACHE
from .quality_runner import read_rgb
from .vision import cv2


def main():
    root=CACHE/'inputs/keyboard';manifest=json.loads((root/'input.json').read_text())
    rgb=read_rgb(root,manifest,1254)
    with np.load(CACHE/'diagnostics/cnos-1254-proposals.npz',allow_pickle=False) as z:masks=z['masks'].copy()
    prior=cv2.imread(str(CACHE/'results/keyboard/recovery-1239/segmentation/masks/1238.png'),cv2.IMREAD_GRAYSCALE)>0
    fig,axes=plt.subplots(2,3,figsize=(9,10),layout='constrained')
    selections=[('Current frame',None),('Wrong top identity: 19',masks[19]),('Narrow object proposal: 10',masks[10]),
        ('Large central proposal: 7',masks[7]),('Central proposal: 26',masks[26]),('Previous observed mask (1238)',prior)]
    for ax,(title,mask) in zip(axes.flat,selections):
        ax.imshow(rgb)
        if mask is not None:
            colors=np.zeros((*mask.shape,4));colors[mask]=[.05,1,.5,.6];ax.imshow(colors)
        ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle('Frame 1254: proposal diagnostics; no accuracy gate',fontsize=13)
    path=CACHE/'diagnostics/cnos-proposals-1254.png';fig.savefig(path,dpi=120);plt.close(fig);print(path)


if __name__=='__main__':main()
