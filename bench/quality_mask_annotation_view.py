"""Source-only coordinate views for agent-reviewed diagnostic mask annotations."""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from .quality_assets import ROOT, CACHE


def main():
    for fid,bounds in [(1149,(60,810,540,830)),(1314,(410,740,420,1010)),(1388,(420,730,390,1050))]:
        rgb=Image.open(ROOT/f'artifacts/model-quality/annotations/keyboard/{fid}.jpg')
        fig,ax=plt.subplots(figsize=(9,12 if fid!=1149 else 5),layout='constrained')
        ax.imshow(rgb);ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[3],bounds[2])
        ax.set_xticks(range((bounds[0]//20)*20,bounds[1],20));ax.set_yticks(range((bounds[2]//20)*20,bounds[3],20))
        ax.grid(alpha=.45,color='cyan');ax.tick_params(labelsize=9)
        ax.set_title(f'Original image {fid}: native pixel coordinates; no prediction or reference pose')
        fig.savefig(CACHE/f'diagnostics/source-mask-grid-{fid}.jpg',dpi=150);plt.close(fig)


if __name__=='__main__':main()
