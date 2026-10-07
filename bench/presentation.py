"""Static scientific comparison panel made from the actual benchmark frames."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
from .storage import save_image


def video_panel(output):
    report=json.loads((output/'report.json').read_text())
    objects=report['objects']
    image=Image.new('RGB',(1470,145+len(objects)*435),'#edf2ef')
    draw=ImageDraw.Draw(image)
    font_path=Path('C:/Windows/Fonts/segoeui.ttf')
    def font(size): return ImageFont.truetype(str(font_path),size) if font_path.exists() else ImageFont.load_default(size=size)
    draw.text((20,12),'VisualizeIt: real recorded footage',font=font(30),fill='#183f37')
    draw.text((20,57),'Initialized tracking from 10 annotated onboarding frames; 110 held-out frames per object. Not phone AR.',font=font(20),fill='#35574f')
    for column,label in enumerate(('Original capture','Supplied-pose rendering control','Our estimated-pose overlay')):
        draw.text((20+column*485,108),label,font=font(21),fill='#183f37')
    for row,obj in enumerate(objects):
        q=obj['score']; y=145+row*435
        caption=f"{obj['name']} | median {q['medianErrorPixels']:.2f} px / P95 {q['p95ErrorPixels']:.2f} px (720p equivalent) | availability {q['eligibleVisibleFrameAvailability']*100:.1f}%"
        draw.text((20,y),caption,font=font(20),fill='#183f37')
        sample=obj['thumbnails'][0]
        for column,key in enumerate(('original','control','tracked')):
            frame=Image.open(output/sample[key]).convert('RGB').resize((470,353))
            image.paste(frame,(20+column*485,y+35))
        if q['provisionalTrackingGatePassed']: status='Qualifying motion clip; provisional tracking gate passed.'
        elif q['eligibleVisibleFrameAvailability']>=.9: status='Low-motion control: numeric attachment target met, motion selection gate failed.'
        else: status='Motion selection and tracking availability failed; inspect lost states in the report.'
        draw.text((20,y+393),status,font=font(18),fill='#735038')
    save_image(image,output/'results.jpg',quality=90)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts'/'video-tracking')
    args=parser.parse_args(); video_panel(args.output)
