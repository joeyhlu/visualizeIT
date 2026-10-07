"""Fetch two versioned official model files; verify pinned bytes on every reuse."""
import hashlib
import urllib.request
from .bop_assets import cache_write
from .ycb_assets import ROOT

MODELS={
    'magic_touch-v1.tflite':('https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite','e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25'),
    'hand_landmarker-v1.task':('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task','fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1')
}

def acquire():
    root=ROOT/'.cache/show3d';root.mkdir(parents=True,exist_ok=True)
    for name,(url,digest) in MODELS.items():
        path=root/'models'/name
        if path.exists():data=path.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=30) as response:data=response.read(12*1024*1024+1)
            if len(data)>12*1024*1024:raise ValueError('Model download exceeds bound.')
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Model hash mismatch: '+name)
        if not path.exists():cache_write(root,path,data)
        print(name,'verified',len(data),flush=True)

if __name__=='__main__':acquire()
