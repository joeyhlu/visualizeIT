"""Scoped diagnostic matching upstream OpenCV default initial-guess behavior."""
from contextlib import contextmanager
from .vision import cv2


@contextmanager
def without_extrinsic_guess():
    original=cv2.solvePnPRansac
    def solve(*args,**kwargs):
        kwargs['useExtrinsicGuess']=False
        return original(*args,**kwargs)
    cv2.solvePnPRansac=solve
    try:yield
    finally:cv2.solvePnPRansac=original
