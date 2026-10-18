import cv2
import random
import numpy as np
from src.brushes.base_brush import BaseBrush


class WatercolorBrush(BaseBrush):
    """Watercolor brush with jittered thickness and localised blending.

    Uses region-of-interest blending to avoid copying the full canvas.
    """

    def __init__(self, color=(255, 0, 255), thickness=8, opacity=0.4):
        super().__init__(color=color, thickness=thickness, opacity=opacity)

    def stroke(self, canvas, p1, p2):
        jitter_thickness = max(1, self.thickness + random.randint(-2, 4))
        cv2.line(canvas, p1, p2, self.color, jitter_thickness, lineType=cv2.LINE_AA)