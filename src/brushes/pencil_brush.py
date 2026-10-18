import cv2
import numpy as np
from src.brushes.base_brush import BaseBrush


class PencilBrush(BaseBrush):
    """Semi-transparent pencil brush using localised blending.

    Draws a thin, partially opaque line by blending only the affected
    region rather than copying the entire canvas.
    """

    def __init__(self, color=(255, 0, 255), thickness=8, opacity=0.6):
        super().__init__(color=color, thickness=thickness, opacity=opacity)

    def stroke(self, canvas, p1, p2):
        thin_thickness = max(1, self.thickness // 3)
        cv2.line(canvas, p1, p2, self.color, thin_thickness, lineType=cv2.LINE_AA)