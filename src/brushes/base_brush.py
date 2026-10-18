from abc import ABC, abstractmethod

class BaseBrush(ABC):
    def __init__(self, color=(255, 0, 255), thickness=8, opacity=1.0):
        self.color = color
        self.thickness = thickness
        self.opacity = float(opacity)

    @abstractmethod
    def stroke(self, canvas, p1, p2):
        pass