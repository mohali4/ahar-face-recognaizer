from .base import (
    baseRecognizer, 
    baseRecognizerPool,
    baseDetector
)
import dlib
class dlibFaceDetector(baseDetector):
    def __init__(self):
        self._detector = dlib.get_frontal_face_detector()
    def detect(self, buffer):
        return self._detector(buffer)
