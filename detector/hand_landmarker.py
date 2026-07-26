import mediapipe as mp
from mediapipe.tasks.python import vision
from detector.base_detector import BaseDetector
class HandLandmarkerDetector(BaseDetector):
    def __init__(
        self,
        model_path:str,
        callback,
        running_mode=vision.RunningMode.LIVE_STREAM
    ):
        self.model_path=model_path
        self.callback=callback
        self.running_mode=running_mode
        self.detector=None
    def create(self):
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=self.running_mode,
            result_callback=self.callback
        )
        self.detector=vision.HandLandmarker.create_from_options(
            options
        )
        return self
    def detect(self, image, timestamp):
        self.detector.detect_async(
            image,
            timestamp
        )

    def close(self):
        if self.detector is not None:
            self.detector.close()

    def __enter__(self):
        return self.create()
    
    def __exit__(self, exc_type, exc, tb):
        self.close()