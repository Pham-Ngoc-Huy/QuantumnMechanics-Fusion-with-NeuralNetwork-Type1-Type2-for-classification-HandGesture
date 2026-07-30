import cv2
import mediapipe as mp

class Webcam:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam.")

    def read(self):
        success, frame = self.cap.read()
        if not success:
            return None
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        return frame, mp_image

    def show(self, frame):
        
        cv2.imshow(
            "Virtual Sensor Glove",
            frame
        )

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()