from config.config import Config
from camera.webcam import Webcam
from detector.callback import LandmarkVisualizer
from detector.hand_landmarker import HandLandmarkerDetector
from virtual_sensor.geometry.estimate_palm_plane import EstimatePalmPlaneCross
import cv2

def main():
    config=Config("config/config.yaml")

    callback=LandmarkVisualizer(config=config)

    with HandLandmarkerDetector(
            model_path=config.model_path,
            callback=callback
        ) as detector:

        with Webcam() as camera:

            timestamp = 0   
            while True:
                result = camera.read()

                if result is None:
                    break

                frame, mp_image = result

                detector.detect(
                    image=mp_image,
                    timestamp=timestamp
                )
                callback.draw(frame=frame)
                camera.show(frame=frame)
                # this 1000/ fps means
                # 1000 miliseconds = 1s -> 1/fps -> duraction of a single frame in milliseconds
                # for example: 60fps -> 1000/60 = 16.66 ms 
                timestamp += int(1000 / config.fps)
                if ord("q") == (cv2.waitKey(1) & 0xFF):
                    break

if __name__ == "__main__":
    main()