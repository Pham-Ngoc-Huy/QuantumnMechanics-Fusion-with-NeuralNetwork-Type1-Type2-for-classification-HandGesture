import cv2
import numpy as np

from log import logger
from virtual_sensor.classifier import GestureClassifier
from virtual_sensor.generator import GenerateVirtualSensor
from virtual_sensor.geometry import (
    AngleBetweenCosine,
    EstimatePalmPlaneCross,
    ProjectToPalmPlane,
)
from virtual_sensor.vectorize import MatlabFeatureVector


class LandmarkPrinter:
    def __call__(self, result, output_image, timestamp):
        if len(result.hand_landmarks) == 0:
            return
        logger.info("=" * 80)
        logger.info(f"Timestamp: {timestamp}")
        hand = result.hand_landmarks[0]
        for idx, lm in enumerate(hand):
            logger.info(f"{idx:02d} x={lm.x:.4f} y={lm.y:.4f} z={lm.z:.4f}")


class LandmarkVisualizer:
    def __init__(self, config):
        self.result = None
        self.config = config
        self.sensors_arrangement = self.config.sensors
        self.connections = self.config.connections
        self.landmarks = None
        self.projector = ProjectToPalmPlane()
        self.angle = AngleBetweenCosine()
        # projector
        self.plane_estimator = EstimatePalmPlaneCross()
        self.generator = GenerateVirtualSensor()
        self.vectorize = MatlabFeatureVector()
        self.classifier = GestureClassifier(model_path=self.config.model_gesture)
        self.dimension_activities = self.config.dimension_activities

    def __call__(self, result, output_image, timestamp):
        self.result = result
        if not result.hand_landmarks:
            self.landmarks = None
            return

        hand = result.hand_landmarks[0]
        self.landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

        for idx, lm in enumerate(result.hand_landmarks[0]):
            logger.info(f"{idx:02d} x={lm.x:.4f} y={lm.y:.4f} z={lm.z:.4f}")
        logger.info("Callback")

    def draw(self, frame):
        if self.landmarks is None:
            return
        center, normal = self.plane_estimator.estimate(self.landmarks)

        sensors = self.generator.generate(landmarks=self.landmarks, palm_normal=normal)

        feature_vector = self.vectorize.convert(config=self.sensors_arrangement, sensors=sensors)

        gesture_class, confidence = self.classifier.predict(feature_vector)
        activity = self.dimension_activities[gesture_class]
        for hand in self.result.hand_landmarks:
            self._draw_hand(frame=frame, hand_landmarks=hand)
        self._draw_project_to_palm_plane(frame=frame, palm_normal=normal)
        self._draw_angle_calculation(frame=frame, palm_normal=normal)
        self._draw_palm_plane(frame=frame, palm_center=center, palm_normal=normal)
        self._draw_virtual_sensor(frame=frame, sensors=sensors)
        self._draw_feature_vector(frame=frame, feature_vector=feature_vector)

        self._draw_prediction(
            frame=frame,
            gesture_class=gesture_class,
            activity=activity,
            confidence=confidence,
        )

    def _draw_prediction(self, frame, gesture_class, activity, confidence):
        cv2.putText(
            frame,
            f"Gesture : {gesture_class}",
            (20, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Activity : {activity}",
            (20, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence : {confidence:.2%}",
            (20, 490),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def _draw_feature_vector(self, frame, feature_vector):
        x = frame.shape[1] - 250
        y = 30
        cv2.putText(
            frame,
            "MATLAB-feature-vectors",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        y += 30

        for i, value in enumerate(feature_vector):
            cv2.putText(
                frame,
                f"[{i:02d}] {value:6.1f}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            y += 20

            if y > frame.shape[0] - 20:
                break

    def _draw_virtual_sensor(self, frame, sensors):
        x = 20
        y = 30
        dy = 25

        cv2.putText(
            frame,
            "Virtual-Sensor",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        y += 30
        for key, value in sensors.items():
            cv2.putText(
                frame,
                f"{key:<10}: {value:6.1f}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += dy

    def _draw_hand(self, frame, hand_landmarks):
        h, w = frame.shape[:2]

        points = []

        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)

            points.append((x, y))

            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        for start, end in self.connections:
            cv2.line(frame, points[start], points[end], (255, 0, 0), 2)

    def _draw_angle_calculation(self, frame, palm_normal: np.ndarray):
        """
        @brief Visualize angle calculation between two projected finger vectors.

        Red   : Index finger vector
        Green : Middle finger vector
        White : Included angle
        """
        h, w = frame.shape[:2]
        index_start = self.landmarks[5]
        index_end = self.landmarks[6]

        middle_start = self.landmarks[9]
        middle_end = self.landmarks[10]

        index_vector = index_end - index_start
        middle_vector = middle_end - middle_start

        index_proj = self.projector.project(vector=index_vector, plane_normal=palm_normal)

        middle_proj = self.projector.project(vector=middle_vector, plane_normal=palm_normal)

        angle = self.angle.calculate(index_proj, middle_proj)

        scale = 0.20

        index_start_px = (int(index_start[0] * w), int(index_start[1] * h))

        middle_start_px = (int(middle_start[0] * w), int(middle_start[1] * h))

        index_end_px = (
            int((index_start[0] + index_proj[0] * scale) * w),
            int((index_start[1] + index_proj[1] * scale) * h),
        )

        middle_end_px = (
            int((middle_start[0] + middle_proj[0] * scale) * w),
            int((middle_start[1] + middle_proj[1] * scale) * h),
        )

        cv2.arrowedLine(frame, index_start_px, index_end_px, (0, 0, 255), 2, tipLength=0.2)

        cv2.arrowedLine(frame, middle_start_px, middle_end_px, (0, 255, 0), 2, tipLength=0.2)

        text_pos = (
            int((index_start_px[0] + middle_start_px[0]) / 2),
            int((index_start_px[1] + middle_start_px[1]) / 2) - 20,
        )

        cv2.putText(
            frame,
            f"{angle:.1f} deg",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def _draw_project_to_palm_plane(self, frame, palm_normal: np.ndarray):
        """
        @brief Visualize vector projection onto the palm plane.
            Red: Original finger vector
            Green: Projected vector on palm plane
        """

        h, w = frame.shape[:2]
        # Estimate palm plane
        start = self.landmarks[5]
        end = self.landmarks[6]

        finger_vector = end - start

        projected = self.projector.project(vector=finger_vector, plane_normal=palm_normal)

        start_px = (int(start[0] * w), int(start[1] * h))

        scale = 0.25

        end_original = start + finger_vector * scale
        end_projected = start + projected * scale

        end_original_px = (int(end_original[0] * w), int(end_original[1] * h))

        end_projected_px = (int(end_projected[0] * w), int(end_projected[1] * h))

        cv2.arrowedLine(frame, start_px, end_original_px, (0, 0, 255), 2, tipLength=0.2)

        cv2.arrowedLine(frame, start_px, end_projected_px, (0, 255, 0), 2, tipLength=0.2)

        cv2.putText(
            frame,
            "Projection",
            (start_px[0] + 10, start_px[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    def _draw_palm_plane(self, frame, palm_center: np.ndarray, palm_normal: np.ndarray):
        """
        @brief Draw palm center and palm normal.
            Red Dot: Palm center
            Yellow Line: Palm normal (projected to image plane)
        """

        h, w = frame.shape[:2]

        # Convert normalized coordinates to image pixels
        center_px = (int(palm_center[0] * w), int(palm_center[1] * h))
        scale = 0.15  # length in normalized coordinate: đủ để thấy được vector
        end = palm_center + palm_normal * scale
        end_px = (int(end[0] * w), int(end[1] * h))
        # Palm center
        cv2.circle(
            frame,
            center_px,
            6,
            (0, 0, 255),  # Red
            -1,
        )
        # Palm normal
        cv2.arrowedLine(
            frame,
            center_px,
            end_px,
            (0, 255, 255),  # Yellow
            2,
            tipLength=0.2,
        )
        # Debug text
        cv2.putText(
            frame,
            "Palm-Plane",
            (center_px[0] + 10, center_px[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
