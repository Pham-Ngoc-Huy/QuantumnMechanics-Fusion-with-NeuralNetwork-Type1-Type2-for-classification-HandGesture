from abc import ABC, abstractmethod
import numpy as np
from log import logger
class WristEstimation(ABC):
    @abstractmethod
    def calculate(
        self,
        landmarks: np.ndarray,
        palm_normal: np.ndarray
    ):
        """
        wrist_f : float
            Wrist flexion angle (degree)

        wrist_a : float
            Wrist abduction angle (degree)
        """
        pass

class WristAngle(WristEstimation):
    def __init__(
        self,
        projector,
        angle_calculator
    ):
        self.projector = projector
        self.angle = angle_calculator

    def calculate(
        self,
        landmarks: np.ndarray,
        palm_normal: np.ndarray
    ):
        """
        Compute wrist orientation.

        Wrist direction is approximated by

            Wrist  ---> Palm Center

        The wrist vector is decomposed into

            - normal component
            - palm-plane component

        to estimate

            wrist_f
            wrist_a
        """

        wrist = landmarks[0]

        palm_center = (
            landmarks[5] +
            landmarks[9] +
            landmarks[13] +
            landmarks[17]
        ) / 4

        wrist_vector = palm_center - wrist

        # vector projected onto palm plane
        wrist_proj = self.projector.project(
            vector=wrist_vector,
            plane_normal=palm_normal
        )

        wrist_f = self.angle.calculate(
            wrist_vector,
            wrist_proj
        )

        reference = np.array([1.0, 0.0, 0.0])

        wrist_a = self.angle.calculate(
            wrist_proj,
            reference
        )
        logger.info(f"Wrist Estimated: \n"
                    f"wrist-f: {wrist_f} \n"
                    f"wrist-a: {wrist_a}")
        return wrist_f, wrist_a
