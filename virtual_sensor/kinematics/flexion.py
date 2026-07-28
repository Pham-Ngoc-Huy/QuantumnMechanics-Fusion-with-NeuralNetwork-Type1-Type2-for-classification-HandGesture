from abc import ABC, abstractmethod
import numpy as np
from log import logger
# Compute MCP/PIP/DIP flexion angles
class FlexionEstimation(ABC):
    """
    @brief: calculating flexion angles for hand gestures relies on tracking
        joint coordinates and measuring the bend of the fingers.

        Equation:
            - Dot product formula: Compute the angle degree using the dot product of the two vectors:
                degree = arccos(A . B / |A||B|)
            - State mapping: Map the resulting degree value (0 to 90+ degrees) to gesture states like open, neutral, or closed flexion

    @param landmark detection
    @param vector formation
    @param angle calculation

    @book{
        szeliski2022computer,
        title={Computer Vision: Algorithms and Applications},
        author={Szeliski, Richard},
        edition={2},
        year={2022},
        publisher={Springer}
    }
    """
    @abstractmethod
    def calculate(self, landmarks: np.ndarray):
        pass

class FlexionAngle(FlexionEstimation):
    """
    Compute MCP/PIP/DIP flexion using the cosine rule.

    This class does not implement angle mathematics directly.
    It delegates angle computation to AngleBetweenCosine,
    keeping geometry and kinematics separated.
    """

    def __init__(self, angle_calculator):
        self.angle = angle_calculator

    def calculate(
        self,
        landmarks: np.ndarray,
        joint
    ):
        """
        @brief: angle calculated (degree)
        @param landmarks: ndarray (21,3)
        @param joint: tuple(int, int, int)
            example: (5,6,7)
            joint = 6
        """
        p1 = landmarks[joint[0]]
        p2 = landmarks[joint[1]]
        p3 = landmarks[joint[2]]

        v1 = p1 - p2
        v2 = p3 - p2
        logger.info(f"Calculate flexion-angle: \n"
                    f"v1: {v1} \n"
                    f"v2: {v2} \n"
                    f"result: {self.angle.calculate(v1,v2)}")
        return self.angle.calculate(v1,v2)