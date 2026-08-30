from abc import ABC, abstractmethod

import numpy as np

from log import logger


class AbductionEstimation(ABC):
    """
    @brief: Estimate finger abduction/adduction angle
        Abduction is defined as the angular separation between
        two adjacent fingers after projecting both finger vectors onto the palm plane

    @references{
        author={Richard,Szeliski},
        title={Computer Vision: Algorithms and Applications},
        edition={2},
        year={2022},
        publisher={Springer}
    }
    @reference{
        author={David A. Winter},
        title={Biomechanics and Motor Control of Human Movement},
        edition={4},
        year={2009},
        publisher={Wiley}
    """

    @abstractmethod
    def calculate(
        self,
        landmarks: np.ndarray,
        palm_normal: np.ndarray,
        finger_1: tuple,
        finger_2: tuple,
    ):
        pass


class AbductionAngle(AbductionEstimation):
    """
    @brief: Compute the abduction angle between two adjacent fingers.
        procedures:
            finger_lanmarks -> finger_vectors -> project_on_palm_plane -> angle_between_vectors -> abduction_angle
    """

    def __init__(self, projector, angle_calculator):
        self.projector = projector
        self.angle = angle_calculator

    def calculate(
        self,
        landmarks: np.ndarray,
        palm_normal: np.ndarray,
        finger1: tuple,
        finger2: tuple,
    ):
        p1 = landmarks[finger1[0]]
        p2 = landmarks[finger1[1]]

        p3 = landmarks[finger2[0]]
        p4 = landmarks[finger2[1]]

        v1 = p2 - p1
        v2 = p4 - p3

        v1 = self.projector.project(vector=v1, plane_normal=palm_normal)

        v2 = self.projector.project(vector=v2, plane_normal=palm_normal)
        logger.info(f"Calculate abduction-angle: \nv1: {v1} \nv2: {v2} \nresult: {self.angle.calculate(v1, v2)}")
        return self.angle.calculate(v1, v2)
