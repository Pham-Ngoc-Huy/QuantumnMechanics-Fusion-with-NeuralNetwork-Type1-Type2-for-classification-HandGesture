from abc import ABC, abstractmethod
from log import logger
import numpy as np

class ProjectToPlane(ABC):
    @abstractmethod
    def project(
        self,
        vector: np.ndarray,
        plane_normal: np.ndarray
    ):
        pass

class ProjectToPalmPlane(ProjectToPlane):
    def project(
        self,
        vector: np.ndarray,
        plane_normal: np.ndarray
    ):
        """
        @brief: Project a 3D vector onto the palm plane.
        @params vector : ndarray (3,). Input vector.
        @params plane_normal : ndarray (3,). Unit normal vector of the palm plane.
        Returns
        -------
        ndarray (3,)
            Projected vector lying on the palm plane.
        """

        # Remove the component along the plane normal.
        projected = vector - np.dot(vector, plane_normal) * plane_normal
        projected = projected / np.linalg.norm(projected)

        logger.info(f"Calculated Projector: \n"
                    f"Projected: {projected}")
        return projected