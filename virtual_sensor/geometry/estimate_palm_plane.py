from log import logger
from abc import ABC, abstractmethod
import numpy as np

class EstimatePalmPlane(ABC):

    @abstractmethod
    def estimate(self, landmarks: np.ndarray):
        """
        Estimate palm plane from hand landmarks.

        Parameters
        ----------
        landmarks : ndarray (21,3)

        Returns
        -------
        palm_center : ndarray (3,)
        palm_normal : ndarray (3,)
        """
        pass


class EstimatePalmPlaneCross(EstimatePalmPlane):

    def estimate(self, landmarks: np.ndarray):
        """
            @brief: Estimate the palm plane using three anatomical landmarks:
                - Wrist (0)
                - Index MCP (5)
                - Little MCP (17)

                These three landmarks approximately define the palm plane.
                The plane normal is computed using the cross product of two
                vectors lying on the palm.

            @param landmarks : np.ndarray. Hand landmarks with shape (21, 3).

            Returns: (
                palm_center : np.ndarray. Center of the estimated palm plane.
                palm_normal : np.ndarray. Unit normal vector of the palm plane.
            )
        """
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        little_mcp = landmarks[17]

        # Two vectors on the palm
        v1 = index_mcp - wrist 
        v2 = little_mcp - wrist

        # Cross product -> tích có hướng 2 vector (dot product) -> tính tọa độ vector vuông góc
        palm_normal = np.cross(v1, v2)

        # normalize = sqrt(x^2 + y^2 + z^2) <Euclidean distance>
        norm = np.linalg.norm(palm_normal)

        if norm < 1e-8:
            raise ValueError("Palm plane cannot be estimated.")

        # normalizing
        palm_normal /= norm

        # estimate the palm center as the centroid of the three defining points
        palm_center = (wrist + index_mcp + little_mcp) / 3 
        logger.info(f"Estimated-PalmPlane by Cross: \n"
                    f"Palm-Center: {palm_center} \n"
                    f"Palm-Normal: {palm_normal}")
        return palm_center, palm_normal

# class EstimatePalmPlanePCA(EstimatePalmPlane):
# TODO:
# Replace the three-point estimation with a PCA-based plane fitting
# using multiple palm landmarks (0, 5, 9, 13, 17) to improve
# robustness against MediaPipe landmark noise.

