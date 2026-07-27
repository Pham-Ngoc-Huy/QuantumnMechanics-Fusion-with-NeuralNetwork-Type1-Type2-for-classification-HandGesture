from abc import ABC, abstractmethod
from log import logger
import numpy as np

class AngleBetween(ABC):
    @abstractmethod
    def calculate(self):
        pass

class AngleBetweenCosine(AngleBetween):
    def calculate(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray
    ) -> float:

        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            raise ValueError("Cannot compute angle from zero-length vector.")

        # Normalize vectors
        u1 = vector1 / norm1
        u2 = vector2 / norm2

        # Dot product
        cosine = np.dot(u1, u2)

        # Numerical stability
        cosine = np.clip(cosine, -1.0, 1.0)

        # Angle in radians
        angle_rad = np.arccos(cosine)

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)

        logger.info(
            "Angle Between Vectors\n"
            f"Vector-1: {vector1}\n"
            f"Vector-2: {vector2}\n"
            f"Cos(theta): {cosine:.4f}\n"
            f"Angle: {angle_deg:.2f} deg"
        )

        return angle_deg