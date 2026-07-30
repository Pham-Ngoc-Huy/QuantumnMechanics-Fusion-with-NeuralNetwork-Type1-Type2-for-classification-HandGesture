from abc import ABC, abstractmethod
import numpy as np

class SensorToVector(ABC):
    """
    @brief: Convert virtual sensor into a feature vector
    """
    @abstractmethod
    def convert(self, config, sensors:dict):
        pass

class MatlabFeatureVector(SensorToVector):
    def convert(self, config, sensors:dict):
        sensors_arrangement = [key for key, _ in config.items() if key != "mcp2_a"]

        vector=[]
        for sensor in sensors_arrangement:
            vector.append(sensors[sensor])

        return np.array(
            vector,
            dtype=np.float32
        )