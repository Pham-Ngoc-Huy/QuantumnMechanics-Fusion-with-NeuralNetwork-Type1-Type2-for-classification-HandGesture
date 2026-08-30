from abc import ABC, abstractmethod


class BaseDetector(ABC):
    @abstractmethod
    def create(self):
        """Create detector instance."""
        pass

    @abstractmethod
    def detect(self, image, timestamp):
        """Run detection."""
        pass

    @abstractmethod
    def close(self):
        """Release detector."""
        pass
