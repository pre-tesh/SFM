from abc import ABC, abstractmethod


"""
    Abstract base class for all feature extractors.
    Every extractor (SIFT, ORB, etc.) must implement extract().
"""
class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self,image):
        pass