"""ORB detector is an alternative to SIFT for detecting keypoints , we use the opencv implementation for this as well ."""
import cv2 as cv
from features.base import FeatureExtractor

class ORBExtractor(FeatureExtractor):
    def __init__(self,n_features=5000):
        self.orb = cv.ORB_create(nfeatures=n_features)

    def extract(self,image):
        """Detects ORB keypoints and computes their descriptors for the given image."""

        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray,None)
        return keypoints, descriptors    