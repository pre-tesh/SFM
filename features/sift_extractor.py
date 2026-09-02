"""in this file we borrow the works of smart developers and researchers who worked hard for us implementing SIFT IN opencv .
SIFT inherenetly is a complex multistage algorithm and could be another project in itself. so for now weuse th opencv implemetnation of SIFT ."""

import cv2 as cv
from features.base import FeatureExtractor

class SIFTExtractor(FeatureExtractor):
    def __init__(self,n_features=0):
        self.sift = cv.SIFT_create(nfeatures=n_features)

    def extract(self,image):
        """Detects SIFT keypoints and computes their descriptors for the given image."""

        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray,None)
        return keypoints, descriptors


