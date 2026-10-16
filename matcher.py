import cv2
import numpy as np


class FeatureMatcher:
    def __init__(self, ratio_thresh=0.82):
        self.ratio_thresh = ratio_thresh
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.keypoints = None
        self.descriptors = None
        self.match_cache = {}

    def set_features(self, keypoints, descriptors):
        """Set per-image features and clear cached matches for the old ordering."""
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.match_cache.clear()

    def get_matches(self, i, j):
        """Return cached matches and point coordinates for an image pair."""
        if self.keypoints is None or self.descriptors is None:
            raise ValueError("Set image features before requesting matches")

        pair = (min(i, j), max(i, j))
        if pair not in self.match_cache:
            self.match_cache[pair] = self.match_features(
                self.keypoints[pair[0]], self.descriptors[pair[0]],
                self.keypoints[pair[1]], self.descriptors[pair[1]],
            )

        matches, pts1, pts2 = self.match_cache[pair]
        if i > j:
            matches = [(idx2, idx1) for idx1, idx2 in matches]
            return matches, pts2, pts1
        return matches, pts1, pts2

    def match_features(self, kp1, des1, kp2, des2):

        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)
        candidates = self.flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for pair in candidates:
            if len(pair) == 2:
                first, second = pair
                if first.distance < self.ratio_thresh * second.distance:
                    good_matches.append((first.queryIdx, first.trainIdx))

        if good_matches:
            pts1 = np.float32([kp1[i].pt for i, _ in good_matches])
            pts2 = np.float32([kp2[j].pt for _, j in good_matches])
        else:
            pts1 = np.empty((0, 2), dtype=np.float32)
            pts2 = np.empty((0, 2), dtype=np.float32)
        return good_matches, pts1, pts2
