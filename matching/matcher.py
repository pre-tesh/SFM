import cv2
import numpy as np
import config
LOWE_RATIO_THRESH = 0.75

class FeatureMatcher:

    def _compute_distances(self , des1, des2):
        """We compute the eucliedean distance of descriptor 1 and descriptr 2 from img 1
          and img 2 resp we can choose any type of distance metric but we will use euclidean distance(l2) for now for simplicity so its like
          calculating distance between points in a 128 dimensional space since the descriptors are 
          of shape (n,128)"""
        """so arguments are 1 n*128 array des1 and m,128 for des2 adn we return distances
         which will be a numpy array of shape n*m n being the no of descriptors 
         in des 1 and m being no of descroptors in des2 """

        #calculate the euclidean distance(l2) between each descriptor in des1 and des2 
        # here we sqaush the 128 numbers into one valuable number , the sqaured distance from origin .
        sq1 = np.sum(des1 ** 2, axis=1, keepdims=True)   # (N, 1)
        sq2 = np.sum(des2 ** 2, axis=1, keepdims=True)   # (M, 1)
        # then we multiply each descriptor of des1 with each descriptor of des2 .
        dot = des1 @ des2.T                               # (N, M)
        # (a-b)^2 = a^2 + b^2 - 2ab
        distance = np.sqrt(np.maximum(sq1 + sq2.T - 2 * dot, 0))  # (N, M)
        # now distance is a nxm array telling teh euclidean distance between all features of des1 and des2
        return distance


    def match_features ( self , kp1 , des1 , kp2 , des2):
        """
        Args:
            kp1, kp2 : list of KeyPoint (image 1 & 2)
            des1, des2 : np.ndarray of shape (N, 128) and (M, 128)
        Returns:
            good_matches : list of (i, j) tuples
                i  index of keypoint in kp1
                j  index of matching keypoint in kp2
            pts1 : np.ndarray of shape (K, 2)  – (x, y) in image 1
            pts2 : np.ndarray of shape (K, 2)  – (x, y) in image 2
        """
        # Ensure we are working with float values
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)

        #then we compute a distance matrix of shape (n,m) where n is no of descriptors in des1 and m is no of descriptors in des2
        distances = self._compute_distances(des1, des2)
        good_matches = []
        pts1 = []
        pts2 = []

        for i in range(len(des1)):
            dists = distances[i]            # distances from descriptor i to every descriptor in img2
        
            sorted_index = np.argsort(dists) 

            best_index= sorted_index[0]     # take the 2 least distant point's indices
            second_best_index = sorted_index[1]

            best_dist = dists[best_index]           # take the 2 least distant point's distances
            second_best_dist = dists[second_best_index]

            # Apply Lowe's ratio test
            if best_dist < config.LOWE_RATIO_THRESH * second_best_dist:
            # keep the match
                good_matches.append((i, best_index))
            # store the pixel coordinates (x, y) of both keypoints
                pts1.append(kp1[i].pt)          # (x, y) from image 1
                pts2.append(kp2[best_index].pt)   # (x, y) from image 2
        # Convert coordinate lists to NumPy arrays (float32 is OpenCV‑friendly)
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)


        print(f"[Matcher] {len(kp1)} descriptors in img1, {len(kp2)} in img2")
        print(f"[Matcher] {len(good_matches)} good matches after Lowe's ratio test")
        return good_matches, pts1, pts2
        
        