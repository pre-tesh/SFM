import numpy as np 
from geometry.fundamental import eat_pointalgo
import config

def ransac_F(pts1,pts2):
    """ Working of Ransac in this case:
        1.Pick 8 random matches 
        2.Calculate F from these 8 points
        3.Check ALL matches against this F
        4.Count inliers and update F if best
        5.Recompute F using all inliers.
    """
    N = len(pts1)
    best_f = None
    best_inlier_count = 0
    best_inliers = np.zeros(N, dtype=bool)

    #convert to homogeneous coordinates because the fundamental matrix is a 3x3 matrix .
    pts1_h = np.column_stack([pts1,np.ones(N)])
    pts2_h = np.column_stack([pts2,np.ones(N)])

    for i in range(1000): #we run ransac for 1000 iterations

        # 1. Randomly pick 8 indices
        indices = np.random.choice(N, 8, replace=False)
        pts1_samples = pts1[indices]
        pts2_samples = pts2[indices]

        #2. Caclulate F from these 8 points
        F = eat_pointalgo(pts1_samples, pts2_samples)

        #3 for the calculated F we check all matches against this F and count the inliers
        """ the way this error works is 
            for a correct fundamental matrix F , pt1.T * F * pt2 = 0
            so we can calculate the error by just summing the absolute value of pt1.T * F * pt2 for all points 
            and if this error is less than the threshold (0.1) we can say that this point is an inlier.
        """
        error = np.abs(np.sum(pts2_h * (F @ pts1_h.T).T, axis=1))
        is_inlier = error < 0.1

        #4. Count inliers and update F if best
        inlier_count = np.sum(is_inlier)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_f = F
            best_inliers = is_inlier

    #5. Recompute F using all inliers
    pts1_inliers = pts1[best_inliers]
    pts2_inliers = pts2[best_inliers]
    best_f = eat_pointalgo(pts1_inliers, pts2_inliers)
    return best_f, best_inliers