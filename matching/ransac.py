import numpy as np 
from geometry.fundamental import eat_pointalgo
import config

def ransac_F(pts1,pts2):
    """ now here we use RANSAC to find F and remove out outlier matches """
    N = len(pts1)
    best_f = None
    best_inlier_count = 0
    best_inlier_mask = np.zeros(N, dtype=bool)

    #convert to homogeneous coordinates
    pts1_h = np.column_stack([pts1,np.ones(N)])
    pts2_h = np.column_stack([pts2,np.ones(N)])

    for i in range(config.RANSAC_ITERATIONS):

        # 1. Randomly pick 8 mathces 
        ind = np.random.choice(N, 8, replace=False)
        pts1_sample = pts1[ind]
        pts2_sample = pts2[ind]

        #2. Caclulate F from these 8 points
        F = eat_pointalgo(pts1_sample, pts2_sample)

        #3 Check ALL matches against this F 
        error = np.abs(np.sum(pts2_h * (F @ pts1_h.T).T, axis=1))
        inlier_mask = error < config.RANSAC_THRESHOLD

        #4. Count inliers and update F if best
        inlier_count = np.sum(inlier_mask)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_f = F
            best_inlier_mask = inlier_mask

    #5. Recompute F using all inliers
    pts1_inliers = pts1[best_inlier_mask]
    pts2_inliers = pts2[best_inlier_mask]
    best_f = eat_pointalgo(pts1_inliers, pts2_inliers)
    return best_f, best_inlier_mask