import numpy as np
import config

def triangulate_point(P1,P2,pt1,pt2):
    """ Triangulate a single point from two views.
    P1 and P2 are the projection matrices for the two views.
    pt1 and pt2 are the corresponding points in the two images.
    """
    A = np.zeros((4, 4))
    A[0] = pt1[0] * P1[2] - P1[0]
    A[1] = pt1[1] * P1[2] - P1[1]
    A[2] = pt2[0] * P2[2] - P2[0]
    A[3] = pt2[1] * P2[2] - P2[1]

    # Solve for the 3D point using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[-1]  # Convert to non-homogeneous coordinates

    return X[:3]

def triangulate_points(P1, P2, pts1, pts2):
    """ Triangulate multiple points from two views.
    P1 and P2 are the projection matrices for the two views.
    pts1 and pts2 are the corresponding points in the two images.
    """
    points_3d = []
    for pt1, pt2 in zip(pts1, pts2):
        X = triangulate_point(P1, P2, pt1, pt2)
        points_3d.append(X)
    return np.array(points_3d)