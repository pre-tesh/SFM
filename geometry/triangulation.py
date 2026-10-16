import numpy as np
import cv2 as cv
import config

def triangulate_point(P1, P2, pt1, pt2):
    """ Triangulate a single point from two views. """
    A = np.zeros((4, 4))
    A[0] = pt1[0] * P1[2] - P1[0]
    A[1] = pt1[1] * P1[2] - P1[1]
    A[2] = pt2[0] * P2[2] - P2[0]
    A[3] = pt2[1] * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[-1]) < 1e-7:
        return None
    X /= X[-1]
    return X[:3]

def triangulate_points(P1, P2, pts1, pts2):
    """ Vectorized triangulation of multiple points from two views using OpenCV. """
    if len(pts1) == 0:
        return np.zeros((0, 3))
    
    pts1_t = np.ascontiguousarray(pts1.T, dtype=np.float64)
    pts2_t = np.ascontiguousarray(pts2.T, dtype=np.float64)
    
    points_4d = cv.triangulatePoints(P1, P2, pts1_t, pts2_t)
    # Homogeneous coordinates may have either sign: X/w == (-X)/(-w).
    # Clamping negative w corrupts valid points and produces huge coordinates.
    w = points_4d[3]
    valid = np.isfinite(points_4d).all(axis=0) & (np.abs(w) > 1e-12)
    points_3d = np.full((len(pts1), 3), np.nan, dtype=np.float64)
    points_3d[valid] = (points_4d[:3, valid] / w[valid]).T
    return points_3d


def valid_triangulation_mask(points, pts1, pts2, K, R1, t1, R2, t2,
                             max_error=12.0, min_angle_degrees=1.0):
    """Require finite points, positive depth, reprojection fit and parallax."""
    camera1 = points @ R1.T + t1
    camera2 = points @ R2.T + t2
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels1 = camera1 @ K.T
        pixels2 = camera2 @ K.T
        error1 = np.linalg.norm(pixels1[:, :2] / pixels1[:, 2:3] - pts1, axis=1)
        error2 = np.linalg.norm(pixels2[:, :2] / pixels2[:, 2:3] - pts2, axis=1)
        ray1 = points - (-R1.T @ t1)
        ray2 = points - (-R2.T @ t2)
        cosine = np.sum(ray1 * ray2, axis=1) / (
            np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return (np.isfinite(points).all(axis=1)
            & (camera1[:, 2] > 1e-8) & (camera2[:, 2] > 1e-8)
            & (error1 < max_error) & (error2 < max_error)
            & (angle >= min_angle_degrees))
