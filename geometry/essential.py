import numpy as np 
import config 

def compute_essentialmatrix(F):
    """ compute essential matrix from fundamental matrix 
    
        F is fundametnal matrix. this works in pixel coordinates.
        E is essential matrix, this works in normalized coordinates.
        K is camera intrinsic matrix ,
       which converts pixel coordinates to normalized coordinates.

    """
    
    K = config.K
    E = K.T @ F @ K
    return E

def decompose_essentialmatrix(E):
    """ Decompose essential matrix into rotation and translation
    matrices. 
    This function returns four possible solutions for the camera pose.
    """
    U , s , vT = np.linalg.svd(E)

    S = np.diag([1, 1, 0])
    E = U @ S @ vT

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
# the two possible rotations
    R1 = U @ W @ vT
    R2 = U @ W.T @ vT


#make sure r1 and r2 are proper rotation matrices (det(R) = 1)
    if np.linalg.det(R1) < 0: R1 =-R1
    if np.linalg.det(R2) < 0: R2= -R2

    #translation is the third collumn
    t = U[:, 2]

    poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    return poses

def check_cheirality(R, t, pts1, pts2):
    """ Check the cheirality condition for a given rotation and translation.
    This function returns the number of points that are in front of both cameras.
    """
    from geometry.triangulation import triangulate_points
    K = config.K

    #Camera 1 is at origin, so its projection matrix is [I|0]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    #Camera 2 projection matrix is [R|t]
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    # Triangulate points
    points_3d = triangulate_points(P1, P2, pts1, pts2)

    # Check how many points are in front of both cameras
    in_front_of_camera1 = points_3d[:, 2] > 0
    points_3d_camera2 = (R @ points_3d.T + t.reshape(3, 1)).T
    in_front_of_camera2 = points_3d_camera2[:, 2] > 0

    return np.sum(in_front_of_camera1 & in_front_of_camera2)