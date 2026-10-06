import numpy as np
import config

def linear_pnp(points_3d , points_2d):
    """
        here we estimate the cameera pose by 
        linear transfromation .
        i.e make matrix a and solve with svd 

    inputs : 1. points)_3d = (n,3) array of known 3d points
             2. points_2d = (n,2) array fo their 2d projectison in the new image

    output :
        R = 3x3 rotation matrxi 
        and t a translation vector .
        """
    K = config.K
    K_inv = np.linalg.inv(K)

    N = len(points_3d)

    if N < 6:
        raise ValueError("linear_pnp requires at least 6 point correspondences")

    # now we normalize the 2d pointns : remove the camera intrinsics 
    homo_pts = np.column_stack([points_2d, np.ones(N)])
    norm_pts = (K_inv @ homo_pts.T).T

    # Build the linear system (2 equations per point)
    A = np.zeros((2 * N, 12))
    for i in range(N):
        X = np.append(points_3d[i], 1.0)
        u = norm_pts[i, 0]
        v = norm_pts[i, 1]

        A[2 * i] = np.concatenate([X, np.zeros(4), -u * X])
        A[2 * i + 1] = np.concatenate([np.zeros(4), X, -v * X])

    # Solve with SVD
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # Extract R and t
    R = P[:, :3]
    t = P[:, 3]

    # Project to the closest proper rotation matrix
    U, _, Vt2 = np.linalg.svd(R)
    R = U @ Vt2
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    return R, t