import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
import config


def rodrigues_to_matrix(rvec):
    """Convert 3-parameter Rodrigues vector to 3x3 rotation matrix."""
    return Rotation.from_rotvec(rvec).as_matrix()


def matrix_to_rodrigues(R):
    """Convert 3x3 rotation matrix to 3-parameter Rodrigues vector."""
    return Rotation.from_matrix(R).as_rotvec()


def project_point(X, R, t, K):
    """Project a 3D point onto a camera image plane."""
    x = K @ (R @ X + t)
    if x[2] == 0:
        return np.array([0.0, 0.0])
    return x[:2] / x[2]


def pack_params(camera_poses, points_3d, fixed_cam_idx=0):
    """Pack all camera poses and 3D points into one 1D vector."""
    params = []
    for cam_idx in sorted(camera_poses.keys()):
        if cam_idx == fixed_cam_idx:
            continue
        R, t = camera_poses[cam_idx]
        rvec = matrix_to_rodrigues(R)
        params.extend(rvec)
        params.extend(t)
    for pt in points_3d:
        params.extend(pt)
    return np.array(params, dtype=np.float64)


def unpack_params(params, n_cameras, n_points, camera_indices, fixed_cam_idx=0, fixed_R=None, fixed_t=None):
    """Unpack 1D vector back into camera poses and 3D points."""
    camera_poses = {}
    idx = 0
    for cam_idx in camera_indices:
        if cam_idx == fixed_cam_idx:
            camera_poses[cam_idx] = (fixed_R, fixed_t)
            continue
        rvec = params[idx:idx+3]
        t = params[idx+3:idx+6]
        R = rodrigues_to_matrix(rvec)
        camera_poses[cam_idx] = (R, t)
        idx += 6
    points_3d = params[idx:].reshape(-1, 3)
    return camera_poses, points_3d


def compute_residuals(params, n_cameras, n_points, camera_indices,
                      observations, K, fixed_cam_idx, fixed_R, fixed_t):
    """Compute reprojection error for every observation."""
    camera_poses, points_3d = unpack_params(
        params, n_cameras, n_points, camera_indices,
        fixed_cam_idx, fixed_R, fixed_t
    )
    residuals = np.zeros(2 * len(observations))
    for i, (cam_idx, pt_idx, observed_2d) in enumerate(observations):
        R, t = camera_poses[cam_idx]
        X = points_3d[pt_idx]
        projected = project_point(X, R, t, K)
        residuals[2*i]     = projected[0] - observed_2d[0]
        residuals[2*i + 1] = projected[1] - observed_2d[1]
    return residuals


def build_sparse_jacobian(observations, n_cameras, n_points, camera_indices, fixed_cam_idx):
    """
    Build a sparse Jacobian sparsity structure.
    
    Each observation (cam_i, point_j) only depends on:
      - camera_i's 6 params (R and t)
      - point_j's 3 params (X, Y, Z)
    
    By telling scipy this structure, it skips computing 
    derivatives for all the zeros. This is what makes BA fast.
    """
    n_obs = len(observations)
    n_free_cams = n_cameras - 1
    n_params = n_free_cams * 6 + n_points * 3

    cam_param_idx = {}
    pos = 0
    for cam_idx in sorted(camera_indices):
        if cam_idx == fixed_cam_idx:
            continue
        cam_param_idx[cam_idx] = pos
        pos += 6

    A = lil_matrix((2 * n_obs, n_params), dtype=int)

    for i, (cam_idx, pt_idx, _) in enumerate(observations):
        if cam_idx != fixed_cam_idx:
            cam_start = cam_param_idx[cam_idx]
            A[2*i,     cam_start:cam_start+6] = 1
            A[2*i + 1, cam_start:cam_start+6] = 1

        pt_start = n_free_cams * 6 + pt_idx * 3
        A[2*i,     pt_start:pt_start+3] = 1
        A[2*i + 1, pt_start:pt_start+3] = 1

    return A


def bundle_adjustment(camera_poses, all_3d_points, observations):
    """
    Run bundle adjustment with outlier filtering and sparse Jacobian.
    
    Fixes applied vs naive version:
    1. Filter outlier 3D points BEFORE optimization (reduces noise)
    2. Use sparse Jacobian structure (massive speedup)
    3. Use 'trf' method instead of 'lm' (supports sparse Jacobian)
    4. Limit iterations to 50 (prevents hanging)
    """
    K = config.K
    fixed_cam_idx = 0
    fixed_R, fixed_t = camera_poses[fixed_cam_idx]
    camera_indices = sorted(camera_poses.keys())
    n_cameras = len(camera_indices)

    # --- Step 1: Filter outlier 3D points ---
    median = np.median(all_3d_points, axis=0)
    dists = np.linalg.norm(all_3d_points - median, axis=1)
    threshold = np.percentile(dists, 95)
    valid_mask = dists < threshold

    old_to_new = {}
    new_idx = 0
    for old_idx in range(len(all_3d_points)):
        if valid_mask[old_idx]:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    filtered_points = all_3d_points[valid_mask]

    filtered_obs = []
    for cam_idx, pt_idx, obs_2d in observations:
        if pt_idx in old_to_new:
            filtered_obs.append((cam_idx, old_to_new[pt_idx], obs_2d))

    n_points = len(filtered_points)
    print(f"[Bundle Adjustment] Filtered to {n_points} points, {len(filtered_obs)} observations")

    # --- Step 2: Pack parameters ---
    x0 = pack_params(camera_poses, filtered_points, fixed_cam_idx)
    print(f"[Bundle Adjustment] {n_cameras} cameras, {n_points} points")
    print(f"[Bundle Adjustment] {len(x0)} parameters to optimize")

    initial_residuals = compute_residuals(
        x0, n_cameras, n_points, camera_indices,
        filtered_obs, K, fixed_cam_idx, fixed_R, fixed_t
    )
    initial_error = np.mean(np.abs(initial_residuals))
    print(f"[Bundle Adjustment] Initial mean reprojection error: {initial_error:.2f} pixels")

    # --- Step 3: Build sparse Jacobian ---
    J_struct = build_sparse_jacobian(filtered_obs, n_cameras, n_points, camera_indices, fixed_cam_idx)

    # --- Step 4: Optimize ---
    result = least_squares(
        compute_residuals,
        x0,
        args=(n_cameras, n_points, camera_indices,
              filtered_obs, K, fixed_cam_idx, fixed_R, fixed_t),
        jac_sparsity=J_struct,
        method='trf',
        verbose=2,
        max_nfev=50,
        x_scale='jac',
        ftol=1e-4,
        xtol=1e-4
    )

    # --- Step 5: Unpack ---
    optimized_poses, optimized_points = unpack_params(
        result.x, n_cameras, n_points, camera_indices,
        fixed_cam_idx, fixed_R, fixed_t
    )

    final_error = np.mean(np.abs(result.fun))
    print(f"[Bundle Adjustment] Final mean reprojection error: {final_error:.2f} pixels")
    print(f"[Bundle Adjustment] Improvement: {initial_error:.2f} → {final_error:.2f} pixels")

    return optimized_poses, optimized_points