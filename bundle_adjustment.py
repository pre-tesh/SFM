import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

import config


def rodrigues_to_matrix(rvec):
    """Convert 3-parameter Rodrigues vector to 3x3 rotation matrix."""
    return Rotation.from_rotvec(rvec).as_matrix()


def matrix_to_rodrigues(R):
    """Convert 3x3 rotation matrix to 3-parameter Rodrigues vector."""
    return Rotation.from_matrix(R).as_rotvec()


def project_point(X, R, t, K):
    """Project a 3D point using signed homogeneous division."""
    x = K @ (R @ X + np.asarray(t).reshape(3))
    depth = x[2]
    if abs(depth) < 1e-12:
        depth = -1e-12 if depth < 0 else 1e-12
    return x[:2] / depth


def pack_params(camera_poses, points_3d, fixed_cam_idx=0):
    """Pack all camera poses and 3D points into one 1D vector."""
    params = []
    for cam_idx in sorted(camera_poses.keys()):
        if cam_idx == fixed_cam_idx:
            continue
        R, t = camera_poses[cam_idx]
        rvec = matrix_to_rodrigues(R)
        params.extend(rvec)
        params.extend(np.asarray(t).reshape(3))
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


def _observation_arrays(observations, camera_indices):
    """Convert external camera IDs to dense slots once, before optimization."""
    camera_slots = {cam_idx: i for i, cam_idx in enumerate(camera_indices)}
    return (
        np.array([camera_slots[o[0]] for o in observations], dtype=np.intp),
        np.array([o[1] for o in observations], dtype=np.intp),
        np.asarray([o[2] for o in observations], dtype=np.float64).reshape(-1, 2),
    )


def _project_observations(rotations, translations, points, arrays, K):
    cam_slots, point_indices, _ = arrays
    camera_points = (
        np.einsum('nij,nj->ni', rotations[cam_slots], points[point_indices])
        + translations[cam_slots]
    )
    homogeneous = camera_points @ K.T
    # Avoid singular division during trial steps without turning negative depth positive.
    depth = homogeneous[:, 2]
    safe_depth = np.where(
        np.abs(depth) < 1e-12, np.where(depth < 0, -1e-12, 1e-12), depth
    )
    return homogeneous[:, :2] / safe_depth[:, None], camera_points[:, 2]


def compute_residuals(params, n_cameras, n_points, camera_indices,
                      observations, K, fixed_cam_idx, fixed_R, fixed_t,
                      observation_arrays=None, baseline_constraint=None):
    """Vectorized reprojection residuals; original positional arguments still work.

    Optional precomputed arrays avoid traversing observations at each solver step.
    baseline_constraint is (seed camera slot, initial baseline norm, weight).
    """
    arrays = observation_arrays
    if arrays is None:
        arrays = _observation_arrays(observations, camera_indices)
    fixed_slot = camera_indices.index(fixed_cam_idx)
    free_slots = np.delete(np.arange(n_cameras), fixed_slot)
    camera_params = params[:6 * len(free_slots)].reshape(-1, 6)
    rotations = np.empty((n_cameras, 3, 3))
    translations = np.empty((n_cameras, 3))
    if len(free_slots):
        rotations[free_slots] = Rotation.from_rotvec(camera_params[:, :3]).as_matrix()
        translations[free_slots] = camera_params[:, 3:]
    rotations[fixed_slot] = fixed_R
    translations[fixed_slot] = np.asarray(fixed_t).reshape(3)
    points = params[6 * len(free_slots):].reshape(n_points, 3)
    projected, _ = _project_observations(rotations, translations, points, arrays, K)
    residuals = (projected - arrays[2]).ravel()
    if baseline_constraint is not None:
        seed_slot, baseline_norm, weight = baseline_constraint
        fixed_center = -rotations[fixed_slot].T @ translations[fixed_slot]
        seed_center = -rotations[seed_slot].T @ translations[seed_slot]
        gauge_error = weight * (np.linalg.norm(seed_center - fixed_center) / baseline_norm - 1.0)
        residuals = np.append(residuals, gauge_error)
    return residuals


def build_sparse_jacobian(observations, n_cameras, n_points, camera_indices,
                          fixed_cam_idx, baseline_cam_idx=None):
    """Observation dependencies plus an optional seed-baseline constraint row."""
    n_obs = len(observations)
    free_cameras = [c for c in sorted(camera_indices) if c != fixed_cam_idx]
    n_free_cams = len(free_cameras)
    n_params = n_free_cams * 6 + n_points * 3
    cam_param_idx = {c: 6 * i for i, c in enumerate(free_cameras)}
    A = lil_matrix((2 * n_obs + int(baseline_cam_idx is not None), n_params), dtype=int)
    for i, (cam_idx, pt_idx, _) in enumerate(observations):
        if cam_idx != fixed_cam_idx:
            cam_start = cam_param_idx[cam_idx]
            A[2*i:2*i+2, cam_start:cam_start+6] = 1
        pt_start = n_free_cams * 6 + pt_idx * 3
        A[2*i:2*i+2, pt_start:pt_start+3] = 1
    if baseline_cam_idx is not None:
        cam_start = cam_param_idx[baseline_cam_idx]
        # For a non-origin fixed camera, the baseline depends on seed rotation too.
        A[-1, cam_start:cam_start+6] = 1
    return A.tocsr()


def _robust_loss_with_gauge(z):
    """Soft-L1 for image residuals, quadratic loss for the scale constraint."""
    root = np.sqrt(1.0 + z)
    rho = np.vstack((2.0 * (root - 1.0), 1.0 / root, -0.5 / root**3))
    rho[:, -1] = (z[-1], 1.0, 0.0)
    return rho


def bundle_adjustment(camera_poses, all_3d_points, observations):
    """Optimize valid multi-view tracks while preserving full point-array indices."""
    if not camera_poses or len(all_3d_points) == 0 or len(observations) == 0:
        return camera_poses, all_3d_points

    K = np.asarray(config.K, dtype=np.float64)
    points = np.asarray(all_3d_points, dtype=np.float64)
    usable_poses = {
        c: (np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3))
        for c, (R, t) in camera_poses.items()
        if np.all(np.isfinite(R)) and np.all(np.isfinite(t))
    }
    if len(usable_poses) < 2 or not np.all(np.isfinite(K)):
        return camera_poses, all_3d_points
    camera_indices = sorted(usable_poses)
    fixed_cam_idx = 0 if 0 in usable_poses else camera_indices[0]
    fixed_R, fixed_t = usable_poses[fixed_cam_idx]
    fixed_center = -fixed_R.T @ fixed_t
    # Camera 1 is the usual seed; with other camera IDs use the first nonzero baseline.
    baseline_cam_idx = None
    baseline_norm = 0.0
    for c in camera_indices:
        if c == fixed_cam_idx:
            continue
        R, t = usable_poses[c]
        baseline_norm = np.linalg.norm(-R.T @ t - fixed_center)
        if np.isfinite(baseline_norm) and baseline_norm > 1e-12:
            baseline_cam_idx = c
            break
    if baseline_cam_idx is None:
        print('[Bundle Adjustment] No nonzero seed baseline. Skipping BA.')
        return camera_poses, all_3d_points

    finite_points = np.all(np.isfinite(points), axis=1)
    valid_obs = []
    for cam_idx, pt_idx, observed_2d in observations:
        xy = np.asarray(observed_2d, dtype=np.float64)
        if (cam_idx in usable_poses
                and isinstance(pt_idx, (int, np.integer))
                and 0 <= pt_idx < len(points) and finite_points[pt_idx]
                and xy.shape == (2,) and np.all(np.isfinite(xy))):
            valid_obs.append((cam_idx, pt_idx, xy))
    if not valid_obs:
        return camera_poses, all_3d_points

    arrays = _observation_arrays(valid_obs, camera_indices)
    rotations = np.array([usable_poses[c][0] for c in camera_indices])
    translations = np.array([usable_poses[c][1] for c in camera_indices])
    projected, depths = _project_observations(rotations, translations, points, arrays, K)
    errors = np.linalg.norm(projected - arrays[2], axis=1)
    good = np.isfinite(errors) & (depths > 1e-12) & (errors < 12.0)
    valid_obs = [o for o, keep in zip(valid_obs, good) if keep]

    # Count distinct cameras, not duplicated measurements of the same point/view.
    point_views = {}
    for cam_idx, pt_idx, _ in valid_obs:
        point_views.setdefault(pt_idx, set()).add(cam_idx)
    kept_indices = np.array(sorted(p for p, views in point_views.items() if len(views) >= 2), dtype=np.intp)
    remap = {p: i for i, p in enumerate(kept_indices)}
    filtered_obs = [(c, remap[p], xy) for c, p, xy in valid_obs if p in remap]
    print(f'[Bundle Adjustment] Kept {len(kept_indices)} points, {len(filtered_obs)} observations')
    if not filtered_obs:
        return camera_poses, all_3d_points

    # Only optimize cameras with retained observations. Keep other poses unchanged.
    active_cameras = {c for c, _, _ in filtered_obs}
    if fixed_cam_idx not in active_cameras or baseline_cam_idx not in active_cameras:
        print('[Bundle Adjustment] Seed cameras have no retained tracks. Skipping BA.')
        return camera_poses, all_3d_points
    camera_indices = sorted(active_cameras)
    active_poses = {c: usable_poses[c] for c in camera_indices}
    n_cameras, n_points = len(camera_indices), len(kept_indices)
    arrays = _observation_arrays(filtered_obs, camera_indices)
    baseline_constraint = (camera_indices.index(baseline_cam_idx), baseline_norm, 1000.0)
    x0 = pack_params(active_poses, points[kept_indices], fixed_cam_idx)
    args = (n_cameras, n_points, camera_indices, filtered_obs, K,
            fixed_cam_idx, fixed_R, fixed_t, arrays, baseline_constraint)
    initial_residuals = compute_residuals(x0, *args)
    initial_error = np.mean(np.abs(initial_residuals[:-1]))
    sparsity = build_sparse_jacobian(
        filtered_obs, n_cameras, n_points, camera_indices, fixed_cam_idx, baseline_cam_idx
    )
    result = least_squares(
        compute_residuals, x0, args=args, jac_sparsity=sparsity,
        method='trf', tr_solver='lsmr',
        # SciPy supports callable losses; some stubs only declare string losses.
        loss=_robust_loss_with_gauge,  # type: ignore[arg-type]
        f_scale=5.0, max_nfev=100, x_scale='jac', ftol=1e-4, xtol=1e-4,
    )
    optimized_poses, optimized_points = unpack_params(
        result.x, n_cameras, n_points, camera_indices, fixed_cam_idx, fixed_R, fixed_t
    )

    # Remove numerical gauge drift exactly with a reprojection-invariant similarity.
    seed_R, seed_t = optimized_poses[baseline_cam_idx]
    final_baseline = np.linalg.norm(-seed_R.T @ seed_t - fixed_center)
    if not np.isfinite(final_baseline) or final_baseline <= 1e-12:
        print('[Bundle Adjustment] Degenerate optimized baseline. Keeping initial reconstruction.')
        return camera_poses, all_3d_points
    scale = baseline_norm / final_baseline
    optimized_points = fixed_center + scale * (optimized_points - fixed_center)
    for c, (R, t) in optimized_poses.items():
        if c != fixed_cam_idx:
            center = fixed_center + scale * (-R.T @ t - fixed_center)
            optimized_poses[c] = (R, -R @ center)

    final_error = np.mean(np.abs(result.fun[:-1]))
    print(f'[Bundle Adjustment] Mean reprojection error: {initial_error:.4f} -> {final_error:.4f} pixels; '
          f'{result.nfev} evaluations (status {result.status})')
    optimized_all_points = points.copy()
    optimized_all_points[kept_indices] = optimized_points
    output_poses = camera_poses.copy()
    output_poses.update(optimized_poses)
    # Preserve the original fixed-pose objects, including the caller's translation shape.
    output_poses[fixed_cam_idx] = camera_poses[fixed_cam_idx]
    return output_poses, optimized_all_points
