"""Incremental Structure-from-Motion reconstruction pipeline."""

import argparse
import os
import time

import cv2 as cv
import numpy as np

from sift_extractor import SIFTExtractor
import config
from ransac import ransac_F
from geometry.essential import compute_essentialmatrix
from geometry.triangulation import triangulate_points, valid_triangulation_mask
from matcher import FeatureMatcher


def parse_args():
    parser = argparse.ArgumentParser(description="Structure from Motion")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument(
        "--n_features", type=int, default=6000, help="Max SIFT features per image"
    )
    parser.add_argument(
        "--match_neighbors",
        type=int,
        default=10,
        help="Recent registered cameras to match against",
    )
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument(
        "--cpu_cores",
        type=int,
        default=8,
        help="Workers for parallel feature extraction",
    )

    return parser.parse_args()


def load_images(folder):
    """Load readable images in filename order."""
    images = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        image = cv.imread(path)
        if image is not None:
            images.append((filename, image))

    print(f"Loaded {len(images)} images from '{folder}'")
    return images


def register_remaining_images(
    images,
    all_keypoints,
    matcher,
    K,
    point_map,
    camera_poses,
    all_3d_points,
    match_neighbors,
):
    """Register remaining cameras and triangulate new 3D points."""
    for new_idx in range(2, len(images)):
        t_img_start = time.time()
        print(f"\n--- Adding image {new_idx + 1}/{len(images)} ({images[new_idx][0]}) ---")

        pts_3d_for_pnp = []
        pts_2d_for_pnp = []
        seen_new_kp = set()
        seen_3d_points = set()
        pnp_associations = []
        cached_reg_matches = {}

        # Use the seed camera plus recent cameras for PnP correspondences.
        registered = sorted(
            camera_idx for camera_idx in camera_poses if camera_idx != new_idx
        )
        neighbors = [0] if 0 in registered else []
        neighbors.extend(camera_idx for camera_idx in registered if camera_idx != 0)
        neighbors = sorted(set(neighbors[:1] + neighbors[-match_neighbors:]))

        for reg_idx in neighbors:
            reg_matches, _, _ = matcher.get_matches(reg_idx, new_idx)
            cached_reg_matches[reg_idx] = reg_matches

            for kp_reg_idx, kp_new_idx in reg_matches:
                if kp_reg_idx not in point_map[reg_idx] or kp_new_idx in seen_new_kp:
                    continue

                point_3d_idx = point_map[reg_idx][kp_reg_idx]
                if point_3d_idx in seen_3d_points:
                    continue

                pts_3d_for_pnp.append(all_3d_points[point_3d_idx])
                pts_2d_for_pnp.append(all_keypoints[new_idx][kp_new_idx].pt)
                pnp_associations.append((kp_new_idx, point_3d_idx))
                seen_new_kp.add(kp_new_idx)
                seen_3d_points.add(point_3d_idx)

        print(
            f"  Found {len(pts_3d_for_pnp)} 3D↔2D correspondences "
            f"from {len(camera_poses)} registered cameras"
        )
        if len(pts_3d_for_pnp) < 6:
            print(f"   Not enough correspondences, skipping image {new_idx + 1}")
            continue

        pnp_points_3d = np.asarray(pts_3d_for_pnp, dtype=np.float64)
        pnp_points_2d = np.asarray(pts_2d_for_pnp, dtype=np.float64)

        # Estimate the new camera pose, then refine it using only RANSAC inliers.
        try:
            success, rvec, tvec, inliers = cv.solvePnPRansac(
                pnp_points_3d,
                pnp_points_2d,
                K,
                None,
                iterationsCount=5000,
                reprojectionError=15.0,
            )
            if (
                not success
                or inliers is None
                or len(inliers) < 15
                or len(inliers) / len(pts_3d_for_pnp) < 0.15
            ):
                print(f"   PnP failed for image {new_idx + 1}")
                continue

            inlier_indices = inliers.ravel().astype(np.intp)
            rvec, tvec = cv.solvePnPRefineLM(
                pnp_points_3d[inlier_indices],
                pnp_points_2d[inlier_indices],
                K,
                None,
                rvec,
                tvec,
            )
            R_new, _ = cv.Rodrigues(rvec)
            t_new = tvec.flatten()
            print(f"  PnP inliers: {len(inliers)}/{len(pts_3d_for_pnp)}")
        except Exception as exc:
            print(f"  PnP failed for image {new_idx + 1}: {exc}")
            continue

        camera_poses[new_idx] = (R_new, t_new)
        P_new = K @ np.hstack([R_new, t_new.reshape(3, 1)])

        # Link PnP inliers to existing tracks when they reproject cleanly.
        attached_points = set()
        for inlier_idx in inlier_indices:
            kp_idx, pt_idx = pnp_associations[int(inlier_idx)]
            X_cam = R_new @ all_3d_points[pt_idx] + t_new
            if X_cam[2] <= 1e-8:
                continue

            pixel = K @ X_cam
            error = np.linalg.norm(
                pixel[:2] / pixel[2] - all_keypoints[new_idx][kp_idx].pt
            )
            if error < 12.0:
                point_map[new_idx][kp_idx] = pt_idx
                attached_points.add(pt_idx)

        # Match against every registered camera to link tracks and triangulate.
        new_triangulated = []
        linked_observations = len(attached_points)
        triangulation_candidates = 0
        geometry_rejected = 0
        all_registered = sorted(
            camera_idx for camera_idx in camera_poses if camera_idx != new_idx
        )

        for reg_idx in all_registered:
            if reg_idx not in cached_reg_matches:
                cached_reg_matches[reg_idx], _, _ = matcher.get_matches(
                    reg_idx, new_idx
                )

            R_reg, t_reg = camera_poses[reg_idx]
            P_reg = K @ np.hstack([R_reg, t_reg.reshape(3, 1)])
            pts_reg_list = []
            pts_new_list = []
            pair_kp_indices = []

            for kp_reg_idx, kp_new_idx in cached_reg_matches[reg_idx]:
                if kp_new_idx in point_map[new_idx]:
                    continue

                if kp_reg_idx in point_map[reg_idx]:
                    pt_idx = point_map[reg_idx][kp_reg_idx]
                    if pt_idx >= len(all_3d_points) or pt_idx in attached_points:
                        continue

                    X_cam = R_new @ all_3d_points[pt_idx] + t_new
                    if np.isfinite(X_cam).all() and X_cam[2] > 1e-8:
                        pixel = K @ X_cam
                        error = np.linalg.norm(
                            pixel[:2] / pixel[2] - all_keypoints[new_idx][kp_new_idx].pt
                        )
                        if error < 12.0:
                            point_map[new_idx][kp_new_idx] = pt_idx
                            attached_points.add(pt_idx)
                            linked_observations += 1
                    continue

                pts_reg_list.append(all_keypoints[reg_idx][kp_reg_idx].pt)
                pts_new_list.append(all_keypoints[new_idx][kp_new_idx].pt)
                pair_kp_indices.append((kp_reg_idx, kp_new_idx))

            if not pts_reg_list:
                continue

            pts_reg_arr = np.array(pts_reg_list, dtype=np.float64)
            pts_new_arr = np.array(pts_new_list, dtype=np.float64)
            pts_3d_tri = triangulate_points(P_reg, P_new, pts_reg_arr, pts_new_arr)
            valid = valid_triangulation_mask(
                pts_3d_tri, pts_reg_arr, pts_new_arr, K, R_reg, t_reg, R_new, t_new
            )
            triangulation_candidates += len(valid)
            geometry_rejected += int(np.sum(~valid))

            for (kp_reg_idx, kp_new_idx), X, keep in zip(
                pair_kp_indices, pts_3d_tri, valid
            ):
                if (
                    not keep
                    or kp_reg_idx in point_map[reg_idx]
                    or kp_new_idx in point_map[new_idx]
                ):
                    continue

                new_3d_idx = len(all_3d_points) + len(new_triangulated)
                new_triangulated.append(X)
                point_map[reg_idx][kp_reg_idx] = new_3d_idx
                point_map[new_idx][kp_new_idx] = new_3d_idx

        if new_triangulated:
            all_3d_points = np.vstack([all_3d_points, np.array(new_triangulated)])

        print(f"  Linked {linked_observations} observations to existing points")
        print(
            f"  Triangulation: {triangulation_candidates} candidates, "
            f"{geometry_rejected} rejected by geometry"
        )
        print(
            f"  Triangulated {len(new_triangulated)} new points "
            f"(Total: {len(all_3d_points)}) "
            f"[Time: {time.time() - t_img_start:.2f}s]"
        )

    return all_3d_points


def main():
    args = parse_args()
    t_start = time.time()
    images = load_images(args.images)

    if len(images) < 2:
        print("Error: Need at least 2 images for SfM.")
        return

    # -------------------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------------------
    extractor = SIFTExtractor(n_features=args.n_features)
    t0 = time.time()
    all_keypoints, all_descriptors = extractor.parallel_extract(images, args.cpu_cores)
    print(f"Feature Extraction completed in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # Seed pair selection
    # -------------------------------------------------------------------------
    matcher = FeatureMatcher()
    matcher.set_features(all_keypoints, all_descriptors)

    print("\nSelecting best seed pair...")
    t0 = time.time()
    best_seed_count = 0
    seed_candidates = []
    best_seed_i, best_seed_j = 0, 1  # fallback
    n_imgs = len(images)
    # Skip adjacent frames; they often have too little baseline for a good seed.
    for skip in [2, 3, 4, 5]:
        for i in range(n_imgs - skip):
            j = i + skip
            des_i = all_descriptors[i]
            des_j = all_descriptors[j]
            if des_i is None or des_j is None or len(des_i) < 8 or len(des_j) < 8:
                continue
            matches, _, _ = matcher.get_matches(i, j)
            if len(matches) >= 30:
                seed_candidates.append((len(matches), i, j))
            if len(matches) > best_seed_count:
                best_seed_count = len(matches)
                best_seed_i, best_seed_j = i, j

    # Test the strongest matches geometrically, not just by match count.
    best_valid_count = 0
    for count, i, j in sorted(seed_candidates, reverse=True)[:12]:
        _, p1, p2 = matcher.get_matches(i, j)
        try:
            F_candidate, mask = ransac_F(p1, p2)
            a, b = p1[mask], p2[mask]
            _, R, t, pose_mask_candidate = cv.recoverPose(
                compute_essentialmatrix(F_candidate), a, b, config.K)
            t = t.ravel()
            X = triangulate_points(
                config.K @ np.column_stack((np.eye(3), np.zeros(3))),
                config.K @ np.column_stack((R, t)), a, b)
            valid = valid_triangulation_mask(
                X, a, b, config.K, np.eye(3), np.zeros(3), R, t)
            valid &= pose_mask_candidate.ravel().astype(bool)
            valid_count = int(valid.sum())
        except (ValueError, np.linalg.LinAlgError, cv.error):
            valid_count = 0
        print(
            f"  Seed candidate [{i}, {j}]: {count} matches, "
            f"{valid_count} geometrically valid points"
        )
        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_seed_count, best_seed_i, best_seed_j = count, i, j
    if best_valid_count < 30:
        print(
            "ERROR: No stable seed pair found. "
            "Check calibration and capture translated viewpoints."
        )
        return
    print(
        f"  Best seed: img[{best_seed_i}] & img[{best_seed_j}]  "
        f"({best_valid_count} valid points)  [{time.time() - t0:.1f}s]"
    )

    # Reorder so the seed pair is always at indices 0 and 1.
    remaining = [k for k in range(n_imgs) if k != best_seed_i and k != best_seed_j]
    remaining.sort(key=lambda k: min(abs(k - best_seed_i), abs(k - best_seed_j)))
    new_order = [best_seed_i, best_seed_j] + remaining
    images = [images[k] for k in new_order]
    all_keypoints = [all_keypoints[k] for k in new_order]
    all_descriptors = [all_descriptors[k] for k in new_order]
    matcher.set_features(all_keypoints, all_descriptors)

    # -------------------------------------------------------------------------
    # Initial reconstruction
    # -------------------------------------------------------------------------
    t0 = time.time()
    good_matches, pts1, pts2 = matcher.get_matches(0, 1)

    os.makedirs(args.output, exist_ok=True)

    cv_matches = [
        cv.DMatch(_queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0)
        for i, j in good_matches
    ]
    match_img = cv.drawMatches(
        images[0][1], all_keypoints[0],
        images[1][1], all_keypoints[1],
        cv_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    out_path = os.path.join(args.output, "matches_1_2.jpg")
    cv.imwrite(out_path, match_img)
    print(f"\nMatch visualization saved to: {out_path}")

    if len(good_matches) < 8:
        print("ERROR: Seed pair has fewer than 8 matches.")
        return
    F, inlier_mask = ransac_F(pts1, pts2)
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]

    # Recover the seed cameras' relative pose from the calibarted matches.
    K = config.K
    E = compute_essentialmatrix(F)
    n_inliers_pose, best_R, best_t, pose_mask = cv.recoverPose(
        E, pts1_inliers, pts2_inliers, K
    )
    best_t = best_t.flatten()

    print(
        f"\nInitial pose recovered: {n_inliers_pose}/{len(pts1_inliers)} "
        "inliers via recoverPose"
    )

    if best_R is None or n_inliers_pose < 6:
        print("ERROR: Could not recover initial pose. Try a different image pair as seed.")
        return

    # The first camera is assumed to be the cneter of the world coordinate frame.
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([best_R, best_t.reshape(3, 1)])

    points_3d = triangulate_points(P1, P2, pts1_inliers, pts2_inliers)
    print(f"Triangulated {len(points_3d)} initial 3D points in {time.time() - t0:.2f}s")

    camera_poses = {
        0: (np.eye(3), np.zeros(3)),
        1: (best_R, best_t),
    }

    # Maps each image keypoint to its 3D point index.
    point_map = [{} for _ in range(len(images))]
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]

    # recoverPose provides the cheirality mask; validate reprojection as well.
    pose_valid = pose_mask.ravel().astype(bool) & valid_triangulation_mask(
        points_3d, pts1_inliers, pts2_inliers, K,
        np.eye(3), np.zeros(3), best_R, best_t)
    
    print(f"Accepted {np.sum(pose_valid)} seed points after cheirality and reprojection points s")

    all_3d_points = []
    for idx, (kp1_idx, kp2_idx) in enumerate(inlier_matches):
        if idx < len(points_3d) and pose_valid[idx]:
            all_3d_points.append(points_3d[idx])
            point_map[0][kp1_idx] = len(all_3d_points) - 1
            point_map[1][kp2_idx] = len(all_3d_points) - 1

    all_3d_points = np.array(all_3d_points) if all_3d_points else np.zeros((0, 3))

    if len(all_3d_points) < 6:
        print("ERROR: Seed pair has too few geometrically valid points. No output overwritten.")
        return

    print(f"\n{'='*50}")
    print(
        f"Starting incremental SfM with {len(all_3d_points)} initial 3D points "
        f"across {len(images)} images"
    )
    print(f"{'='*50}")

    # -------------------------------------------------------------------------
    # Incremental reconstruction

    all_3d_points = register_remaining_images(
        images,
        all_keypoints,
        matcher,
        K,
        point_map,
        camera_poses,
        all_3d_points,
        args.match_neighbors,
    )

    # -------------------------------------------------------------------------
    # Bundle adjustment and export

    from bundle_adjustment import bundle_adjustment

    if len(all_3d_points) == 0:
        print("ERROR: No 3D points reconstructed. Check camera intrinsics and image overlap.")
        return

    observations = []
    for img_idx in camera_poses.keys():
        for kp_idx, pt3d_idx in point_map[img_idx].items():
            if pt3d_idx < len(all_3d_points):
                observed_2d = np.array(all_keypoints[img_idx][kp_idx].pt)
                observations.append((img_idx, pt3d_idx, observed_2d))

    print(f"\n{'='*50}")
    print(f"Running Bundle Adjustment...")
    print(f"{'='*50}")

    t_ba = time.time()
    camera_poses, all_3d_points = bundle_adjustment(camera_poses, all_3d_points, observations)
    print(f" Bundle Adjustment completed in {time.time() - t_ba:.2f}s")

    # Reject points that no longer project consistently after refinement. Each
    # retained point needs support from at least two geometrically valid views.
    from visualization.pointcloud_viz import export_ply

    # Keep points supported by at least two geometrically consistent views.
    valid_view_counts = np.zeros(len(all_3d_points), dtype=int)
    for img_idx, (R, t) in camera_poses.items():
        items = list(point_map[img_idx].items())
        if not items:
            continue
        kp_indices, pt_indices = np.array(items, dtype=int).T
        camera_points = all_3d_points[pt_indices] @ R.T + t
        projected = camera_points @ K.T
        observed = np.array([all_keypoints[img_idx][k].pt for k in kp_indices])
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = np.linalg.norm(projected[:, :2] / projected[:, 2:3] - observed, axis=1)
        good = np.isfinite(errors) & (camera_points[:, 2] > 1e-8) & (errors < 12.0)
        valid_view_counts[np.unique(pt_indices[good])] += 1
    final_mask = np.isfinite(all_3d_points).all(axis=1) & (valid_view_counts >= 2)
    final_points = all_3d_points[final_mask]
    if len(final_points) == 0:
        print("ERROR: No points pass final reprojection validation. No output overwritten.")
        return
    print(
        f"Final geometric validation: {len(final_points)}/{len(all_3d_points)} "
        "points retained"
    )

    # Sample one observed colour per track and convert OpenCV BGR to RGB.
    print("Sampling real image colors for point cloud...")
    all_colors = np.zeros((len(all_3d_points), 3), dtype=np.uint8)   # RGB
    color_assigned = np.zeros(len(all_3d_points), dtype=bool)

    for img_idx in camera_poses.keys():
        _, img = images[img_idx]
        img_h, img_w = img.shape[:2]
        for kp_idx, pt3d_idx in point_map[img_idx].items():
            if pt3d_idx >= len(all_3d_points) or color_assigned[pt3d_idx]:
                continue
            kp = all_keypoints[img_idx][kp_idx]
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if 0 <= x < img_w and 0 <= y < img_h:
                b, g, r = img[y, x]          # OpenCV stores BGR
                all_colors[pt3d_idx] = [r, g, b]
                color_assigned[pt3d_idx] = True

    final_colors = all_colors[final_mask]

    ply_path = os.path.join(args.output, "reconstruction.ply")
    export_ply(final_points, ply_path, colors=final_colors)

    # For world-to-camera X_cam = R X_world + t, the camera centre in world
    # coordinates is C = -R^T t. Save it for the interactive viewer.
    cam_centers = np.array([-R.T @ t for R, t in camera_poses.values()])
    cameras_path = os.path.join(args.output, "cameras.npz")
    np.savez(cameras_path, centers=cam_centers)

    # Produce both a quick static preview and an interactive HTML viewer.
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        final_points[:, 0], final_points[:, 1], final_points[:, 2],
        c=final_colors / 255.0, s=1, alpha=0.6
    )
    for cam_idx, (R_cam, t_cam) in camera_poses.items():
        C = -R_cam.T @ t_cam
        ax.scatter(*C, c='red', s=100, marker='^', depthshade=False)
        ax.text(C[0], C[1], C[2], f' Cam {cam_idx+1}', fontsize=8, color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(
        f"SfM Reconstruction ({len(final_points)} points, {len(camera_poses)} cameras)"
    )
    plot_path = os.path.join(args.output, "reconstruction_final.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Interactive 3D viewer (HTML — opens in browser automatically)
    from visualization.viewer import visualize
    html_path = os.path.join(args.output, "reconstruction_viewer.html")
    visualize(
        ply_path=ply_path,
        cameras_path=cameras_path,
        output_html=html_path,
        open_browser=True,
    )

    total_elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f" Pipeline Finished in {total_elapsed:.2f} seconds!")
    print(f" Static preview  → {plot_path}")
    print(f" PLY file        → {ply_path}")
    print(f" Interactive 3D  → {html_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
