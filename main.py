import argparse
import cv2 as cv
import os

from features.sift_extractor import SIFTExtractor
from features.orb_detector import ORBExtractor
import config
from matching.ransac import ransac_F
from geometry.essential import compute_essentialmatrix, decompose_essentialmatrix

def parse_args():
    parser = argparse.ArgumentParser(description = "Structure from Motion")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--extractor", type=str, default="sift", choices=["sift", "orb"])
    parser.add_argument("--n_features", type=int, default=10000)
    parser.add_argument("--output", type=str, default="output")

    return parser.parse_args()

def load_images(folder):
    """
    Load all images from a folder.
    
    Returns:
        list of (filename, image) tuples, sorted by filename
    """
    images = []
    valid_extensions = (".jpg", ".jpeg", ".png")
    
    filenames = [filename for filename in os.listdir(folder) if filename.lower().endswith(valid_extensions)]
    filenames.sort()
    for filename in filenames:
        path = os.path.join(folder, filename)
        image = cv.imread(path)
        if image is not None:
            images.append((filename, image))
    
    print(f"Loaded {len(images)} images from '{folder}'")
    return images

def get_extractor(args):
    if args.extractor == "sift":
        return SIFTExtractor(n_features=args.n_features)
    elif args.extractor == "orb":
        return ORBExtractor(n_features=args.n_features)
    else:
        raise ValueError(f"use either 'sift' or 'orb': {args.extractor}")

def main():
    args      = parse_args()
    extractor = get_extractor(args)
    images    = load_images(args.images)

    print(f"\nUsing extractor: {args.extractor}")
    print("-" * 40)

    # Extract features for all images
    all_keypoints   = []
    all_descriptors = []

    for i, (filename, image) in enumerate(images):
        kp, des = extractor.extract(image)
        all_keypoints.append(kp)
        all_descriptors.append(des)
        print(f"[{i + 1}/{len(images)}] {filename} : {len(kp)} keypoints")

    # Match the first two images (we'll loop over all pairs later)
    from matching.matcher import FeatureMatcher

    matcher = FeatureMatcher()
    good_matches, pts1, pts2 = matcher.match_features(
        all_keypoints[0], all_descriptors[0],
        all_keypoints[1], all_descriptors[1]
    )

    # Visualize the matches and save to output folder
    os.makedirs(args.output, exist_ok=True)

    # Convert our (i, j) tuples to cv2.DMatch so drawMatches can use them
    cv_matches = [cv.DMatch(_queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0)
                  for i, j in good_matches]

    match_img = cv.drawMatches(
        images[0][1], all_keypoints[0],
        images[1][1], all_keypoints[1],
        cv_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    out_path = os.path.join(args.output, "matches_1_2.jpg")
    cv.imwrite(out_path, match_img)
    print(f"\n Match visualization saved to: {out_path}")

    F, inlier_mask = ransac_F(pts1, pts2)
    # Keep only inlier points for the next steps
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]
    print(f"\nFundamental Matrix F:")
    print(F)

    # --- Module 4: Essential Matrix ---
    E = compute_essentialmatrix(F)
    print(f"\nEssential Matrix E:")
    print(E)

    poses = decompose_essentialmatrix(E)

    # --- Module 5: Pick correct pose via cheirality check ---
    from geometry.essential import check_cheirality
    from geometry.triangulation import triangulate_points
    import numpy as np

    print("\nCheirality check (which pose puts points in front of both cameras?):")

    best_count = 0
    best_R = None
    best_t = None

    for i, (R, t) in enumerate(poses):
        count = check_cheirality(R, t, pts1_inliers, pts2_inliers)
        print(f"  Pose {i+1}: {count}/{len(pts1_inliers)} points in front")
        if count > best_count:
            best_count = count
            best_R = R
            best_t = t

    print(f"\n Best pose: {best_count} points in front of both cameras")
    print(f"R =\n{best_R}")
    print(f"t = {best_t}")

    # --- Triangulate final 3D points with the best pose ---
    K = config.K
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([best_R, best_t.reshape(3, 1)])

    points_3d = triangulate_points(P1, P2, pts1_inliers, pts2_inliers)

    print(f"\n Triangulated {len(points_3d)} 3D points!")
    print(f"   X range: [{points_3d[:,0].min():.2f}, {points_3d[:,0].max():.2f}]")
    print(f"   Y range: [{points_3d[:,1].min():.2f}, {points_3d[:,1].max():.2f}]")
    print(f"   Z range: [{points_3d[:,2].min():.2f}, {points_3d[:,2].max():.2f}]")

    # --- Visualization: 3D scatter plot ---
    import matplotlib.pyplot as plt

    # Filter outliers: keep only points within a reasonable range
    median = np.median(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - median, axis=1)
    threshold = np.percentile(distances, 90)
    mask = distances < threshold
    filtered_points = points_3d[mask]

    print(f"\n Plotting {len(filtered_points)}/{len(points_3d)} points (filtered outliers)")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        filtered_points[:, 0],
        filtered_points[:, 1],
        filtered_points[:, 2],
        c=filtered_points[:, 2],
        cmap='viridis',
        s=2
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SfM 3D Reconstruction (2 views)')

    plot_path = os.path.join(args.output, "reconstruction_3d.png")
    plt.savefig(plot_path, dpi=150)
    print(f" 3D plot saved to: {plot_path}")
    plt.show()

    # --- Module 6: Incremental SfM ---
    camera_poses = {}
    camera_poses[0] = (np.eye(3), np.zeros(3))
    camera_poses[1] = (best_R, best_t)

    point_map = [{} for _ in range(len(images))]

    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]

    if len(inlier_matches) == 0:
        raise ValueError("No inlier matches between the first two images; cannot initialize SfM.")

    all_3d_points = []
    for idx, (kp1_idx, kp2_idx) in enumerate(inlier_matches):
        all_3d_points.append(points_3d[idx])
        point_map[0][kp1_idx] = idx
        point_map[1][kp2_idx] = idx

    all_3d_points = np.array(all_3d_points)

    print(f"\n{'='*50}")
    print(f"Starting incremental SfM with {len(all_3d_points)} initial 3D points")
    print(f"{'='*50}")

    from geometry.pnp import linear_pnp

    for new_idx in range(2, len(images)):
        print(f"\n--- Adding image {new_idx + 1} ---")

        prev_idx = new_idx - 1
        new_matches, _, _ = matcher.match_features(
            all_keypoints[prev_idx], all_descriptors[prev_idx],
            all_keypoints[new_idx], all_descriptors[new_idx]
        )

        pts_3d_for_pnp = []
        pts_2d_for_pnp = []

        for kp_prev_idx, kp_new_idx in new_matches:
            if kp_prev_idx in point_map[prev_idx]:
                point_3d_idx = point_map[prev_idx][kp_prev_idx]
                pts_3d_for_pnp.append(all_3d_points[point_3d_idx])
                pts_2d_for_pnp.append(all_keypoints[new_idx][kp_new_idx].pt)

        if len(pts_3d_for_pnp) < 6:
            print(f"   Not enough correspondences, skipping image {new_idx + 1}")
            continue

        pts_3d_for_pnp = np.array(pts_3d_for_pnp)
        pts_2d_for_pnp = np.array(pts_2d_for_pnp)

        print(f"  Found {len(pts_3d_for_pnp)} 3D↔2D correspondences for PnP")

        # Use OpenCV's RANSAC-based PnP for robust camera pose estimation
        # Our from-scratch linear_pnp (in geometry/pnp.py) works but has no
        # outlier rejection — one bad match ruins the pose. In incremental SfM,
        # errors compound across views, so robustness is critical here.
        K = config.K
        try:
            success, rvec, tvec, inliers = cv.solvePnPRansac(
                pts_3d_for_pnp, pts_2d_for_pnp, K, None,
                iterationsCount=10000, reprojectionError=5.0
            )
            if not success or inliers is None or len(inliers) < 6:
                print(f"  ⚠️ PnP failed for image {new_idx + 1}")
                continue
            R_new, _ = cv.Rodrigues(rvec)
            t_new = tvec.flatten()
            print(f"  PnP inliers: {len(inliers)}/{len(pts_3d_for_pnp)}")
        except Exception as exc:
            print(f"  PnP failed for image {new_idx + 1}: {exc}")
            continue

        camera_poses[new_idx] = (R_new, t_new)
        print(f"  Camera {new_idx + 1} registered")

        R_prev, t_prev = camera_poses[prev_idx]
        P_prev = K @ np.hstack([R_prev, t_prev.reshape(3, 1)])
        P_new = K @ np.hstack([R_new, t_new.reshape(3, 1)])

        new_triangulated = []
        for kp_prev_idx, kp_new_idx in new_matches:
            if kp_prev_idx in point_map[prev_idx]:
                continue

            pt_prev = np.array(all_keypoints[prev_idx][kp_prev_idx].pt, dtype=np.float64)
            pt_new = np.array(all_keypoints[new_idx][kp_new_idx].pt, dtype=np.float64)

            from geometry.triangulation import triangulate_point
            X = triangulate_point(P_prev, P_new, pt_prev, pt_new)

            if X is None:
                continue

            new_3d_idx = len(all_3d_points) + len(new_triangulated)
            new_triangulated.append(X)
            point_map[prev_idx][kp_prev_idx] = new_3d_idx
            point_map[new_idx][kp_new_idx] = new_3d_idx

        if new_triangulated:
            all_3d_points = np.vstack([all_3d_points, np.array(new_triangulated)])

        print(f"  📍 Triangulated {len(new_triangulated)} new 3D points")
        print(f"  📊 Total 3D points: {len(all_3d_points)}")

    # --- Module 7: Bundle Adjustment ---
    from optimization.bundle_adjustment import bundle_adjustment

    # Build the observation list from point_map
    # Each observation = (camera_index, 3d_point_index, observed_2d_pixel)
    observations = []
    for img_idx in camera_poses.keys():
        for kp_idx, pt3d_idx in point_map[img_idx].items():
            if pt3d_idx < len(all_3d_points):
                observed_2d = np.array(all_keypoints[img_idx][kp_idx].pt)
                observations.append((img_idx, pt3d_idx, observed_2d))

    print(f"\n{'='*50}")
    print(f"Running Bundle Adjustment...")
    print(f"{'='*50}")

    camera_poses, all_3d_points = bundle_adjustment(camera_poses, all_3d_points, observations)

    # --- Module 8: PLY Export ---
    from visualization.pointcloud_viz import export_ply

    # Filter outliers before export
    median = np.median(all_3d_points, axis=0)
    dists_from_median = np.linalg.norm(all_3d_points - median, axis=1)
    threshold = np.percentile(dists_from_median, 90)
    final_mask = dists_from_median < threshold
    final_points = all_3d_points[final_mask]

    ply_path = os.path.join(args.output, "reconstruction.ply")
    export_ply(final_points, ply_path)

    # --- Final Visualization ---
    print(f"\n{'='*50}")
    print(f"FINAL RESULT: {len(final_points)} 3D points from {len(camera_poses)} cameras")
    print(f"{'='*50}")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        final_points[:, 0],
        final_points[:, 1],
        final_points[:, 2],
        c=final_points[:, 2],
        cmap='viridis',
        s=1,
        alpha=0.6
    )

    for cam_idx, (R_cam, t_cam) in camera_poses.items():
        C = -R_cam.T @ t_cam
        ax.scatter(*C, c='red', s=100, marker='^', depthshade=False)
        ax.text(C[0], C[1], C[2], f'  Cam {cam_idx+1}', fontsize=8, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'SfM Reconstruction — After Bundle Adjustment ({len(final_points)} points)')

    plot_path = os.path.join(args.output, "reconstruction_final.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n✅ Final reconstruction saved to: {plot_path}")
    print(f"✅ PLY file saved to: {ply_path} (open in MeshLab)")
    plt.show()


if __name__ == "__main__":
    main()
